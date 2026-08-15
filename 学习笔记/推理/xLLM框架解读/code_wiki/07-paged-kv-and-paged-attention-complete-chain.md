# xLLM Paged KV 与 Paged Attention 全量解读：从显存容量到模型读写

> 本文是 `code_wiki` 的第七篇核心文章。
>
> [第二篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/02-scheduler-complete-chain.md:1)讲调度，[第四篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/04-chunked-prefill-complete-chain.md:1)讲 Chunked Prefill。
>
> 本文专门回答它们共同依赖的底层问题：KV Cache 怎样估算、分块、分配、写入、读取、共享、迁移和释放。
>
> 结论全部以当前工作区源码为准。

---
## 1. 先建立最终认知

Paged KV 不是一个孤立 Attention kernel。

它是贯穿以下层次的系统协议：

```text
模型配置
  -> KV 容量估算
  -> 物理 cache tensor 创建
  -> BlockManager 管理 block id
  -> Sequence 保存逻辑 block table
  -> Scheduler 做容量准入
  -> BatchInputBuilder 生成 slot mapping
  -> 模型把新 K/V 写入物理 slot
  -> Attention 根据 block table 读取完整历史
  -> 请求结束、Prefix Cache、PD 或 Host Offload 接管 block 生命周期
```

最重要的四个对象必须分开：

| 对象 | 含义 | 典型表示 |
|---|---|---|
| token 进度 | 已完成多少 token 的 cache | `kv_cache_tokens_num` |
| 逻辑 block | Sequence 的第几个 KV 页 | block table 的列号 |
| 物理 block | 全局池中的实际页 | `Block::id()` |
| 物理 slot | 某 token 在 cache tensor 中的位置 | `block_id * block_size + offset` |

例如：

```text
block_size = 4
Sequence token 0..8
block_table = [12, 7, 21]
```

其映射是：

```text
token 0 -> physical block 12, offset 0 -> slot 48
token 3 -> physical block 12, offset 3 -> slot 51
token 4 -> physical block 7,  offset 0 -> slot 28
token 8 -> physical block 21, offset 0 -> slot 84
```

逻辑历史连续，不要求物理地址连续。

这正是“分页”的含义。

Paged KV 主要解决：

- 不按最大长度为每个请求预留显存；
- Sequence 增长时不搬迁整段 KV；
- 请求任意加入、结束时可回收离散页；
- Prefix Cache 可共享完整页；
- Scheduler 可用统一 block 数做容量会计；
- PD、Host Offload 可按 block 搬运。

Paged KV 不直接解决：

- 模型每 token KV 本体过大；
- Attention 扫描长历史的计算复杂度；
- TP/CP/PD 的通信；
- 低精度 KV 的精度问题；
- KDA/SSM 递归状态的保存。

这些要分别由 GQA、MLA、线性/稀疏 Attention、量化和分布式协议解决。

---

## 2. 为什么连续 KV 分配不适合在线服务

### 2.1 按最大长度预留

假设服务最大上下文 32768，某请求实际只生成 200 token。

如果启动时就为它预留 32768 token KV：

```text
waste_ratio = 1 - 200 / 32768 ≈ 99.39%
```

大量短请求会让显存被“可能永远用不到的未来”占满。

### 2.2 按当前长度动态扩容

如果每次 Decode 都扩展连续 tensor：

```text
[K0..Kt] -> allocate larger tensor -> copy -> append Kt+1
```

则历史 KV 会被重复搬运。

Decode 本应只追加一个 token，却退化成随上下文长度增长的复制。

### 2.3 变长请求造成外部碎片

在线请求长度、到达和结束时间不同。

连续分配会留下许多大小不同的空洞。

即使总空闲显存足够，也可能找不到某请求需要的大连续区域。

Paged KV 把外部碎片问题转换为固定页分配。

只要 free list 里还有足够数量的页，就不关心它们是否相邻。

### 2.4 Paged KV 的代价

分页不是免费午餐。

它引入：

- 每条 Sequence 的 block table；
- 每 token 到物理页的间接寻址；
- 最后一页的内部碎片；
- block 引用计数和共享语义；
- Prefix Cache 驱逐；
- 多平台 metadata 转换；
- 更复杂的异常与并发安全。

因此 Paged KV 是用软件复杂度换显存利用率和服务灵活性。

---

## 3. 三种“分页”必须分清

### 3.1 逻辑 KV block

这是 Scheduler 和 Sequence 看到的单位。

默认 [KVCacheConfig](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:29) 中：

```text
block_size = 128 token
```

Sequence 长度 129 至少需要两个逻辑 block。

### 3.2 设备物理 page

普通 `BlockManagerImpl` 中，一个逻辑 block id 直接索引 cache tensor 第一维。

XTensor/VMM 路径还会把逻辑 block 映射到更底层的物理页池。

其默认 `phy_page_granularity_size` 是 2 MiB。

所以“128 token 的 KV block”和“2 MiB 物理页”不是同一个单位。

### 3.3 Attention kernel 的 tile/page

FlashInfer、FIA 或自定义 kernel 内部还会按 warp、tile、page 处理。

那是设备计算实现。

kernel tile 不等于 Scheduler 的 KV block。

调优时不能把三者的 size 混为一个参数。

---

## 4. Paged Attention 的数学要求

### 4.1 连续表示下的 Attention

对于当前 query `q_t`：

```text
score_i = q_t · k_i / sqrt(d)
prob    = softmax(mask(score))
o_t     = Σ_i prob_i v_i
```

历史索引为：

```text
i = 0 ... t
```

### 4.2 分页只改变地址函数

分页后 K/V 的逻辑内容不变。

只是读取地址改成：

```text
logical_block = floor(i / B)
offset        = i % B
physical_id   = block_table[logical_block]
K_i           = key_cache[physical_id, offset, ...]
V_i           = value_cache[physical_id, offset, ...]
```

因此正确实现应与连续 KV 数学等价。

### 4.3 Decode 与 Chunked Prefill 的区别

Decode 通常：

```text
q_len = 1
kv_len = history + 1
```

Chunked Prefill：

```text
q_len = current_chunk
kv_len = old_history + current_chunk
```

二者都读 paged history。

首块普通 Prefill 没有历史，可直接使用当前 dense K/V，之后仍把它们写入 paged cache。

### 4.4 写入必须先于读取

当前 token 的 query 应看到当前 token 的 K/V。

所以统一顺序是：

```text
计算当前 Q/K/V
  -> 按 new_cache_slots 写 K/V
  -> Attention 读取更新后的完整 cache
```

如果先 Attention 后写 cache，当前 chunk 内的 causal 语义会错误。

---

## 5. 当前源码地图

### 5.1 配置与容量

- [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:29)：block size、cache budget、dtype、Prefix、XTensor。
- [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:39)：每 token/每 block 字节与容量。
- [kv_cache_capacity.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_capacity.h:24)：估算结果契约。
- [kv_cache_shape.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:62)：模型几何到设备 tensor shape。
- [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:400)：收集最小可用内存并分配 cache。

### 5.2 Block 与生命周期

- [block.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.h:35)：BlockType、id、原子引用计数、hash。
- [block.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.cpp:27)：RAII copy/move/free。
- [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:50)：free list、padding block、Prefix Cache、增量分配。
- [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:34)：每个 DP rank 的管理器池。
- [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:116)：KV、SWA、C4、C128、LINEAR 复合管理。

### 5.3 Sequence 与 Batch

- [sequence_kv_state.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.h:31)：Sequence 的所有 cache 状态。
- [sequence_kv_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.cpp:184)：共享块挂载与逻辑进度。
- [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:671)：生成长度、slot、block table 与 FlashInfer CSR。
- [attention_metadata_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/common/attention_metadata_builder.cpp:46)：Batch 到 AttentionMetadata。

### 5.4 设备执行

- [attention.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/attention.cpp:39)：C++ NPU paged 写入和 FIA。
- [flashinfer_attention.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/flashinfer_attention.cpp:112)：CUDA FlashInfer prefill/decode。
- [npu_paged_attention.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:222)：Python NPU MHA/MLA。
- [flashinfer.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/flashinfer.py:106)：Python CUDA wrapper plan 与 execute。

---

## 6. 配置如何决定 cache

### 6.1 `block_size`

[KVCacheConfig](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:59) 默认：

```text
block_size = 128
```

它表示每个逻辑 KV block 容纳的 token 数。

它同时影响：

- Sequence 分配粒度；
- block table 长度；
- Prefix Cache 最小复用粒度；
- 最后一页浪费；
- PD 稳定传输边界；
- linear-state checkpoint 对齐约束；
- 部分 kernel 的 page size。

### 6.2 `max_cache_size`

值为 0 表示由可用设备内存计算。

大于 0 时作为 KV Cache 显式上限。

它不是整个进程显存上限。

### 6.3 `max_memory_utilization`

[LLMEngine](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:438) 从 worker 报告的 available memory 中扣除：

```text
buffer_memory = total_memory * (1 - utilization)
available_memory -= buffer_memory
```

随后再与 `max_cache_size` 取最小值。

所有 worker/rank 最终取最小可用容量。

任何一个 rank 被其他进程占用，都会压低整个并行组 KV block 数。

### 6.4 `kv_cache_dtype`

`auto` 跟随模型 dtype。

估算器还识别 int8/fp8 为 1 byte。

但运行时支持矩阵更窄。

当前 KV INT8 在 worker 中明确只支持 MLU，并排除 VLM 和 PD。

### 6.5 Prefix 与 XTensor

`enable_prefix_cache` 默认 true。

`enable_in_batch_prefix_cache` 默认 false。

Engine 构造 BlockManagerPool 时，若启用 XTensor，会强制关闭普通 Prefix Cache。

所以 XTensor 和普通 Prefix 共享不是简单叠加关系。

### 6.6 flags 与 JSON

[KVCacheConfig::initialize()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.cpp:140) 先读取 flags，再用 JSON 中已出现的字段覆盖。

排查配置时应查看最终 config，而不是只看启动命令。

---

## 7. KV 容量如何估算

### 7.1 普通 MHA/GQA 每 token 字节

[kv_slot_size()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:53) 对普通 Attention 使用：

```text
slot_size = 2 * cache_dtype_bytes * head_dim * local_kv_heads
```

`2` 表示 K 和 V。

总 block 字节还要乘：

```text
block_size * full_attention_layers
```

### 7.2 TP 为什么降低本 rank KV

`local_kv_heads` 通常按 TP world size 切分。

因此每 rank 保存自己的 KV head shard。

但 GQA 的 KV head 数少于 TP size 时，具体映射可能复制或至少保留一个本地 head。

不能仅用 `global_heads / TP` 做无条件估算。

### 7.3 MLA 每 token 字节

MLA 路径使用：

```text
slot_size = bytes * (kv_lora_rank + qk_rope_head_dim)
```

它保存压缩 latent 和位置分量。

不保存普通每个 KV head 的完整 K/V。

这就是 Kimi/DeepSeek MLA 大幅降低长上下文 KV 的根本原因。

### 7.4 Indexer Cache

Lightning Indexer 还要增加：

```text
index_slot_size = dtype_bytes * index_head_dim
```

若 INT8 indexer：

```text
1 * index_head_dim + 4-byte per-token scale
```

Indexer 层数可能少于 full-attention 层数。

### 7.5 量化 scale

普通 KV 低精度 scale 估算为：

```text
2 * sizeof(float) * local_kv_heads
```

MLA 为一个 float scale slot。

所以低精度本体减半并不代表总 cache 精确减半。

### 7.6 线性状态也占同一预算

混合模型会从 KV cache budget 中为 conv/SSM state 预留容量。

Prefix Cache 开启时，自动策略约把 47% 预算分配给 linear-state slots。

这意味着 Kimi K3 的 4 GiB “KV cache”不一定全部变成 MLA pages。

### 7.7 block 数最终公式

标准全 Attention 可近似：

```text
n_blocks = floor(
  available_cache_bytes /
  bytes_per_full_cache_block
)
```

实际实现还扣除：

- linear-state cache；
- indexer cache 与 scale；
- RDMA padding；
- speculative state；
- 平台布局对齐。

### 7.8 为什么取所有 worker 最小值

并行模型每个 rank 必须为相同逻辑请求提供可兼容 block table。

如果 rank 0 有 1000 blocks，rank 1 只有 800，调度到第 801 block 时 rank 1 会失败。

因此 Engine 采用 worker 可用容量下界。

这是分布式正确性，不是保守浪费。

---

## 8. KVCacheCapacity 是什么

[KVCacheCapacity](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_capacity.h:24) 不是 tensor。

它是容量规划结果，包含：

```text
n_blocks
cache_size_in_bytes
block_size
slot_size
index_slot_size
scale_slot_size
linear_slot_size
num_linear_state_blocks
num_full_attention_layers
num_linear_attention_layers
```

DeepSeek V4 还包含：

```text
swa_count
c4_count
c128_count
```

Engine 先得到 Capacity，再构造 KVCacheShape。

这是“能放多少”与“tensor 长什么样”的分界。

---

## 9. KVCacheShape 如何把容量变成 tensor

### 9.1 普通默认布局

[init_key_cache_shape()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:269) 生成：

```text
[n_blocks, block_size, local_kv_heads, head_dim]
```

V cache 同形。

### 9.2 MLU/ILU 转置布局

这些平台把维度 1、2 交换：

```text
[n_blocks, local_kv_heads, block_size, head_dim]
```

因此不能把某平台 tensor shape 直接套到另一个 backend。

### 9.3 MLA 默认布局

MLA key cache：

```text
[n_blocks, block_size, 1, kv_lora_rank]
```

MLA value/rope cache：

```text
[n_blocks, block_size, 1, qk_rope_head_dim]
```

代码中的 `key/value` 名称在 MLA 语义下更接近 `nope/rope` 两份 cache。

### 9.4 NPU NZ 布局

部分 DeepSeek MLA NPU 路径使用 16 对齐 NZ 形态：

```text
[n_blocks, ceil(dim/16), block_size, 16]
```

这是物理布局变换，不改变 block id 的逻辑意义。

### 9.5 MLU/DCU MLA 合并布局

这些 backend 可把 latent 与 rope dim 合入 key cache 最后一维，并删除 value cache。

所以“Paged KV 一定有 K tensor 和 V tensor”并不成立。

### 9.6 Linear state shape

Conv state：

```text
[num_linear_state_blocks, conv_state_len, projected_dims]
```

SSM state：

```text
[num_linear_state_blocks * checkpoint_stride,
 local_v_heads, key_dim, value_dim]
```

它们以 slot id 管理，不按 token page 逐步增长。

---

## 10. Worker 怎样真正创建 KV tensor

Engine 通过 worker RPC 分发同一份 KVCacheShape。

普通角色调用：

```text
allocate_kv_cache_async(shape)
```

PD 角色调用：

```text
allocate_kv_cache_with_transfer_async(shape)
```

Worker 还会检查：

- linear attention 与 lighting indexer 不能同时存在；
- KV quant backend/角色限制；
- grouped cache 的 PD 组合限制；
- model layer type 到 cache layer 的映射；
- sleep mode/XTensor allocator。

每个模型层通常持有一个 `KVCache` 视图。

所有层共享同一套物理 block id 空间。

也就是说 block id 7 在每个 full-attention 层都代表该 Sequence 的同一 token 范围。

只是每层的 K/V 内容不同。

---

## 11. Block 对象不保存 K/V 数据

[Block](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.h:88) 只保存：

```text
id
size
atomic ref_count pointer
owning manager
prefix hash
```

K/V 数据在大 tensor 中。

`Block` 是物理页的 RAII 句柄。

### 11.1 copy 的含义

copy constructor 不复制 KV。

它复制 id/hash/manager，并增加共享 refcount。

因此 Prefix Cache 或 beam fork 可以便宜地共享同一物理页。

### 11.2 move 的含义

move 把句柄所有权转移，并把源对象置为 invalid。

不会调整 refcount。

### 11.3 析构的含义

析构减少 refcount。

最后一个句柄消失时，通过 owning manager 把 id 归还 free list。

### 11.4 为什么 refcount 是原子的

源码注释明确：PD scheduler、prefill thread pool 和异步 teardown 可能跨线程复制/销毁 Block alias。

原子计数避免同一物理页被并发错误释放。

### 11.5 manager 指针为何重要

不同 BlockType 可能来自不同 leaf manager。

析构必须回到正确的 pool。

并发 wrapper 还会把 manager 重定向到带锁封装。

---

## 12. BlockManagerImpl 的 free list

### 12.1 初始化

[BlockManagerImpl 构造](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:50) 建立：

```text
free_blocks
usage_accounted_ids
num_free_blocks
num_used_blocks
optional PrefixCache
```

block ids 逆序压入，使第一次 pop 得到较小 id。

### 12.2 block 0 永久保留

构造器立即分配 id 0 作为 padding block。

`free(0)` 明确不归还。

原因是 padded block table 的 0 必须指向安全地址。

它不能成为用户真实 KV 页。

### 12.3 分配 N 个 block

[allocate()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:78) 先调用 `has_enough_blocks`。

成功后从 free list 末尾取 id。

`usage_accounted_ids` 防止同一 id 被重复计入 used。

### 12.4 Prefix Cache 驱逐优先于失败

free blocks 不足且 Prefix Cache 开启时：

```text
need_evict = requested - current_free
prefix_cache.evict(need_evict)
```

只有驱逐后仍不足才返回失败。

### 12.5 增量增长公式

[allocate_for_sequence()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:238)：

```text
held = sequence 当前持有 block 数
needed = ceil(num_tokens / block_size)
additional = max(0, needed - held)
```

只分配差额。

Decode 跨越 block 边界时通常只新增一页。

未跨边界则不分配。

---

## 13. 为什么现在是 CompositeBlockManager

经典模型只有一套 flat KV blocks。

当前 xLLM 还要统一管理：

```text
KV        普通/MLA paged cache
SWA       sliding-window cache
C4        4x compressed cache
C128      128x compressed cache
LINEAR    GDN/KDA recurrent state slot
EMBEDDING speculative embedding row
```

[BlockType](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.h:43) 固定了类型与导出顺序。

CompositeBlockManager 把每种资源交给一个 leaf。

### 13.1 Admission leaf

并非所有 leaf 都参与按 token 容量准入。

LINEAR 每 Sequence 只需一个 live slot。

EMBEDDING 也是单资源 slot。

把它们当 `block_size=1` 的 token 页会错误压低可用 token 容量。

### 13.2 Prefix-capable leaf

每种 leaf 是否支持 Prefix Cache 独立判断。

PD Decode 角色跳过 SWA 和 LINEAR prefix restore。

普通 KV、C4、C128 仍可参与。

### 13.3 创建 leaf

[build_composite_leaves()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:116) 根据模型和配置构建：

- flat BlockManagerImpl；
- SlidingWindowBlockManager；
- LinearStateBlockManager；
- XTensorBlockManagerImpl；
- 需要时的并发 wrapper。

### 13.4 为什么要固定 multi-block 导出顺序

worker 的 DSA group infos 要与 scheduler block tables 对齐。

源码固定：

```text
[SWA, C4, C128]
```

不能依赖 `std::map` 或配置遍历顺序。

否则 rank/kernel 会把 C4 页误读为 SWA 页。

---

## 14. BlockManagerPool 与 DP

每个 DP rank 有独立 CompositeBlockManager。

[get_dp_rank()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:116) 的规则是：

```text
Sequence 已绑定 dp_rank -> 继续使用
否则 -> 选择 free blocks 最多的 manager
```

绑定后不能随意换 DP rank。

因为其 block ids 只在该 DP 副本的物理 cache 中有效。

### 14.1 新 Sequence 分配顺序

[BlockManagerPool::allocate()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:173) 执行：

```text
判断是否完全空
  -> 首次尝试 Prefix 共享
  -> 处理 beam copy-on-write
  -> Composite 各 leaf 增量增长
  -> 失败时仅对 started_empty 做完整回滚/reset
```

Prefix attach 必须在普通增长前。

否则会先分配并计算一份本可共享的 KV。

### 14.2 为什么 `started_empty` 看所有类型

DeepSeek V4 没有 `BlockType::KV`，只有 SWA/C4/C128。

如果只检查 KV 是否为空，每次增长都会误判为新请求。

失败时还可能把已有多池 block 全部释放。

当前源码用 `has_any_blocks()` 避免该问题。

### 14.3 Beam Search copy-on-write

beam 子序列可共享父序列历史 block。

当它们要写同一个最后部分 block 时，必须新分配一页并复制旧页。

Pool 记录：

```text
src_block_id -> dst_block_id
```

Worker 在写新 token 前完成 swap/copy。

这与操作系统 fork 后的 copy-on-write 类似。

---

## 15. SequenceKVState 是逻辑真相

KV tensor 不知道哪个请求拥有哪页。

这个关系保存在 [KVCacheState](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.h:31)。

主要字段包括：

```text
composite_blocks[BlockType]
num_owned_shared_blocks
num_cached_blocks
kv_cache_tokens_num
transfer cursors
beam source blocks
linear restore/checkpoint state
```

### 15.1 blocks 与 cached tokens 不是一回事

Sequence 可以已经分配完整 Prompt footprint 的 blocks，但只有部分 token 真正计算完成。

所以：

```text
allocated_capacity != kv_cache_tokens_num
```

Chunked Prefill 正依赖这个差异。

### 15.2 `current_max_tokens_capacity`

普通 KV：

```text
num_blocks(KV) * block_size
```

DeepSeek V4 则取 C4/C128 两种增量组容量的最小值。

SWA ring 不参与线性容量。

因为 token 进度可以超过 sliding window 的物理 ring 容量。

### 15.3 cache slot 公式

`cache_slots(type, start, end)` 对逻辑 token 区间生成：

```text
slot = blocks[type][token / B].id * B + token % B
```

返回长度必须等于 `end - start`。

它就是 Batch 的 `new_cache_slots` 来源。

### 15.4 精确相同 Prompt 的特殊处理

[add_shared_blocks()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.cpp:184) 若共享 token 数恰好等于当前全部 token，会退回最后一个完整 block。

原因是模型至少还需处理一个 token 才能产生下一 token logits。

源码会：

```text
shared--
pop_back()
kv_cache_tokens_num = 前一个完整 block 边界
```

这看似“少命中一块”，实际保证 forward 不为空。

---

## 16. Scheduler 怎样使用 block 容量

Scheduler 的 token budget 与 KV block budget 是两套约束。

```text
token budget: 本 step 最多算多少 query token
block budget: Sequence 现在和未来是否有物理 cache 页
```

### 16.1 新请求准入

普通 scheduler 会调用 BlockManagerPool 为目标 token 数分配。

当前 Chunked Prefill 新请求可按完整 Prompt footprint 做 admission。

这样避免先算半个 Prompt，随后因 KV 容量不足反复抢占重算。

### 16.2 Decode 增长

Decode 每步 token 数通常加 1。

只有当新长度跨越 block 边界时，`allocate_for_sequence` 才需要新页。

例如 `B=128`：

```text
len 127 -> 128: 仍在第 1 页
len 128 -> 129: 需要第 2 页
```

### 16.3 分配失败

流程通常先驱逐 Prefix Cache 冷页。

仍不足时，普通调度可抢占 Sequence，释放其 blocks，稍后重新 Prefill。

Zero-Eviction 则在准入前模拟未来增长，尽量不进入此状态。

### 16.4 为什么 block utilization 不等于 token utilization

10 条每条 1 token 的 Sequence 会占 10 个 block。

若 `B=128`：

```text
token payload utilization = 10 / (10*128) = 0.78%
block utilization = 10 / total_blocks
```

短请求高并发时内部碎片最明显。

---
## 17. BatchInputBuilder 如何把逻辑页变成模型输入

Scheduler 选好 Sequence 并确保容量后，BatchInputBuilder 才能生成设备输入。

[process_single_sequence()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:671) 读取：

```text
n_tokens
n_kv_cache_tokens
allowed_max_tokens
```

计算：

```text
q_seq_len = min(n_tokens - n_kv_cache_tokens, allowed)
seq_len   = n_kv_cache_tokens + q_seq_len
```

随后先检查：

```text
current_max_tokens_capacity >= seq_len
```

也就是说 Builder 不负责临时救火分配。

它要求 Scheduler/BlockManager 已兑现容量承诺。

### 17.1 为什么保存旧 cached 数

`kv_cache_tokens_nums` 保存本 step 开始前的历史长度。

模型用它区分：

- 首次 Prefill；
- 后续 Chunked Prefill；
- Decode；
- 线性状态是否有 initial state。

### 17.2 `setup_kv_cache_info`

[setup_kv_cache_info()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:961) 首先把逻辑进度增加 `q_seq_len`。

然后导出 block metadata。

正常 flat KV 路径生成：

```text
new_cache_slots
block_tables
paged_kv_indices
paged_kv_indptr
paged_kv_last_page_len
write_block_ids
```

### 17.3 `new_cache_slots`

它只覆盖本轮新增 token：

```text
[old_cached, seq_len)
```

长度等于所有 Sequence 的 `q_seq_len` 之和。

它告诉 reshape/scatter kernel 当前 K/V 写到哪里。

### 17.4 `block_tables`

它覆盖 Sequence 当前持有的完整逻辑历史。

每行是一个 Sequence。

每列是逻辑 block index。

行长不足 batch 最大值时用 block 0 padding。

### 17.5 FlashInfer CSR 表达

同一组页还会展平成：

```text
paged_kv_indices       所有 Sequence 的物理 block id 串联
paged_kv_indptr        每条 Sequence 在 indices 中的起止
paged_kv_last_page_len 每条 Sequence 最后一页有效 token 数
```

这是 CSR 风格的 ragged page 描述。

### 17.6 `last_page_len`

源码使用：

```text
seq_len % B == 0 ? B : seq_len % B
```

整除时最后一页是满页，不能记录为 0。

否则 kernel 会把它理解为空页。

### 17.7 `write_block_ids`

Builder 从：

```text
kv_cache_block_idx = old_cached / block_size
```

开始把后续 block id 加入写集合。

Graph/并发路径可用它判断本轮哪些物理页会被修改。

---

## 18. 一个完整 metadata 算例

设：

```text
block_size = 4

A: old_cached=5, q_len=1, block_table=[12,7]
B: old_cached=2, q_len=3, block_table=[3,19]
```

A 的新 token 是逻辑位置 5：

```text
logical block = 1
offset = 1
slot = 7*4+1 = 29
```

B 的新 token 是位置 2、3、4：

```text
position 2 -> block 3, offset 2 -> slot 14
position 3 -> block 3, offset 3 -> slot 15
position 4 -> block 19, offset 0 -> slot 76
```

最终：

```text
new_cache_slots = [29, 14, 15, 76]

block_tables =
  A [12, 7]
  B [ 3,19]

paged_kv_indices = [12,7,3,19]
paged_kv_indptr  = [0,2,4]
paged_kv_last_page_len = [2,1]
```

模型产生 4 行新 K/V。

scatter kernel 按 `[29,14,15,76]` 写入。

Attention 再分别读取 A 的 `[12,7]` 与 B 的 `[3,19]`。

两条 Sequence 即使物理页交错也不会互相可见。

---

## 19. AttentionMetadata 是统一桥梁

[build_attention_metadata()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/common/attention_metadata_builder.cpp:46) 将 `ModelInputParams` 转成 Layer 统一视图。

关键字段：

```text
slot_mapping
block_table
q_seq_lens
kv_seq_lens
kv_cache_tokens_nums
is_prefill
is_chunked_prefill
```

### 19.1 NPU 长度不是 cumulative

BatchInputBuilder 在 NPU 上保存逐 Sequence 长度：

```text
[len_A, len_B, ...]
```

### 19.2 CUDA/MLU/DCU 部分路径使用 cumulative

这些平台可保存：

```text
[0, len_A, len_A+len_B, ...]
```

字段名字相同不代表布局相同。

backend builder 负责转换。

### 19.3 MIXED 也必须使用 paged history

同批包含 Decode 和 Chunked Prefill 时，metadata 标记 chunked 语义。

因为所有行都要通过各自 block table 读取历史。

### 19.4 padding row

Decode Graph 可能把 batch padding 到 bucket。

padding row 的 block table 指向保留的 block 0。

slot mapping 也必须避免写入用户页。

---

## 20. C++ NPU Paged Attention

[AttentionImpl::forward()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/attention.cpp:39) 的统一入口先处理 cache 写入。

大致顺序：

```text
reshape q/k/v
  -> reshape_paged_cache(k, v, slot_mapping)
  -> 根据 forward type 选择 prefill/decode
```

### 20.1 首次 Prefill

没有旧历史时，可直接把当前 dense K/V 交给 Prefill Attention。

同时 K/V 已写入 paged cache，为后续 Decode 准备。

### 20.2 Chunked Prefill

Q 只包含当前 chunk。

K/V 参数改为整份 cache tensor。

同时传：

```text
block_table
q lengths
total kv lengths
block_size
sparse_mode=3
causal mask
```

### 20.3 Decode

Decode Q 通常每条一行。

FIA 根据 block table 和总上下文读取历史。

不会先把离散 KV gather 成一份永久连续 tensor。

### 20.4 kernel 契约

NPU kernel 对 dtype、head 数、layout、block size 和 length dtype 有约束。

分页逻辑正确不代表任何 shape 都能进入同一个 kernel。

不满足条件必须走 fallback 或做 padding。

---

## 21. CUDA FlashInfer Paged Attention

[FlashInferAttentionImpl::forward()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/flashinfer_attention.cpp:112) 同样先写 cache。

### 21.1 Ragged Prefill

首块 Prefill 可使用 ragged dense K/V wrapper。

这样省去对刚写入 page 的再次间接读取。

### 21.2 Paged Chunked Prefill

[chunked_prefill_forward()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/flashinfer_attention.cpp:224) 使用：

```text
query
k_cache
v_cache
paged_kv_indptr
paged_kv_indices
paged_kv_last_page_len
qo_indptr
```

wrapper 会先更新 plan，再运行 batch paged prefill。

### 21.3 Decode wrapper

Decode 使用 BatchDecodeWithPagedKVCache。

plan 与 run 分离。

shape、页表或 batch 改变时需要更新 plan。

### 21.4 workspace

FlashInfer 需要 workspace buffer。

所以 PagedAttention 内存不只有 K/V tensor。

workspace、plan metadata 和临时输出也要计入峰值。

---

## 22. Python backend 的当前实现边界

### 22.1 Python CUDA FlashInfer

[FlashInferBackend.prepare()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/flashinfer.py:106) 根据阶段准备 ragged/paged/decode wrapper。

[execute()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/flashinfer.py:235)：

```text
PREFILL         -> current dense K/V
CHUNKED_PREFILL -> paged cache tuple
DECODE          -> paged cache tuple
```

其语义与 C++ 路径一致。

### 22.2 Python NPU 通用 MHA

[NpuPagedAttentionBackend.execute()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:222) 会先写 paged cache。

但当前 `is_prefill or is_chunked_prefill` 都进入 dense `_prefill(q,k,v)`。

该函数使用当前输入 K/V，并把 KV length 设为 Q length。

因此通用 Python NPU MHA 后续 chunk 的历史读取存在已知边界。

这不代表所有 NPU paged 路径都错误。

C++ NPU 普通 Attention 正确使用 paged history。

Python Kimi MLA 也有专门 paged chunked/decode 实现。

### 22.3 为什么必须按模型路径验收

“框架有 block table”不证明模型实际消费了它。

验证必须追到最终 backend 调用参数。

尤其要看 K/V 来自：

```text
当前 dense tensor
还是
完整 paged cache
```

---

## 23. Kimi K3 的 Paged MLA

Kimi K3 的 Full Attention 不是普通 MHA。

它保存：

```text
nope_cache: kv latent
rope_cache: positional key slice
```

当前五层配置的已确认形状：

```text
[27625, 128, 1, 512]
[27625, 128, 1, 64]
```

### 23.1 为什么没有普通 V cache

MLA 通过低秩 latent 和权重吸收重构 Attention。

逻辑上的 V 信息包含在 latent 中。

第二份 cache 保存 RoPE/位置相关部分。

### 23.2 写入路径

[execute_mla()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:249) 先调用 `reshape_paged_cache` 写入当前 latent/PE。

然后按阶段选择：

```text
dense prefill
dense chunked prefill with pages
dense decode with pages
```

### 23.3 TP4 head padding

Kimi 全局 96 query heads。

TP4 每 rank 24 heads。

当前 FIA 路径要求支持的 head bucket，代码会把 24 临时补到 32，计算后裁回 24。

这与 KV block table 无关，但属于 paged MLA kernel 契约。

### 23.4 positional slice 不做 RoPE

Kimi K3 的 positional slice 参与 score，但按项目基准不执行 RoPE。

Paged cache 必须保存与模型数学一致的 slice。

不能因为字段叫 `rope_cache` 就机械套普通 DeepSeek RoPE 处理。

### 23.5 当前离线失败证据

最新离线 TP4 dump 已运行过前三个 KDA 层。

在第 4 层 Full MLA 的 `xllm_ops.reshape_paged_cache` 报 NPU `Invalid argument`。

这说明：

- block/capacity 初始化已完成；
- KDA state 路径已进入 forward；
- Full MLA 的实际 paged write metadata/kernel 仍有离线场景问题；
- 不能据在线短 smoke 宣称所有 Paged MLA 场景已经完成。

本文把它记录为当前验证边界，而不是 Paged KV 原理缺失。

---

## 24. KDA 为什么不是 Paged KV

Kimi K3 的 KDA 层不保存逐 token K/V 页面。

它保存：

```text
conv_state
recurrent_state
```

每条活跃 Sequence 使用一个 LINEAR live slot。

### 24.1 同一请求的双缓存系统

Kimi Sequence 同时持有：

```text
MLA layers -> BlockType::KV pages
KDA layers -> BlockType::LINEAR slot
```

Scheduler 的完整容量必须同时满足两类资源。

### 24.2 token 边界一致

若 `kv_cache_tokens_num=4096`：

- MLA pages 必须代表 `[0,4096)`；
- KDA recurrent/conv state 也必须代表同一前缀。

只恢复一类会让模型层间语义错位。

### 24.3 Prefix Cache checkpoint

普通 KV 可按完整 block hash 共享。

KDA 必须在相同 token 边界保存 linear-state checkpoint。

因此 linear prefix cache 要求 chunk size 与 block size 对齐。

### 24.4 Speculative 状态

多 token speculative verify 可能需要多份临时 SSM state。

容量估算器把 `num_speculative_tokens+1` 纳入 linear slot size。

候选拒绝时不能把未接受 state 提交为正式历史。

---

## 25. DeepSeek V4 多池分页

DeepSeek V4 不使用单一 `BlockType::KV`。

当前抽象为：

```text
SWA   sliding-window ring
C4    4x compressed token cache
C128  128x compressed token cache
```

### 25.1 为什么不是一张 block table

不同层使用不同压缩率和窗口语义。

同一 token 进度需要多张物理表。

BatchInputBuilder 导出 `multi_block_tables`。

### 25.2 SWA ring

物理 SWA 页数受 window 限制。

逻辑 token 进度可无限向前。

旧页会循环覆盖。

所以 SWA 页数不能用于 `current_max_tokens_capacity` 的线性上限。

### 25.3 C4/C128 容量

两者按压缩 token 比率增长。

Sequence 能继续的容量由最紧的增量组决定。

### 25.4 padding 值不同

普通 `block_tables` 用 0 padding。

`multi_block_tables` 先用 -1 表示 gap/无效位置，再由具体 metadata builder 处理。

### 25.5 Host Offload 当前不兼容

Engine 明确拒绝 DeepSeek V4 的 `host_blocks_factor > 1`。

当前 host offload 只查 flat `BlockType::KV`。

若不 fail loud，会分配大量 host cache 却实际不搬任何 SWA/C4/C128 页。

---

## 26. Prefix Cache 如何建立在 Paged KV 上

Prefix Cache 的缓存对象就是已完成的物理 blocks。

它不复制 K/V 内容。

而是保留 Block alias。

### 26.1 完整 block 才适合共享

部分 block 后续还会继续写。

若提前发布给其他请求，会发生读写竞态和语义变化。

所以稳定复用粒度通常是完整 block。

### 26.2 链式 hash

每个 block hash 包含前一 block hash。

相同局部 token 只有在相同完整前缀后才命中。

### 26.3 引用计数语义

一个缓存页可能同时有：

```text
PrefixCache owner
Sequence A owner
Sequence B owner
```

释放 A 不能回收物理页。

只有最后一个 owner 消失才回 free list。

### 26.4 used 与 free 的双账本

`usage_accounted_ids` 记录该页是否被活跃使用统计计入。

Prefix alias 的存在不总等于活跃 Sequence 使用。

因此 `num_used_blocks`、`num_free_blocks`、prefix cache block 数不是简单相加关系。

### 26.5 In-Batch publish

启用后，已准入 Prefill 的完整页可在同批后续请求中共享。

必须确保生产者已经完成写入，消费者才可读取。

详见第八篇 Prefix Cache 专题。

---

## 27. PD 分离中的物理页协议

P 和 D 拥有各自的物理 block pool。

P 的 block id 12 对 D 没有直接意义。

传输必须显式建立：

```text
P local block ids
  -> D remote block ids
```

### 27.1 D 先分配目标页

D 根据 Prompt 长度和已有 Prefix hit 分配自己的 blocks。

然后把目标 ids/地址告知 P。

### 27.2 P 构造 step transfer window

BatchInputBuilder 根据 `next_transfer_block_idx` 只发送新稳定页。

部分尾页因为下一 chunk 还会更新，可以重复传输。

### 27.3 Prefix 命中减少传输

D 若已有前 N 个共享 blocks，会把 cursor 推进到 N。

P 从第一个缺失页开始发送。

### 27.4 grouped cache

多池模型按 group id 分别传输。

每组有独立 cursor。

group id 是稳定协议标识，不应依赖本地 map 顺序。

### 27.5 KDA state

PD 对混合模型还要传 LINEAR state slot 内容。

即使 KV blocks 全部 Prefix hit，linear-state-only transfer 也不能丢。

### 27.6 内存峰值

交接期间 P 与 D 短时同时持有同一逻辑前缀的 KV。

还要加传输 buffer。

PD 不是全局瞬时 KV 内存减半。

---

## 28. Host Offload 与层级分页

`host_blocks_factor > 1` 会建立 host block pool。

设备页与 host 页仍是两套 id 空间。

HierarchyBlockManager 生成 D2H/H2D pair。

### 28.1 D2H 候选

通常要求：

- Prefix Cache 页；
- 完整计算；
- 不被活跃 Sequence 独占写入；
- host 有空闲页。

### 28.2 H2D restore

host 命中后必须在设备重新分配目标页。

再按 layer/batch copy K/V。

copy 完成前不能把目标页交给 Attention。

### 28.3 overlap

D2H 可与后续计算重叠。

H2D 若能根据 waiting queue 提前预取，也可隐藏一部分。

当前请求立即需要的 restore 是硬关键路径。

### 28.4 远端 store 的当前边界

配置层有 `enable_kvcache_store`。

但 Worker 当前 hierarchy transfer 初始化硬编码：

```text
enable_kvcache_store(false)
```

因此不能把 host offload 等同于远端全局 KV store 已贯通。

---

## 29. XTensor/VMM 的另一种分页

XTensor 使用物理页池支持 cache tensor 的虚拟地址管理。

Engine 可从 `PhyPagePool` 的总页数计算可用 cache bytes。

Composite leaf 选择 `XTensorBlockManagerImpl`。

### 29.1 为什么 block memory size 除以 2

构造 XTensor leaf 时：

```text
block_mem_size = block_size * slot_size / 2
```

这里 `slot_size` 包含 K+V，单个 tensor 页按一半计算。

### 29.2 Prefix Cache 被关闭

当前 Engine 对 XTensor 模式关闭普通 Prefix Cache。

因此不能假定 VMM 页能直接复用普通 BlockManagerImpl 的 hash/LRU 语义。

### 29.3 物理页预留

KV manager 初始化后会 reserve padding blocks 并启动预分配相关流程。

虚拟地址稳定有利于 Graph/跨进程注册。

但页表、物理池和回收协议更复杂。

---

## 30. Graph、Overlap 与 Paged KV

### 30.1 Graph 为什么喜欢大静态 cache tensor

KV cache 在模型生命周期内地址稳定。

每 step 变化的是：

```text
block_table
slot_mapping
lengths
```

这些可写入 persistent input buffer。

Graph 无需重新分配 K/V tensor。

### 30.2 Graph 不会固定 Sequence 页表

不同请求的 block ids 每轮变化。

Graph replay 前必须更新 metadata。

捕获旧 block table 并复用会读取别人的 KV。

### 30.3 Double buffer

schedule overlap 可能一边执行 batch N，一边准备 batch N+1 的 block tables。

metadata buffer 需要双份或明确 event。

不能覆盖设备仍在读取的那份。

### 30.4 block 分配可与设备执行重叠

CPU free-list/hash/metadata 工作可在上一 forward 时运行。

但最终页所有权必须在 next batch 提交前冻结。

### 30.5 PD/Host copy overlap

异步 copy stream 需要与 block 生命周期联动。

页仍在传输时：

- 源页不能释放/覆盖；
- 目标页不能读取；
- Scheduler 不能再次分配给其他请求。

---

## 31. 通信、内存与精度全账本

### 31.1 单设备通信

普通 Paged KV 不增加网络通信。

它增加设备内间接寻址和 page metadata 读取。

### 31.2 TP

每个 TP rank 保存自己的 head/latent shard。

逻辑 block ids 在 rank 间应一致。

Paged KV 本身不新增 TP collective。

Attention/Linear 原有 collective 仍存在。

### 31.3 CP/KV Split

上下文或 KV 分片到不同 ranks 后，page 只在本地覆盖部分历史。

Attention 需要额外 gather、ring 或 all-to-all 协议。

逻辑 block size 还可能乘 `kv_split_size_effective`。

### 31.4 主内存与网络

Host Offload 增加 D2H/H2D。

PD 增加实例间 KV 传输。

远端 store 还会增加查询和网络 buffer。

### 31.5 HBM 内存

主要项：

```text
n_blocks * block_size * per-token per-layer cache bytes
```

附加项：

- indexer/scale cache；
- linear state；
- block table tensors；
- workspace；
- Graph persistent buffers；
- communication buffers；
- padding block 和尾页碎片。

### 31.6 Host 内存

包括：

- Scheduler/Sequence block vectors；
- hash/LRU 元数据；
- pinned host KV；
- PD metadata；
- staging buffers。

### 31.7 精度

仅改变布局时数学等价。

允许的差异主要来自 backend kernel 累加顺序。

不允许的差异包括：

- block table 错行；
- slot offset 错一；
- `last_page_len` 错误；
- 当前 K/V 未先写入；
- cache dtype/scale 不匹配；
- Prefix key 语义不完整；
- KDA state 与 MLA token 边界不一致。

---

## 32. Block size 怎样调

### 32.1 小 block

优点：

- 尾页浪费小；
- Prefix 命中粒度细；
- PD 可更早形成稳定页；
- 抢占释放更细。

代价：

- block table 更长；
- hash/LRU 对象更多；
- kernel page traversal 更多；
- 分配/释放频率更高；
- PD 小传输更多。

### 32.2 大 block

优点：

- metadata 少；
- page 内访问连续；
- kernel/传输启动次数少；
- Prefix hash 数少。

代价：

- 每活跃 Sequence 平均浪费约半页；
- 短请求容量差；
- Prefix 命中必须等更长边界；
- beam 尾页 copy-on-write 更大；
- 部分页传输重发成本更高。

### 32.3 内部碎片近似

若结束位置在页内均匀：

```text
average_wasted_tokens_per_sequence ≈ (B-1)/2
```

字节浪费：

```text
active_sequences * (B-1)/2 * per_token_cache_bytes
```

### 32.4 不能只看吞吐

调 block size 应同时看：

- 最大并发 Sequence；
- KV token capacity；
- Prefix token hit rate；
- P99 TTFT/TPOT；
- block allocation latency；
- paged Attention kernel latency；
- PD/Host transfer；
- Graph bucket 内存。

### 32.5 混合模型约束

线性状态 Prefix Cache 要求 chunk stride 是 block size 的整数倍。

KV Split 可能放大逻辑 block size。

平台 kernel 也可能限制支持的 page size。

先满足正确性约束，再做性能扫描。

---

## 33. 正确性不变量

### 33.1 容量层

```text
n_blocks > 0
padding block id == 0
所有并行 rank 使用兼容容量
cache shape 第一维覆盖物理 block id
```

### 33.2 Sequence 层

```text
kv_cache_tokens_num <= current_max_tokens_capacity
block table 逻辑顺序不随物理 id 排序
每个有效 Block 的 manager/refcount 正确
shared block 不被写时覆盖
```

### 33.3 Batch 层

```text
len(new_cache_slots) == Σ q_seq_len
每个 slot 对应本 Sequence 的新增 token
block table 每行只属于一条 Sequence
last_page_len ∈ [1, block_size]
padding row 指向安全 block 0
```

### 33.4 Attention 层

```text
当前 K/V 先写 cache
Decode/Chunked 使用完整 paged history
kv_len 等于可见逻辑历史
causal mask 与 q/kv 偏移一致
backend 解释的 cache layout 与实际 shape 一致
```

### 33.5 Prefix 层

```text
只发布完整稳定 block
hash 包含完整语义前缀
cache alias 保持物理页引用
驱逐只释放无活跃 owner 的页
```

### 33.6 PD/Host 层

```text
源/目标 block 映射一一对应
copy completion 前不读目标页
确认完成前不释放源页
量化 scale 与 cache 数据一起传
异常路径幂等回收
```

### 33.7 Kimi 混合层

```text
MLA paged token 边界 == KDA state 边界
positional slice 数学不被通用 RoPE 改写
TP head padding 计算后裁回真实 heads
拒绝 speculative token 不提交正式 state
```

---

## 34. 测试应该怎样分层

### 34.1 BlockManager 单测

验证：

- id 0 保留；
- 增量分配；
- free list 守恒；
- copy/move/refcount；
- Prefix 驱逐；
- 并发 wrapper；
- 多 leaf rollback。

### 34.2 SequenceKVState 单测

验证：

- token 到 slot 映射；
- shared block attach；
- exact prompt 退回一页；
- beam copy-on-write；
- multi-block export 顺序；
- transfer cursor 单调。

### 34.3 BatchInputBuilder 单测

验证：

- block table 行；
- slot mapping；
- CSR indptr/indices；
- last page length；
- mixed batch；
- padding；
- grouped tables；
- PD 部分页重传。

### 34.4 Cache write kernel

构造随机物理 block table 与 slot mapping。

把 K/V scatter 到 page。

再用 CPU reference 比较每个有效 slot。

DCU 已有 `reshape_paged_cache_test` 类覆盖。

### 34.5 Attention 算子

把连续 K/V reference 与 paged K/V 使用同一 Q 比较。

覆盖：

- 单页和跨页；
- 最后一页 1/B-1/B；
- 多 Sequence 物理页交错；
- Prefill/Chunked/Decode；
- GQA/MLA；
- BF16/量化；
- 非整齐 batch。

### 34.6 请求级测试

同一模型固定输入比较：

```text
连续 reference/eager
vs
paged online path
```

至少比较：

- 首 token logits；
- 多步 greedy token；
- 长上下文；
- Prefix hit/miss；
- 抢占后重算；
- PD transfer；
- Graph replay。

### 34.7 压力与故障

随机创建、增长、fork、取消、释放 thousands of sequences。

每步检查：

```text
free + physically referenced == total usable blocks
没有重复 active ownership
没有 block id 越界
```

模拟传输超时和请求取消，确认页最终回收。

---

## 35. 本仓库 Kimi K3 当前结论

### 35.1 已确认的通用链路

- Engine 能估算 MLA 与 linear-state 混合容量；
- Kimi 被识别为 MLA 模型；
- KVCacheShape 能生成 latent/position paged cache；
- TP4 服务已完成 cache 分配；
- BlockManager、BatchInputBuilder 可生成标准 page metadata；
- Python NPU MLA 有 paged chunked/decode 分支；
- KDA 有独立 LINEAR state slots。

### 35.2 在线 smoke 已证明什么

五层 TP4 服务完成：

```text
短 Prefill
至少一次 Decode
HTTP/流式返回
请求后服务健康
```

它证明短链路可运行。

### 35.3 在线 smoke 没证明什么

它没有证明：

- 多页长 Prompt；
- 多 chunk Paged MLA 等价；
- 长时间连续 Decode 跨多个 block；
- Prefix Cache 命中；
- 抢占与重算；
- PD/Host Offload；
- ACL Graph 下 paged metadata 更新；
- 93 层全模型容量与稳定性。

### 35.4 最新离线证据

离线逐层 dump 在 Full MLA 的 `reshape_paged_cache` 报 `Invalid argument`。

这是当前最直接的 Paged MLA 待修复点。

正式精度对齐前应优先检查：

```text
slot_mapping dtype/shape/device
nope/rope cache physical layout
block_size
cache tensor stride/contiguous
TP rank local head/latent shape
offline Batch metadata 与在线路径差异
custom op 注册版本
```

### 35.5 不能作出的结论

不能因为 cache tensor 成功分配，就说 Paged Attention 已正确。

不能因为 HTTP 200，就说长上下文分页正确。

不能因为 KDA 层通过，就说 MLA 页写入正确。

不能把五层无语义输出用于精度结论。

---

## 36. 常见误解与排查方向

### 36.1 “Paged KV 会减少模型理论 KV 字节”

不会。

它主要减少预留、搬迁和外部碎片。

每 token 本体由 MHA/GQA/MLA/dtype 决定。

### 36.2 “block id 就是显存地址”

不是。

普通 tensor 中它是第一维索引。

XTensor 下还要经过物理页映射。

### 36.3 “分配了 block 就说明 token 已计算”

不是。

allocated capacity 与 `kv_cache_tokens_num` 分离。

### 36.4 “block table 按物理 id 排序”

绝对不能。

其顺序是逻辑 token 顺序。

物理 id 可任意。

### 36.5 “最后一页长度整除时写 0”

错误。

源码写 block_size，表示满页。

### 36.6 “Prefix Cache 命中整条 Prompt 最好”

模型仍需要至少一个输入位置产生 logits。

xLLM 会退回最后一个完整 block。

### 36.7 “block 0 可以给真实请求”

错误。

它是 padding 保留页。

### 36.8 “所有模型只有一张 KV block table”

错误。

DeepSeek V4 使用 SWA/C4/C128 多表。

Kimi 还同时有 LINEAR state slot。

### 36.9 “Python NPU 类名有 Paged 就一定读历史”

错误。

必须看具体 MHA/MLA 分支最终传入 dense K/V 还是 cache。

### 36.10 “Graph 捕获后页表不再变化”

错误。

Graph 地址稳定，metadata 内容仍要每轮刷新。

### 36.11 “KV INT8 只改 tensor dtype”

错误。

还需要 scale shape、写入量化、读取反量化、swap/transfer 同步 scale。

### 36.12 “PD 可以直接复用 P 的 block id”

错误。

P/D 有独立物理池，必须映射和传输。

### 36.13 “Host hit 后可直接 Attention”

错误。

必须恢复到设备页并等待 copy 完成。

### 36.14 “block 越小一定越好”

错误。

元数据、hash、kernel traversal 和传输启动会增加。

### 36.15 “显存还有很多，block 就不会不足”

错误。

权重、Graph、workspace 和 allocator reserve 已占走显存。

KV pool 创建后可用页数是固定会计边界。

---

## 37. 推荐的源码阅读顺序

第一轮先建立数据结构：

1. [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:29)
2. [kv_cache_capacity.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_capacity.h:24)
3. [block.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.h:35)
4. [sequence_kv_state.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.h:31)

第二轮看启动与容量：

5. [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:39)
6. [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:400)
7. [kv_cache_shape.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:62)
8. `worker_impl.cpp::allocate_kv_cache`

第三轮看 block 生命周期：

9. [block.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.cpp:27)
10. [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:50)
11. [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:116)
12. [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:173)

第四轮看请求到 metadata：

13. `scheduler_policy.cpp`
14. [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:671)
15. [setup_kv_cache_info()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:961)
16. [attention_metadata_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/common/attention_metadata_builder.cpp:46)

第五轮看设备读写：

17. [NPU C++ Attention](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/attention.cpp:39)
18. [CUDA FlashInfer](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/flashinfer_attention.cpp:112)
19. [Python NPU backend](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:222)
20. [Python CUDA backend](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/flashinfer.py:235)

最后再读组合特性：

21. Prefix Cache
22. Chunked Prefill
23. Zero-Eviction
24. PD transfer
25. Hierarchy KV
26. Graph
27. Quantized KV
28. Kimi KDA/MLA

---

## 38. 读完本文应该能回答的问题

1. Paged KV 为什么减少外部碎片，却不减少模型理论 KV 字节？
2. 逻辑 block、物理 block、物理 page 和 kernel tile 有何区别？
3. `kv_cache_tokens_num` 为什么不能用 block 数替代？
4. block id 如何转换成 token slot？
5. 为什么 block 0 永久保留？
6. Block copy 为什么不复制 K/V？
7. Prefix Cache 如何通过 refcount 共享物理页？
8. 新请求分配为什么先 Prefix attach 再增长？
9. Beam Search 为什么需要最后一页 copy-on-write？
10. KV capacity 如何考虑 MLA、Indexer、量化 scale 和 linear state？
11. 为什么并行组使用最小 worker 容量？
12. NPU、CUDA、MLU 的 cache shape 有何不同？
13. Batch 如何生成 `new_cache_slots`？
14. `block_tables` 与 FlashInfer CSR 表示如何对应？
15. `last_page_len` 为什么整除时等于 block size？
16. Prefill、Chunked Prefill、Decode 哪些阶段读取 paged history？
17. 为什么类名包含 Paged 仍不能证明后续 chunk 正确？
18. Kimi MLA 的两份 paged cache 各保存什么？
19. KDA state 为什么不能塞进普通 token pages？
20. DeepSeek V4 为什么需要多 block tables？
21. PD 中 P/D block ids 为什么必须映射？
22. Host hit 为什么仍需 H2D？
23. Graph replay 前哪些 page metadata 必须更新？
24. block size 如何影响碎片、Prefix、kernel 和传输？
25. 当前 Kimi 离线 `reshape_paged_cache` 失败说明了什么？

如果这些问题仍无法从源码定位，建议按第 37 节顺序再走一遍。

---

## 39. 最终总结

Paged KV 的核心不是“把 tensor 切成块”。

它建立了一套跨 Scheduler、内存管理和 Attention 的地址翻译协议。

在这套协议里：

```text
Sequence 只维护逻辑历史
BlockManager 决定物理页所有权
BatchInputBuilder 生成本轮写地址与完整读页表
KV kernel 按 slot 写入
Attention kernel 按 block table 读取
RAII/refcount/Prefix/PD/Host 管理页的后续生命周期
```

判断实现是否正确，不能只看 cache 是否分配成功。

必须同时验证：

```text
容量公式正确
物理 shape 正确
逻辑 block 顺序正确
slot mapping 正确
当前 K/V 写入正确
完整历史读取正确
释放与共享正确
跨设备传输正确
模型特有 state 与 token 边界正确
```

对普通 MHA，这是 PagedAttention 的基础正确性。

对 Kimi K3，它还要同时满足 MLA latent pages、positional cache、KDA recurrent state、TP head padding 和项目定义的 Gated-MLA 数学。

当前通用框架链路已经较完整。

但最新离线五层验证在 Full MLA 的 `reshape_paged_cache` 仍有真实 NPU 参数错误。

因此最严谨的当前结论是：

```text
Paged KV 的容量、管理和 metadata 主链已经存在；
Kimi K3 短在线链路曾经运行；
Kimi Full MLA 的离线 paged write 仍需修复并完成逐层精度对齐；
在此之前不能宣称 Kimi K3 的所有 Paged Attention 场景已正式验收。
```

这也是阅读推理框架时最重要的习惯：

区分“抽象存在”“路径可达”“请求成功”和“数学正确且完成压力验收”四个不同层级。

