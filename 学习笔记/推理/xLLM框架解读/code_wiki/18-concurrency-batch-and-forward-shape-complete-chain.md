# xLLM 并发、Batch 与 Forward Shape 全解：从在线请求到模型真实输入

> 本文源码基线：`xllm` 仓库提交 `8891280e35e9ca39762d5fdba9c60aea6850a700`，并以当前工作区中的 Python Attention、ModelExecutor 与 ACL Graph 实现为准。
>
> 本文只讨论自回归 LLM/VLM 在线推理主链。DiT 的 `sp_size`、`cfg_size` 与 `micro_batch_num` 会说明边界，但不会套用 LLM 的 Sequence 语义。

这篇文章要彻底回答一个看似简单、实际贯穿整个推理框架的问题：

> xLLM 一次 `forward` 的 batch size 到底是多少？

答案不是一个数字。

同一个调度 step 中，至少同时存在：服务并发请求数、等待请求数、被调度请求数、Sequence 数、本步 query token 数、每个 DP rank 的本地 token 数、Graph bucket 容量、MoE 路由后的 token-expert 对数。

如果不先给“batch size”标单位，那么下面这些说法可以同时成立：

- 服务中有 100 个未完成请求；
- 当前 step 只选择了 32 条 Sequence；
- 其中 1 条做 256-token Chunked Prefill，31 条各做 1-token Decode；
- 模型输入第一维是 `T = 256 + 31 = 287`；
- `DP=2` 后两个 rank 分别实际计算 160 和 127 个 token；
- Graph 为了复用静态图，把其中一个 Decode batch 从 13 padding 到 16；
- 但真实请求数、真实 Sequence 数和真实输出数都没有因此变成 16。

本文会沿着如下完整链路解释这些数字：

```text
HTTP 请求
  -> 服务并发限流
  -> Request / SequencesGroup / Sequence
  -> Scheduler 双预算选择
  -> Continuous Batching 动态组批
  -> 按 DP rank 拆成 Batch
  -> BatchInputBuilder 压平本步 token
  -> ForwardInput + Attention metadata
  -> TP / CP / EP 并行模型 forward
  -> Graph bucket 静态容量与 padding
  -> logits 选择、采样、Sequence 回填
  -> 流式 HTTP 响应
```

相关专题：

- 在线服务完整链路见 [01-online-service-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/01-online-service-complete-chain.md:1)；
- Scheduler 的队列、抢占与策略见 [02-scheduler-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/02-scheduler-complete-chain.md:1)；
- Graph 静态地址与 replay 见 [03-graph-execution-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/03-graph-execution-complete-chain.md:1)；
- Chunked Prefill 续算细节见 [04-chunked-prefill-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/04-chunked-prefill-complete-chain.md:1)；
- 显存组成与申请时序见 [17-memory-and-vram-management-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/17-memory-and-vram-management-complete-chain.md:1)。

---

## 1. 先建立最终认知

先记住六句话。

第一，`max_concurrent_requests` 限制的是服务入口中“尚未结束的请求”，不是一次模型 forward 的 batch。

第二，Scheduler 同时用 `max_seqs_per_batch` 和 `max_tokens_per_batch` 限制一个 step。前者数 Sequence，后者数“本步新计算的 query token”。

第三，xLLM 的 LLM 输入通常不是 `[B, S]` 的 padding 矩阵，而是把所有 Sequence 本步要算的 token 压成一维：

```text
token_ids.shape = [T]
hidden_states.shape = [T, H]
T = sum(q_seq_lens[i])
```

第四，Decode 的“1”是每条 Sequence 本步的 `q_len = 1`，不是整个 Decode batch 只有一条 Sequence。若本步有 87 条普通 Decode Sequence，则 `T = 87`。

第五，Continuous Batching 每个 step 都会重新选择活跃 Sequence；请求可以进入、完成、抢占和恢复，所以真实 batch 会不断变化。Graph 需要静态 shape，才准备了 `1, 2, 4, 8, 16, 32, 48...` 等不同 bucket。

第六，模型的 Dense/MLP 层只看到压平后的 token 行；Attention 依靠 `q_seq_lens`、累计长度、KV 长度与 `block_tables` 恢复 Sequence 边界；采样阶段再依靠 `selected_token_idxes` 和 `output_targets_` 把输出还给原 Sequence。

源码总入口分别是：

- 服务并发限制：[rate_limiter.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/rate_limiter.cpp:26)；
- 单步预算初始化：[continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:216)；
- DP batch 构造：[batch_factory.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_factory.cpp:32)；
- token 压平：[batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:734)；
- 模型输入生成：[batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:1189)；
- Kimi K3 forward：[kimi_k3_text.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/models/kimi_k3_text.py:1537)。

---

## 2. 为什么“batch size”在推理服务里特别容易说错

训练通常把 batch 写成一个相对稳定的张量：

```text
[batch_size, sequence_length, hidden_size]
```

在线自回归推理不同。

请求到达时间不同，prompt 长度不同，已生成长度不同，停止时间不同，还会发生 Prefix Cache 命中、Chunked Prefill、抢占、投机验证、beam 扩展和 DP 分流。

所以在线服务中的 batch 是一个“每步重新装箱”的集合，而不是从请求开始到结束都固定不变的训练 batch。

更准确的说法应带上单位：

| 名称 | 符号 | 统计对象 | 典型位置 |
|---|---:|---|---|
| 服务在途请求数 | `C_req` | 尚未结束的 Request | RateLimiter / metrics |
| 单步请求数 | `B_req` | 本 step 至少有一条 Sequence 入选的 Request | Scheduler |
| 单步逻辑序列数 | `B_seq` | 本 step 入选的 Sequence | `Batch::size()` |
| 本步 query token 数 | `T_q` | 所有 Sequence 本步新算 token 之和 | `token_ids.numel()` |
| DP 本地 token 数 | `T_q[d]` | DP rank `d` 实际拿到的 token | `dp_global_token_nums[d]` |
| Graph bucket | `G` | 静态图容纳的 token/Decode 行数 | Graph runner |
| KV 上下文长度 | `L_kv[i]` | 第 `i` 条 Sequence 可见的完整上下文 | Attention metadata |
| MoE 路由项数 | `T_route` | token 与 top-k expert 的配对数 | MoE dispatcher |

其中最重要的关系是：

```text
B_req <= B_seq
T_q = sum_i q_i
sum_d T_q[d] = T_q                  # 忽略 Graph/通信 padding 时
G >= T_q[d]                         # 纯 Decode Graph 常见关系
T_route ~= T_q[d] * num_experts_per_tok
```

`B_req <= B_seq` 是因为一个 Request 可以拥有多条 Sequence，例如 `best_of` 或 beam search。

`T_q` 与 `B_seq` 没有固定比例，因为 Prefill 的一条 Sequence 可贡献几百甚至几千个 token，而普通 Decode 通常只贡献一个。

---

## 3. Request、SequencesGroup 与 Sequence 到底是什么关系

### 3.1 Request 是用户语义

一个 HTTP completion/chat 请求进入引擎后形成一个 `Request`。

它保存请求 ID、prompt、采样参数、停止条件和一个 `SequencesGroup`。`Request::sequences()` 实际转发到组中的 Sequence 列表，见 [request.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/request.h:49)。

### 3.2 Sequence 是调度与 KV 管理语义

Sequence 是一条具体候选生成轨迹。

它持有：

- `request_id`；
- prompt 与已生成 token；
- 当前 Prefill/Chunked Prefill/Decode stage；
- KV block 状态；
- 所属 `dp_rank`；
- 解码器与采样状态。

请求 ID 的访问器见 [sequence.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.h:246)，DP 归属见 [sequence.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.h:346)。

### 3.3 一个 Request 不一定等于一条 Sequence

`SequencesGroup` 初始只 `add()` 一条 Sequence，见 [sequences_group.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequences_group.cpp:40)。

普通 `best_of > 1` 时，Prefill 完成后才扩展到多条候选，避免每条候选重复做相同 prompt Prefill。扩展条件与循环见 [sequences_group.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequences_group.cpp:68)。

因此：

```text
1 HTTP Request
  -> 1 SequencesGroup
  -> 1..best_of Sequence
```

这也是为什么 `max_seqs_per_batch=200` 不能简单理解为“一次最多 200 个 HTTP 请求”。

例如 40 个 `best_of=4` 的请求完成 Prefill 后，理论上可形成 160 条候选 Sequence；如果另有普通请求，同一个 step 很快会碰到 Sequence 预算。

---

## 4. 服务并发 `max_concurrent_requests` 不是模型 batch

服务配置默认把 `max_concurrent_requests` 设为 200，见 [service_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/service_config.h:69)。

RateLimiter 的判断是：

```text
当前在途请求数 >= max_concurrent_requests
  -> 拒绝或限制新请求
否则
  -> 在途请求计数 +1
```

对应实现见 [rate_limiter.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/rate_limiter.cpp:26)。请求结束后再减少计数，见 [rate_limiter.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/rate_limiter.cpp:49)。

“在途”包含的可能是：

- 正在 Prefill 的请求；
- 正在 Decode 的请求；
- 在 Scheduler 队列里等待的请求；
- 因 KV 不足被抢占、等待恢复的请求；
- 正在流式返回但尚未完成的请求。

它不保证这些请求同一时刻都进入模型。

例如：

```text
max_concurrent_requests = 200
max_seqs_per_batch      = 32
```

服务可接纳 200 个未完成请求，但一个普通单 Sequence step 最多调度 32 条 Sequence；其余请求留在等待或阶段队列中。

反过来，即使 `max_concurrent_requests=8`，若每个请求 `best_of=4`，Decode 阶段仍可能出现多于 8 条 Sequence。

所以这个参数首先是入口背压和服务稳定性参数，不是算子 shape 参数。

---

## 5. Scheduler 真正使用的是“Sequence 预算 + Token 预算”

xLLM Scheduler 默认配置为：

```text
max_tokens_per_batch = 10240
max_seqs_per_batch   = 200
```

定义见 [scheduler_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/scheduler_config.h:62)。

每次 `prepare_batch()` 都重新创建 `ScheduleBudget`：

```cpp
remaining_token_budget = max_tokens_per_batch;
remaining_seq_budget = max(max_seqs_per_batch, 1);
```

源码见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:225)，预算结构见 [scheduler_policy.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.h:71)。

### 5.1 Sequence 预算解决什么问题

即使总 token 很少，Sequence 太多也会带来：

- 更多 block table 行；
- 更多长度 metadata；
- 更多采样参数；
- 更多请求状态维护；
- Decode Attention 更多独立 KV 链；
- Graph batch 与 kernel batch 变大。

因此不能只限制 token。

### 5.2 Token 预算解决什么问题

Prefill 一条 Sequence 就可能贡献成千上万个 query token。

这些 token 会扩大：

- embedding 与 hidden activation；
- Q/K/V 临时张量；
- MLP/MoE 中间张量；
- KV 写入量；
- TP/EP/CP 通信量；
- 单步延迟。

因此不能只限制 Sequence。

### 5.3 哪个预算先耗尽，哪个决定真实 batch

考虑：

```text
max_seqs_per_batch   = 64
max_tokens_per_batch = 1024
```

- 64 条普通 Decode Sequence：消耗 64 Sequence、64 token，Sequence 预算先满；
- 4 条各 256-token Prefill：消耗 4 Sequence、1024 token，token 预算先满；
- 1 条 768-token chunk + 64 条 Decode 不可能全进，因为 Sequence 和 token 都要检查。

真实 batch 是策略、两个预算、KV 可分配块、延迟预算与队列状态共同求出的结果。

---

## 6. Continuous Batching 到底“连续”在哪里

Continuous Batching 不是把一批请求锁定到全部完成。

它的核心是每个生成 step 都重新组批：

```text
step 0: A(prefill), B(prefill)
step 1: A(decode), B(decode), C(prefill)
step 2: B(decode), C(chunked prefill), D(prefill)  # A 已结束
step 3: B(decode), C(decode), D(chunked prefill), E(decode)
...
```

`ContinuousScheduler::prepare_batch()` 每步都会：

1. 从请求队列吸收新请求；
2. 收集已完成请求；
3. 初始化新预算；
4. 让策略从 Prefill/Chunk/Decode 队列选择 Sequence；
5. 创建新的 DP batches；
6. 交给 Engine 执行。

完整骨架见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:216)。

所以“continuous”指：

- batch 边界不再等到整批请求全部完成；
- 有空位就可以插入新请求或恢复的请求；
- 已完成 Sequence 下一步立即退出；
- batch composition 随 step 连续变化。

它不表示一个 kernel 在时间上永不结束，也不表示一个请求可以跨 DP rank 任意漂移。

---

## 7. Prefill 的 batch size 应该怎样理解

Prefill 处理尚未进入 KV cache 的 prompt token。

对第 `i` 条 Sequence：

```text
q_i = min(
  当前 token 总数 - 已有 KV token 数,
  Scheduler 给这条 Sequence 的 token budget
)
```

这就是 `BatchInputBuilder::process_single_sequence()` 的真实计算，见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:671)。

于是纯 Prefill batch 的模型 token 数为：

```text
T_prefill = q_0 + q_1 + ... + q_(B_seq-1)
```

例子：

| Sequence | prompt 长度 | 已缓存 KV | 本步 budget | `q_i` |
|---|---:|---:|---:|---:|
| A | 800 | 0 | 800 | 800 |
| B | 300 | 0 | 300 | 300 |
| C | 1000 | 768 | 128 | 128 |

那么：

```text
B_seq = 3
T_q   = 800 + 300 + 128 = 1228
q_seq_lens = [800, 300, 128]
```

所以“Prefill batch 很大”通常说的是 `T_q` 很大，不一定是 `B_seq` 很大。

一条超长 prompt 也可能形成 `B_seq=1, T_q=4096` 的大 Prefill forward。

---

## 8. Decode 的 batch size 为什么绝不是固定 1

普通自回归 Decode 每条活跃 Sequence 每一步只需要处理一个新 token：

```text
q_i = 1
```

但一个 step 可以同时处理很多条 Sequence：

```text
T_decode = sum_i 1 = B_seq
```

如果有 96 条活跃 Decode Sequence：

```text
B_seq = 96
q_seq_lens = [1, 1, ..., 1]
T_q = 96
input_ids.shape = [96]
hidden_states.shape = [96, H]
```

“Decode batch size 是 1”混淆了两个维度：

- 每条 Sequence 的 query length 是 1；
- 整个 batch 的 Sequence 数可以是几十、几百。

Scheduler 的普通 Decode 路径会遍历 Request 中未完成的 Sequence，每条增加：

```text
allocated_seqs += 1
allocated_tokens += min_speculative_tokens_required + 1
```

实现见 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:583)。没有投机解码时，`min_speculative_tokens_required=0`，所以每条 Sequence 消耗一个 token 预算。

Decode 的难点不是本步 query 长，而是每条 query 都要读取自己可能很长的历史 KV；所以它通常是访存与调度延迟敏感，而不是大矩阵吞吐型。

---

## 9. Chunked Prefill 怎样改变真实 batch

Chunked Prefill 把一条长 prompt 分多个 step 处理。

它改变的是这条 Sequence 当前 step 的 `q_i`，不是把它伪装成多条 Sequence。

例如 prompt 长 4096，chunk 上限 512：

```text
step 0: B_seq=1, q=[512], cached_kv=0
step 1: B_seq=1, q=[512], cached_kv=512
...
step 7: B_seq=1, q=[512], cached_kv=3584
```

如果允许 Mix Batch，一个 step 还可以是：

```text
Sequence A: chunked prefill q=256
Sequence B..Q: 16 条 decode，各 q=1

B_seq = 17
T_q   = 256 + 16 = 272
```

`max_tokens_per_chunk_for_prefill` 是单条 Prefill Sequence 的 chunk ceiling；`max_tokens_per_batch` 是整个 step 所有 Sequence 的 query-token ceiling。二者不能互换。

配置定义见 [scheduler_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/scheduler_config.h:70)，完整续算链路见 [04-chunked-prefill-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/04-chunked-prefill-complete-chain.md:331)。

---

## 10. Prefill、Decode 与 Mixed 是“整个本地 Batch”的类型

`BatchForwardType` 有：

```text
PREFILL
CHUNKED_PREFILL
DECODE
MIXED
EMPTY
```

定义见 [batch_forward_type.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_forward_type.h:21)。

`Batch::add()` 每加入一条 Sequence 都会更新整个 batch 类型：

- Prefill 中加入 Decode -> Mixed；
- Chunked Prefill 中加入 Decode -> Mixed；
- Decode 中加入非 Decode -> Mixed；
- 只有 Decode Sequence -> Decode。

逻辑见 [batch.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.cpp:75)。

注意：`MIXED` 不是一种新的 Sequence stage。

它表示同一个本地 batch 里同时存在不同 stage 的 Sequence。每条 Sequence 的真实 `q_i` 仍由自己的 token/KV 状态决定。

---

## 11. 是否允许 Mixed Batch 由哪些配置决定

`resolve_batch_mode()` 会联合处理：

- `enable_mix_batch`；
- `enable_chunked_prefill`；
- Scheduler priority strategy；
- `cp_size`；
- 投机 token 数；
- PD 实例角色。

实现见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:63)。

当前明确规则包括：

- 未开启 Chunked Prefill，Prefill 独占 batch，Mix 被关闭；
- `cp_size > 1` 时，Prefill 与 Decode 不混；
- 投机 token 数大于 0 时，Mix 被关闭；
- PD 的 PREFILL 实例使用独占 Prefill batch；
- `multi_slo_and_prio` 强制启用 Chunked Prefill。

这说明配置中的 `enable_mix_batch=true` 只是用户意图，不一定是最终有效模式。

必须看 `resolve_batch_mode()` 后的约束。

策略选择也会变化：

- 不 Mix：`PrefillFirstPolicy`；
- Mix + 常规优先级：`DecodeFirstPolicy`；
- Mix + multi-SLO：`UnifiedPolicy`。

策略枚举解释见 [scheduler_policy.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.h:85)。

---

## 12. 从调度结果到 Batch：Sequence 如何保持身份

Scheduler 最终形成三个并行数组：

```text
running_requests
running_sequences
running_sequences_budgets
```

第 `i` 条 `running_sequences[i]` 对应自己的 `running_sequences_budgets[i]`。

`BatchFactory::create_batches()` 遍历这些 Sequence，并按 `sequence->dp_rank()` 放入对应 `Batch`，见 [batch_factory.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_factory.cpp:32)。

`Batch` 内部保存的是：

```text
sequences_[i]
allowed_max_tokens_[i]
```

`Batch::size()` 返回 `sequences_.size()`，即逻辑 Sequence 数，不是 token 数，见 [batch.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.h:79)。

Continuous Batching 并没有丢失“哪个 Sequence 属于哪个请求”。

在压平前，身份由 `Sequence*` 保持；压平后，Attention 使用长度边界，Embedding/模型扩展功能可使用 `request_ids`；输出时 `Batch::output_targets_` 仍直接保存 `Sequence*`。

---

## 13. BatchInputBuilder 如何把多个 Sequence 压成一个 token 维

对每条 Sequence，Builder 先算：

```text
n_tokens          = Sequence 当前全部 token 数
n_kv_cache_tokens = 已写入 KV cache 的 token 数
q_seq_len         = min(n_tokens - n_kv_cache_tokens, allowed_max_tokens)
seq_len           = n_kv_cache_tokens + q_seq_len
```

然后只抽取索引区间：

```text
[n_kv_cache_tokens, seq_len)
```

并顺序追加到 `flatten_tokens_vec`。

真实循环见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:734)。

假设：

```text
A 本步 token: [a0, a1, a2]
B 本步 token: [b7]
C 本步 token: [c4, c5]
```

压平后：

```text
token_ids = [a0, a1, a2, b7, c4, c5]
positions = [ 0,  1,  2,  7,  4,  5]
q_seq_lens = [3, 1, 2]
```

于是：

```text
B_seq = 3
T_q   = 6
```

这里没有 `[3, max_seq_len]` 的二维 padding 矩阵，也没有让 B 的一个 token占三行。

这就是 packed/flattened batch。

---

## 14. 压平后 Attention 怎样知道 Sequence 边界

只给 `[T]` 的 token_ids 当然不够。

Builder 同时构建：

- `q_seq_lens`：每条 Sequence 本步 query 长度；
- `q_cu_seq_lens`：query 累计边界；
- `kv_seq_lens`：每条 Sequence 当前完整 KV 长度；
- `kv_cache_tokens_nums`：forward 前已缓存 token 数；
- `block_tables`：每条 Sequence 的逻辑 KV block 到物理 block 映射；
- `new_cache_slots`：新 token 应写入的物理 slot；
- Paged KV 的 `indptr / indices / last_page_len`。

它们在 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:1189) 中被写入 `ModelInputParams`。

以上例：

```text
q_seq_lens    = [3, 1, 2]
q_cu_seq_lens = [0, 3, 4, 6]
```

因此 Attention 知道：

```text
flattened rows [0,3) -> A
flattened rows [3,4) -> B
flattened rows [4,6) -> C
```

同时，A/B/C 各自用自己的 block table 访问历史 KV，不会跨请求互相 attention。

这组 metadata 才是 Continuous Batching 保持隔离性的核心，而不是在 token tensor 里保存 HTTP request ID。

---

## 15. Dense/MLP 层为什么不需要知道请求边界

Embedding、RMSNorm、Linear、激活函数和普通 MLP 都是逐 token 行或对最后一维运算。

输入：

```text
hidden_states: [T, H]
```

它们只需要对 T 行做相同变换：

```text
[T, H] @ [H, D] -> [T, D]
```

不需要知道第 17 行属于请求 A 还是请求 B。

Kimi K3 model forward 先 embedding，再按层传递 `hidden_states`；其 residual buffer 第一维直接取 `hidden_states.shape[0]`，见 [kimi_k3_text.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/models/kimi_k3_text.py:1537)。

真正需要 Sequence 边界的是：

- Attention 的因果可见范围；
- KV cache 的读写位置；
- 采样输出选择；
- 某些请求级 embedding/adapter/状态功能。

所以 packed batch 能减少 padding 浪费，同时保持语义正确。

---

## 16. Attention 模型侧怎样消费这个 batch

Python Attention backend 的公开契约要求 metadata 提供 slot mapping、Paged KV 索引、累计长度、KV 长度、Prefill 类型等字段，见 [backend.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/backend.py:30)。

以 NPU Paged Attention 为例：

1. `num_tokens = q.shape[0]`；
2. 把 K/V reshape 成 `[T, kv_heads, head_dim]`；
3. 按 `slot_mapping` 写入 Paged KV；
4. 根据 metadata 选择 Prefill 或 Decode kernel；
5. kernel 使用长度和 block table 保持每条 Sequence 的上下文边界。

入口见 [npu_paged_attention.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:231)。

MLA 也遵循相同的 batch 原则：输入第一维仍是 T，区别是 KV 表示和 Attention 算法变为 latent cache、absorbed/unabsorbed path 或 sparse top-k。分支见 [npu_paged_attention.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:258)。

所以模型类型可以改变每 token 需要保存的 KV、通信和算子，但不改变“多个 Sequence 本步 token 被压平”的基础语义。

---

## 17. logits 为什么通常不对所有 Prefill token 都计算

模型 backbone 输出：

```text
hidden_states.shape = [T, H]
```

但并非每一行都需要进 lm_head。

普通生成通常只需要每条已完成 Prefill 或 Decode Sequence 的最后一个有效 hidden row。中间 Prefill chunk 若还没到采样点，可能根本不产出 next token。

Builder 在需要采样的位置记录：

```text
selected_token_idxes.push_back(flatten_tokens_vec.size() - 1)
```

实现见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:909)。

因此存在第三组容易混淆的数字：

```text
T_model    = backbone 处理的 token 行数
T_selected = 进入 logits 的 hidden 行数
T_sample   = 实际采样输出数
```

通常：

```text
T_sample <= T_selected <= T_model
```

例如一条 512-token 中间 chunk 加 20 条 Decode：

```text
T_model = 532
T_sample 可能只有 20
```

这能显著减少大词表 lm_head 的计算。

---

## 18. 输出怎样回到原来的 Request / Sequence

压平并不意味着输出按猜测还原。

`Batch::refresh_output_targets()` 在 forward 前为需要输出的采样点保存：

```text
{ Sequence*, sample_id, from_sample_slot }
```

源码见 [batch.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.cpp:375)。

采样完成后，`Batch::process_sample_output()` 按 `output_targets_` 直接取回目标 `Sequence*`，追加 token、更新状态并处理 beam，见 [batch.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.cpp:609)。

完整身份链是：

```text
Sequence*
  -> 本步 packed row 区间
  -> selected_token_idx
  -> logits/sample output index
  -> output_targets_[index].sequence
  -> Sequence append token
  -> Request 聚合输出
```

因此请求之间不会因为 Continuous Batching 的进入/退出而串 token。

正确性依赖的不是 batch 顺序永远不变，而是每个 step 都重建一致的 metadata 与输出映射。

---

## 19. 一组仓库测试给出的真实 Mixed Batch

`BatchTest` 构造了四条 Sequence：

- Seq1：9-token Prefill；
- Seq2：1-token Decode；
- Seq3：1-token Decode；
- Seq4：只允许取 4-token chunk。

期望压平输入是：

```text
q_seq_lens = [9, 1, 1, 4]
B_seq      = 4
T_q        = 15
q_max      = 9
type       = MIXED
```

测试源码与期望值见 [batch_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/batch/batch_test.cpp:620)。

它还验证：

```text
selected_token_idxes = [8, 9, 10]
```

即 Seq1/2/3 的采样位置分别落在 packed rows 8、9、10；Seq4 是未完成的中间 chunk，不采样。

这个测试几乎完整回答了三个问题：

1. Prefill 与 Decode 可以在一个 forward 中共存；
2. 模型 batch 第一维是 15，不是 Sequence 数 4；
3. 只有 3 个输出，不是所有 15 个 token 都做采样。

---

## 20. DP 怎样把“全局 batch”拆成本地 batch

DP 的核心是多份模型副本处理不同 Sequence。

Sequence 第一次分配 KV 时，`BlockManagerPool` 选择空闲 block 最多的 DP rank，并把结果写入 `sequence->dp_rank()`；后续直接复用，见 [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:116)。

为什么必须固定？

因为这条 Sequence 的 KV cache、线性状态和相关资源已经属于该 DP rank。每 step 漂移会要求搬迁全部状态，代价大且容易破坏一致性。

`BatchFactory` 据此产生 `dp_size` 个本地 `Batch`：

```text
global selected sequences
  -> dp_rank 0 Batch
  -> dp_rank 1 Batch
  -> ...
```

实现与“生命周期内保持同 rank”的注释见 [batch_factory.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_factory.cpp:56)。

假设 Scheduler 全局选择：

```text
q = [128, 1, 1, 64, 1, 1]
dp_rank = [0, 1, 0, 1, 0, 1]
```

则：

```text
DP0: q=[128,1,1], B_seq=3, T_q[0]=130
DP1: q=[1,64,1],  B_seq=3, T_q[1]=66
global T_q=196
```

所以在 DP 场景下，必须继续追问：说的是 Scheduler 全局 batch，还是某个模型副本的本地 batch？

---

## 21. `dp_global_token_nums` 为什么要广播给每个本地 forward

Engine 为每个 DP batch 调用 `prepare_forward_input()`，再统计：

```text
dp_global_token_nums[d] = local token_ids.numel()
```

源码见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1208)。

之后它把完整向量写进每个 DP rank 的 `ParallelInput`，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1277)。

这是因为 DP/EP 混合通信或 Graph replay 可能需要所有参与 rank 对全局 shape 达成一致：

- 各 rank 实际 token 数不同；
- AllToAll buffer 必须知道每个来源/目标的长度；
- Graph 静态地址需要按统一容量准备；
- 空 DP rank 也可能必须参与 collective。

因此：

```text
本 rank 真正模型输入 = local token_ids
全局 shape 协调信息 = dp_global_token_nums
```

不能看到 `dp_global_token_nums` 就认为每个 DP rank 都重复计算全部 token。

---

## 22. 空 DP rank、fake input 与逻辑 batch 必须分开

某个 step 可能所有活跃 Sequence 都固定在 DP0，DP1 没有真实 Sequence。

但当 EP、mapping 或 Graph collective 要求所有 rank 同步参与时，空 rank 完全跳过可能造成 collective hang 或 shape 不一致。

Worker 因此有条件地给非 VLM 空 shard 注入一个 fake token，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:802)。

必须区分：

```text
logical B_seq = 0
logical T_q   = 0
physical tensor rows for collective/graph = 1 fake row
```

fake row 不是一个新用户请求，也不能产生用户输出。

VLM 路径特意跳过这种注入，因为只有 token 没有合法 Sequence/cache metadata 对 KDA/Paged Attention 是非法输入；源码中的保护说明同样位于 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:807)。

这也是阅读 profiler 时必须同时观察 logical 与 padded/physical 数字的原因。

---

## 23. TP 会不会把 batch size 再除一次

不会。

TP rank 处理同一个 DP-local packed token batch，只是切分权重、Attention heads 或 hidden projection。

例如 DP0 本地：

```text
token_ids.shape = [130]
hidden_states.shape = [130, H]
TP = 4
```

四个 TP rank 都对应这 130 个逻辑 token，不是每个 rank 只拿 32 或 33 个请求。

切分发生在模型维度：

- Column Parallel Linear 切输出维，必要时 AllGather；
- Row Parallel Linear 切输入维，局部 matmul 后 AllReduce。

Python 实现与通信语义见 [linear.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/layers/linear.py:25) 和 [linear.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/layers/linear.py:72)。

对 batch 的正确公式是：

```text
DP: 分 Sequence，通常改变每个模型副本的 T
TP: 分模型维，不改变该 DP 副本的逻辑 T
```

xLLM Worker 中 LLM 的有效 TP 大小由：

```text
tp_size = world_size / (dp_size * cp_size)
```

推导，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:286)。这点还意味着命令行的 `tp_size` 字段不能脱离具体后端启动拓扑孤立理解。

---

## 24. CP 改变的是上下文布局，不是 Scheduler 的 Sequence 数

CP（Context Parallel）让同一 DP-local Sequence 的 token/context 工作在多个 CP rank 间协作。

概念上：

```text
DP 先决定这条 Sequence 属于哪个模型副本
CP 再在该副本内部切分/协同上下文计算
TP 再切模型维
```

因此 Scheduler 选中的 `B_seq` 不会因为 `cp_size=2` 自动翻倍，也不会除以 2。

变化的是：

- 每个 CP rank 处理/持有的 query 或 KV 布局；
- CP plan、reorder 与 load-balance metadata；
- Prefix KV 是否需要 AllGather；
- Attention 内的通信与局部 shape。

配置注释把 CP 描述为 DSA Attention 的 Context Parallel，见 [parallel_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.cpp:27)。`kv_split_size` 还能与 `cp_size` 分离，控制 KV split 宽度，见 [parallel_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.h:63)。

当前 Scheduler 在 `cp_size > 1` 时关闭 Prefill/Decode Mix，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:75)。这是当前实现约束，不是 CP 理论上永远不能 Mixed Batch。

更完整的 CP/PD/并行交互见 [10-distributed-parallelism-and-communication.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/10-distributed-parallelism-and-communication.md:1)。

---

## 25. SP 在 xLLM 中有两种容易混淆的含义

第一种是配置项 `sp_size`。

当前 `parallel_config.cpp` 明确标注它“only used for DiT model”，见 [parallel_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.cpp:38)。

因此讨论 LLM 在线请求 batch 时，不能写：

```text
LLM local batch = global batch / sp_size
```

当前主链没有这样的通用规则。

第二种是某些模型/算子内部所说的 sequence parallel tensor layout。

它可能沿 token 维切激活并与 TP 通信配合，但这是模型算子并行技术；它和框架中的 `Sequence` 对象不是一个概念。

需要始终区分：

```text
Sequence object: 一条请求候选轨迹，有 KV 与生成状态
sequence-parallel tensor shard: 激活张量在设备间的布局
```

同名不代表同一抽象层。

---

## 26. EP 与 MoE 会产生另一种“batch 膨胀”

MoE 层对每个 token 选择 top-k experts。

若本地 packed token 数为 `T`，每 token 选 `K` 个 expert，则路由项数量近似：

```text
T_route = T * K
```

这不是新的 Sequence，也不是新的请求，而是 token-expert pair。

EP AllToAll 前后还可能有：

- 按 expert 的 token 重排；
- capacity/alignment padding；
- DP/EP mapping 对齐；
- 空 rank 参与；
- Graph 静态 buffer 容量。

Worker 会根据 `dp_global_token_nums`、`num_experts_per_tok` 与 mapping 构造 `DpEpPadding`，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:839)。

所以分析 MoE profile 时还要区分：

```text
Scheduler token T
模型 packed token T
路由 token-expert 数 T*K
通信 padding 后 capacity
```

它们可能分别出现在不同日志或算子 shape 中。

---

## 27. PD 分离怎样改变 Prefill 与 Decode batch 分布

PD 分离把 Prefill 和 Decode 放到不同服务实例或资源池。

它不会改变基础定义：

```text
T_q = sum(q_seq_lens)
```

但会改变每类实例看到的 workload：

- Prefill 实例主要看到少量 Sequence、较大的 `q_i`；
- Decode 实例主要看到大量 Sequence、普通情况下 `q_i=1`；
- KV/状态需要从 P 侧传给 D 侧；
- D 侧 Continuous Batching 继续把不同请求的 Decode token 合并。

当前 `resolve_batch_mode()` 对 PD PREFILL 实例关闭 Mix，并使用更清晰的 PrefillFirst 行为，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:85)。

因此 PD 后再谈“服务 batch size”，必须说明是 P 实例还是 D 实例。一个统一实例的 Mixed Batch 分布已经不存在于单个节点，但全链路吞吐仍由两侧能力和 KV 传输共同决定。

---

## 28. 为什么 Graph replay 会出现许多不同 batch size

Graph capture/replay 要求关键 tensor shape 和地址稳定。

Continuous Batching 的真实 Decode Sequence 数却每步变化：

```text
step 100: 13 条活跃 Sequence
step 101: 17 条活跃 Sequence
step 102: 16 条活跃 Sequence
step 103: 31 条活跃 Sequence
```

如果只捕获一个最大图，例如 200，每个 13-token Decode step 都 padding 到 200，会浪费大量计算和显存带宽。

如果为每个整数都捕获一个图，图数量、capture 时间和常驻显存又过高。

xLLM 因而使用 bucket 折中。

Python Decode Graph 的 bucket 规则是：

```text
1 -> 1
2 -> 2
3..4 -> 4
5..8 -> 8
9..16 -> 16
17..32 -> 32
33..48 -> 48
...
```

实现见 [decode_cuda_graph.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/model_executor/runners/decode_cuda_graph.py:33)。

所以日志中出现很多 `batch_size`，恰恰说明系统在用多个静态 shape 覆盖动态 Continuous Batch。

---

## 29. Graph 的 actual batch、padded batch 与输出 batch

以真实 Decode batch 13 为例：

```text
actual logical sequences = 13
actual input tokens      = 13
selected graph bucket    = 16
static graph tensor rows = 16
valid output rows        = 13
```

Graph runner 先计算 `padded_batch_size`，将真实输入填进静态 buffer，replay 后只切出 `[:batch_size]`，见 [decode_cuda_graph.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/model_executor/runners/decode_cuda_graph.py:138)。

C++ ACL Graph persistent params 进一步显式区分：

- `meta.actual_num_sequences`：真实有效数量；
- `meta.num_sequences`：capture/replay 的 padded 数量。

写入逻辑见 [acl_graph_persistent_param.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/acl_graph_persistent_param.cpp:1000)，读取 actual batch 的回退顺序见 [acl_graph_persistent_param.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/acl_graph_persistent_param.cpp:99)。

因此 Graph 里的 `num_sequences=16` 不等于服务突然多出 3 条请求。

那三行只是静态容量，必须在 Attention metadata、输出 slice、采样与 KV 写入中被屏蔽。

---

## 30. 为什么 Graph warmup bucket 与 `max_seqs_per_batch` 有关

C++ warmup 先生成：

```text
1, 2, 4, 8, 16, 32, 48, 64, ... , max_seqs_per_batch
```

实现见 [graph_warmup.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/profile/graph_warmup.cpp:46)。

纯普通 Decode 时：

```text
每条 Sequence 贡献一个 token
因此 B_seq == T_q
```

这时 `max_seqs_per_batch` 自然也是最大 Decode token batch 的重要上界，适合用来生成 bucket。

但不要把这个等式推广到：

- Prefill；
- Chunked Prefill；
- Mixed Batch；
- 投机验证；
- Graph fake/padding；
- CP/EP 重排后的物理 shape。

DP 下还会跳过小于 `dp_size` 的某些全局 warmup bucket，见 [graph_warmup.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/profile/graph_warmup.cpp:68)。这与多 rank 图捕获和最小有效分布有关，不代表每个 rank 都至少有一个真实请求。

---

## 31. Prefix Cache 怎样改变 batch，而不改变请求数

Prefix Cache 命中会增加 `n_kv_cache_tokens`，因此减少本步需要抽取的 `q_seq_len`：

```text
q_i = prompt_tokens - cached_prefix_tokens
```

例如两个 prompt 都长 2048：

```text
A: cache miss -> q=2048
B: hit 1792   -> q=256
```

两者仍各是一条 Sequence，但模型 token 成本相差 8 倍。

这说明仅看“并发请求数”无法估算 Prefill 工作量。

Scheduler、KV manager 与 Builder 必须对已缓存 token 数形成一致认知，否则可能重复写 KV、错误设置 position 或跨 Sequence 读取。

Prefix Cache 全链路见 [08-prefix-cache-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/08-prefix-cache-complete-chain.md:1)。

---

## 32. 投机解码为什么让纯 Decode 不再满足 `T=B_seq`

投机解码会为每条 Sequence 提交多个候选 token 进行验证。

Scheduler 的 Decode token budget 使用：

```text
decode_step_tokens = min_speculative_tokens_required + 1
```

beam 和普通路径都在 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:502) 中体现。

若每条 Sequence 本步验证 5 个位置，32 条 Sequence 可能形成：

```text
B_seq = 32
T_q   = 32 * 5 = 160
```

所以“纯 Decode”只描述阶段，不自动保证 `q_max_seq_len=1`。

Engine 也同时检查：

```text
current_batch_forward_type.is_decode()
q_max_seq_len == 1
```

才把 DP shard 标为普通 Decode，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1258)。

当前 Scheduler 在启用投机 token 时关闭 Prefill/Decode Mix，使形状和状态契约更清晰。

投机解码原理与接受率见 [05-speculative-decoding-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/05-speculative-decoding-complete-chain.md:1)。

---

## 33. Beam Search 与 `best_of` 怎样影响 Sequence batch

Beam Search 或 `best_of` 会让一条 Request 拥有多条候选 Sequence。

Decode 调度不能只给请求分配一个 token，而要给每条活跃候选分配：

```text
allocated_seqs += active_sequences.size()
allocated_tokens += decode_step_tokens * active_sequences.size()
```

Beam 的严格预算与 KV block 预估见 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:488)。

例子：

```text
8 个 HTTP 请求
每个 4 条 active beam
普通 Decode

B_req = 8
B_seq = 32
T_q   = 32
```

如果再有投机验证 4 token：

```text
T_q = 32 * 4 = 128
```

这说明请求、Sequence、token 三层计数缺一不可。

---

## 34. VLM 输入是否会推翻这套 batch 语义

不会，但会增加 token row 的来源。

VLM 可能把部分 token embedding 替换为图像/视频 encoder 产生的 embedding，或者携带额外 MM metadata。

Builder 会批量整理 `mm_data`，并在存在输入 embedding 时拼接，见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:1230)。

模型主干仍需要形成与当前 packed token rows 对齐的 hidden states。

但 VLM 的风险更高：

- 一个逻辑媒体标记可能展开为多行 embedding；
- position 可能使用 M-RoPE；
- 空 DP rank 不能只塞 fake token 绕过 metadata；
- encoder DP 与 LLM decoder DP 可能不同。

所以 VLM 需要额外验证 embedding 行数、position、q lengths 与输出选择的一致性，不能只验证文本 token 数。

---

## 35. Schedule Overlap 中的 fake token 也不是新请求

Schedule Overlap 让 CPU 准备下一步调度/metadata 与设备执行当前 step 重叠。

为了在采样结果尚未回到 CPU 时预排下一步，框架可能使用临时 fake token，稍后再替换真实采样 token。

`Batch::process_sample_output()` 的 `replace_fake_token` 分支说明输出更新与假 token 替换，见 [batch.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.cpp:609)。

这里同样必须区分：

```text
Sequence 数没有增加
请求数没有增加
某一位置的 token 值暂时是假值
```

正确性要求 fake token 不得污染：

- 最终用户输出；
- Prefix Cache hash；
- KV/线性状态边界；
- MTP bootstrap；
- 停止条件。

Schedule Overlap 细节见 [09-async-schedule-and-overlap-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/09-async-schedule-and-overlap-complete-chain.md:1)。

---

## 36. 一次完整 forward 的 shape 推导

考虑当前 step 有五条 Sequence：

| Sequence | 阶段 | 未缓存 token | 本步 budget | `q_i` | DP rank |
|---|---|---:|---:|---:|---:|
| A | Prefill | 300 | 256 | 256 | 0 |
| B | Decode | 1 | 1 | 1 | 0 |
| C | Decode | 1 | 1 | 1 | 1 |
| D | Chunked Prefill | 96 | 64 | 64 | 1 |
| E | Decode | 1 | 1 | 1 | 0 |

全局 Scheduler 视角：

```text
B_req 假设 = 5
B_seq      = 5
T_q        = 256 + 1 + 1 + 64 + 1 = 323
```

DP0：

```text
sequences  = [A, B, E]
q_lens     = [256, 1, 1]
B_seq[0]   = 3
T_q[0]     = 258
type       = MIXED
```

DP1：

```text
sequences  = [C, D]
q_lens     = [1, 64]
B_seq[1]   = 2
T_q[1]     = 65
type       = MIXED
```

若每个 DP 副本内部 `TP=4`：

- DP0 的四个 TP rank 都处理 258 个逻辑 token；
- DP1 的四个 TP rank 都处理 65 个逻辑 token；
- 每个 TP rank 只拥有部分权重/heads；
- TP collective 的消息大小随本地 T 增长。

若当前配置因 CP 禁止 Mix，那么上表不会在同一个 step 成立，Scheduler 会把 Prefill 与 Decode 分开调度。这正说明 shape 推导必须先服从有效 BatchMode。

---

## 37. 连续五步例子：真实 batch 为什么一直变化

假设 `max_seqs=4`、`max_tokens=8`，所有普通 Decode 每步一个 token。

初始请求：

```text
A prompt=6, generate=2
B prompt=2, generate=4
C 在 step 1 到达，prompt=3
D 在 step 3 到达，prompt=1
```

一种可能的调度如下：

| Step | 入选内容 | `B_seq` | `T_q` | 说明 |
|---:|---|---:|---:|---|
| 0 | A prefill 6 + B prefill 2 | 2 | 8 | token 预算耗尽 |
| 1 | A dec 1 + B dec 1 + C prefill 3 | 3 | 5 | Mixed |
| 2 | A dec 1 + B dec 1 + C dec 1 | 3 | 3 | A 结束 |
| 3 | B dec 1 + C dec 1 + D prefill 1 | 3 | 3 | D 进入 |
| 4 | B dec 1 + C dec 1 + D dec 1 | 3 | 3 | 三条 Decode |
| 5 | B dec 1 + D dec 1 | 2 | 2 | C 结束 |

从这个表能看出：

- HTTP 到达批次不是模型执行批次；
- 同一 Request 不会固定占据 batch 某个行号；
- 每步都可有不同 `B_seq` 和 `T_q`；
- `q_seq_lens` 和输出映射每步重建；
- Graph 会在 2、4、8 等 bucket 间选择，而不是只 replay 一个 batch=1 的图。

---

## 38. 配置项全景：它们分别影响哪一个“batch”

| 配置/请求参数 | 直接限制或改变 | 不应误解为 |
|---|---|---|
| `max_concurrent_requests` | 服务在途 Request 上限 | forward batch 上限 |
| `max_seqs_per_batch` | 单 step 逻辑 Sequence 预算 | token 数上限 |
| `max_tokens_per_batch` | 单 step query-token 预算 | KV 总 token 上限 |
| `max_tokens_per_chunk_for_prefill` | 单 Prefill Sequence 单步 token 上限 | 全 batch token 上限 |
| `enable_chunked_prefill` | 长 Prefill 是否跨 step | 是否启用 Prefix Cache |
| `enable_mix_batch` | Prefill/Decode 是否可同 batch | 必然生效的最终模式 |
| `max_model_len` | 单 Sequence 生命周期长度边界 | 单步 batch token 数 |
| `best_of` / beam width | 一 Request 的候选 Sequence 数 | 请求并发数 |
| speculative tokens | 一 Decode Sequence 本步验证 token 数 | Sequence 数 |
| `dp_size` | Sequence 分到多个模型副本 | 权重维切分 |
| LLM effective TP | 权重/head 维切分与 collective | 请求切分 |
| `cp_size` | 上下文计算/布局协作 | Scheduler Sequence 翻倍 |
| `sp_size` | 当前主要是 DiT 配置 | LLM 请求 batch 除数 |
| `ep_size` | expert 放置与 token 路由 | HTTP Request 分片 |
| Graph max/buckets | replay 静态容量 | 真实请求数 |
| Prefix Cache | 降低 Prefill 的 `q_i` | 消灭该 Request |
| KV cache 容量 | 决定可持续驻留/分配的 Sequence | 只影响模型长度 |

这些配置分别属于服务、调度、请求、模型并行、缓存和执行图六个层次。

调参时最危险的做法是把它们都叫“batch size”，然后只改一个参数期待所有层同时变化。

---

## 39. Batch 对通信量的影响

### 39.1 TP 通信

Row Parallel 常见 AllReduce 的激活规模近似：

```text
bytes_tp ~ T_local * H * dtype_bytes
```

Column Parallel 的 lm_head 若 gather output，则规模还与词表 shard 相关。

纯 Decode `T_local` 通常等于本地活跃 Sequence 数，消息较小但频率高、延迟敏感；Prefill `T_local` 大，消息更偏带宽敏感。

### 39.2 DP 通信

纯复制式 DP 的 Attention/Dense 主干不需要在副本间 AllReduce hidden states；每个 DP rank 独立处理自己的 Sequence。

但 DP 与 EP/mapping/Graph 组合时，需要交换 token counts、padding metadata 或参与 collective。不能把“DP 理论独立”理解为实现中绝无跨 DP 通信。

### 39.3 CP 通信

CP 的通信量与 query/context/KV 分片方案有关，可能包括 Prefix KV AllGather、Attention 局部结果归并或 plan metadata。

它受上下文长度影响，不能仅用 `B_seq` 估算。

### 39.4 EP 通信

EP AllToAll 的主规模与：

```text
T_local * top_k * expert_hidden / packing
```

相关，并受专家负载不均和 padding 放大。

因此性能分析至少要同时记录 `B_seq`、`T_q`、各 DP 的 `T_q[d]`、上下文长度分布和 top-k。

---

## 40. Batch 对 overlap 的影响

Overlap 的目标是隐藏不同资源阶段：

- CPU Scheduler 与设备 compute；
- H2D metadata copy 与前一 step kernel；
- TP/EP/CP collective 与 matmul；
- KV transfer 与计算；
- PD 网络传输与两侧服务。

batch 太小时：

- kernel launch 和 collective latency 占比高；
- 可重叠的计算窗口不足；
- Graph 的收益通常更明显。

batch 太大时：

- 单 step latency 变长；
- 新请求要等更久才插入；
- Prefill 可能阻塞 Decode 的 TPOT；
- metadata 和 H2D copy 也会增长；
- 多流可能争用同一 HBM/通信链路。

Chunked Prefill 与 Mix Batch 的价值之一，就是控制长 Prefill 的单步占用，为 Decode 留预算和插入点。

但 overlap 不改变逻辑正确性公式：真正提交给模型的 token 仍必须与本步 Sequence 状态、KV slot 和输出映射一致。

---

## 41. Batch 对显存占用的影响

显存应分为常驻与步级两类。

### 41.1 常驻、主要不随当前 step batch 线性变化

- 模型权重；
- 预分配 KV cache pool；
- Graph 静态输入/输出与捕获 workspace；
- 通信 buffer；
- allocator reserve。

### 41.2 步级、强烈受 `T_q` 影响

- hidden states；
- Q/K/V 临时张量；
- MLP gate/up 中间值；
- 动态量化 scale；
- MoE routing、排序与 expert workspace；
- logits selected rows；
- Attention workspace。

一个粗略 Dense 激活项是：

```text
activation_bytes ~ T_q * (若干个 H 或 intermediate_size) * dtype_bytes
```

因此 `B_seq=1, T_q=4096` 的 Prefill 可能比 `B_seq=256, T_q=256` 的普通 Decode 使用更多步级 activation。

但后者有 256 条独立 KV 链和 metadata，且它们的生命周期 KV 总量可能更高。

Graph bucket `G > actual T` 时，静态 activation/workspace 可能按 G 分配；这解释了捕获多个大 bucket 会增加常驻显存。

完整显存模型见 [17-memory-and-vram-management-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/17-memory-and-vram-management-complete-chain.md:1)。

---

## 42. Batch 对数值精度有什么影响

理论上，把独立 Sequence pack 在一起不应改变每条 Sequence 的数学结果。

Attention 有严格隔离边界，Dense/MLP 逐行计算，输出再按映射回填。

但工程上仍可能出现 batch-dependent 数值差异：

- 不同 Graph/eager kernel 路径；
- 不同 bucket shape 选择不同 tiling；
- TP/EP collective reduction 顺序变化；
- Prefill 与 Decode 使用不同 Attention kernel；
- W8A8 动态量化的 per-token scale 与实现细节；
- MoE token 排序导致浮点累加次序变化；
- CP 分片与归并顺序变化。

这些差异应是可解释的浮点误差，而不是 Sequence 串扰。

Kimi K3 W8A8 动态线性对每 token activation 动态量化，见 [linear.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/layers/linear.py:142)。batch T 改变不应让某 token 使用另一 token 的 scale；测试应验证这一点。

正确性验收不应只比较最终文本，还应比较：

- 单卡与并行 hidden/logits；
- 不同 batch composition 下同一请求的 logits；
- eager 与各 Graph bucket；
- Prefill、Chunked Prefill、Prefix hit 后的边界 logits；
- DP 空 shard 和不均衡 shard。

---

## 43. 最重要的正确性不变量

### 不变量 1：packed token 数守恒

忽略显式 padding/fake row：

```text
token_ids.numel() == sum(q_seq_lens)
```

### 不变量 2：Sequence 数与 metadata 行数一致

Eager 下：

```text
meta.num_sequences == Batch::size()
len(q_seq_lens) == len(kv_seq_lens) == block_tables.rows
```

Graph 下必须额外区分 actual 与 padded。

### 不变量 3：每个 query row 只看自己的上下文

`q_cu_seq_lens`、KV lengths、block table 与 slot mapping 必须共同指向同一 Sequence。

### 不变量 4：KV 写入数量与位置一致

每个真实 query token 应有合法、互不冲突的 `new_cache_slot`；padding/fake row 不得覆盖真实 KV。

### 不变量 5：输出映射不随组批顺序丢失

每个 sampled row 必须准确回到原 `Sequence*` 与 sample slot。

### 不变量 6：DP rank 生命周期稳定

Sequence 的 KV 所在 rank 与模型执行 rank 必须一致。

### 不变量 7：Graph padding 不产生用户语义

padding row 不得生成用户 token、改变停止条件或占用真实请求计数。

---

## 44. 性能调优：应该先看哪个数字

### 场景 A：Prefill 吞吐不足

先看：

- 每步 `T_q` 与 `q_max_seq_len`；
- chunk size；
- Prefix Cache 命中后实际 q；
- TP/CP collective 带宽；
- MLP/MoE activation 峰值；
- 是否被 `max_tokens_per_batch` 限得过小。

不要只看请求数。

### 场景 B：Decode TPOT 高

先看：

- 本地活跃 Decode `B_seq`；
- DP 是否不均衡；
- KV context 长度分布；
- Graph bucket 命中与 padding 比；
- collective latency；
- 是否有大 Prefill 混入；
- batch 太小导致设备利用率低，还是太大导致单步延迟高。

### 场景 C：显存 OOM

先区分：

- 权重/KV pool 常驻 OOM；
- Graph capture 常驻增长；
- 大 `T_q` 的步级 activation 峰值；
- MoE top-k/padding workspace；
- 多 bucket 或多流 buffer。

### 场景 D：吞吐随 DP 增大不线性

检查：

- Sequence 是否均匀固定到 DP ranks；
- prompt/上下文长度是否倾斜；
- 空 shard 是否仍参加 EP/Graph collective；
- 单副本 batch 是否小到无法饱和设备；
- 服务入口并发是否足以喂满所有副本。

---

## 45. 参数调大以后为什么性能可能反而变差

### `max_concurrent_requests` 调大

只会允许更多在途请求。

可能提高排队深度与设备饱和度，也可能增加：

- host 状态；
- KV 驻留压力；
- 排队延迟；
- 抢占与恢复；
- 流式连接资源。

### `max_seqs_per_batch` 调大

可能提高 Decode 吞吐，但也会：

- 增大单 step 延迟；
- 扩大 Graph 最大 bucket；
- 增加 metadata、block table 与采样开销；
- 让小延迟请求等更久。

### `max_tokens_per_batch` 调大

可能提高 Prefill matmul 效率，但会：

- 增大 activation 峰值；
- 拉长 step；
- 压迫 Decode TPOT；
- 放大通信 burst。

### chunk 调大

减少一条 prompt 的 step 数和重复调度开销，但降低 Decode 插入频率并提高峰值。

这些参数不是越大越好，而是在吞吐、首 token 延迟、TPOT、显存与公平性之间取点。

---

## 46. 可观测性应该同时记录哪些维度

一个有用的 step trace 至少应包含：

```text
step_id
service_inflight_requests
waiting_requests
scheduled_requests
scheduled_sequences
prefill_sequences
decode_sequences
q_tokens_global
q_tokens_per_dp
q_max_seq_len
kv_max_seq_len
batch_forward_type_per_dp
actual_graph_rows
graph_bucket_rows
selected_logits_rows
sampled_outputs
kv_blocks_used/free per dp
tp/cp/ep collective bytes and time
```

仅有 `batch_size=32` 的日志几乎不可诊断，因为无法判断它是：

- 32 个请求；
- 32 条 Sequence；
- 32 个 token；
- bucket 32；
- 32 个 routed items；
- 还是 DP 全局总量。

命名时应直接写单位，例如 `scheduled_sequences`、`query_tokens`、`graph_bucket_tokens`。

---

## 47. 测试应怎样覆盖这套语义

### 47.1 BatchInputBuilder 单元测试

覆盖：

- 多 Prefill 不同 q length；
- 多 Decode `q_i=1`；
- Mixed `[9,1,1,4]`；
- Prefix Cache 后只抽未缓存 token；
- selected token indices；
- block table 与 new slots；
- Graph padding actual/padded 区分。

现有 Mixed 基准见 [batch_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/batch/batch_test.cpp:620)。

### 47.2 Scheduler 测试

覆盖：

- Sequence 预算先满；
- token 预算先满；
- DecodeFirst 的 Mixed；
- PrefillFirst 不混；
- beam 多 Sequence 预算；
- speculative 每 Sequence 多 token；
- KV block 不足与抢占。

策略测试入口见 [scheduler_policy_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/scheduler/scheduler_policy_test.cpp:204)。

### 47.3 并行测试

覆盖：

- DP 不均衡 `[T0,T1]`；
- 空 DP shard；
- DP+TP 输出对齐；
- CP q/KV layout；
- EP top-k 与 padding；
- collective 参与 rank 一致。

### 47.4 Graph 测试

至少跑 actual batch：

```text
1, 2, 3, 4, 7, 8, 9, 15, 16, 17, max-1, max
```

重点是 bucket 边界前后，并与 eager logits 对齐。

### 47.5 服务 E2E

并发发送不同 prompt 长度、不同 max_tokens、Prefix hit/miss、best_of 请求，逐请求核对输出，不能只看服务没有 crash。

---

## 48. 当前 Kimi K3 相关支持状态

本文的 batch/forward 基础链是框架通用机制，但 Kimi K3 的具体组合必须按当前验证状态表述。

已确认：

- Kimi K3 Python 模型以 packed `input_ids` 形成 `[T,H]` hidden states；
- TP2 的 5-layer Prefill 对齐已有验证；
- 基础 DP2+TP2 在线服务请求 smoke 已通过；
- VLM backend 的空 DP shard 不注入无 metadata fake token 的保护已进入当前基线。

尚不能由这些结果推出：

- 完整 93-layer 所有 batch shape 均验证；
- Chunked Prefill + KDA/MLA 所有边界已验证；
- Graph 所有 bucket replay 与 eager 全面对齐；
- CP、投机解码、Schedule Overlap、Prefix Cache 任意组合已验证；
- 长时间高并发 DP 负载均衡与稳定性已完成 benchmark。

所以本文对通用机制给出确定结论，对 Kimi K3 组合支持保持“已验证/代码存在/未验证”三层区分。

---

## 49. 常见误解逐条纠正

### 误解 1：Prefill batch 大，Decode batch 固定是 1

纠正：Prefill 通常 `q_i` 大；Decode 通常每 Sequence `q_i=1`，但 `B_seq` 可很大。

### 误解 2：`max_concurrent_requests=200` 就会一次 forward 200 个请求

纠正：它只限制服务在途 Request；Scheduler 还有 Sequence/token/KV/延迟预算。

### 误解 3：`max_seqs_per_batch` 就是 token tensor 第一维

纠正：只在普通纯 Decode、无 padding/投机时常有 `B_seq=T_q`。

### 误解 4：Graph bucket 16 表示 16 个真实请求

纠正：可能只有 13 条真实 Decode Sequence，静态 tensor padding 到 16。

### 误解 5：DP2 后每个 rank 必然拿一半 batch

纠正：Sequence 生命周期固定 rank，长度和完成时间不同会造成 token/Sequence 不均衡。

### 误解 6：TP4 会把 128 条 Sequence 分成每卡 32 条

纠正：TP rank 看相同 128 个逻辑 token，切的是模型维。

### 误解 7：SP 配置能直接切 LLM 在线 Sequence

纠正：当前 `sp_size` 明确主要用于 DiT；算子 sequence parallel 也不是 Request Sequence。

### 误解 8：压平 token 后请求身份丢了

纠正：Attention 用长度与 block table 保边界，输出用 selected index 与 `Sequence*` 映射回填。

### 误解 9：batch 越大吞吐一定越高

纠正：会受 step latency、显存、padding、collective、KV 长度、DP 不均衡和服务 SLO 约束。

---

## 50. 一套不再说错 batch 的分析模板

以后看到任何 `batch_size=N`，按以下顺序追问。

### 第一步：单位是什么

```text
Request / Sequence / query token / KV token / routed token / Graph row ?
```

### 第二步：作用域是什么

```text
服务全局 / Scheduler step / DP local / TP group / CP group / EP group ?
```

### 第三步：logical 还是 physical

```text
真实有效数据 / fake row / alignment padding / Graph bucket capacity ?
```

### 第四步：阶段是什么

```text
Prefill / Chunked Prefill / Decode / speculative verify / Mixed ?
```

### 第五步：时间尺度是什么

```text
当前 step / 请求生命周期 / 一秒统计窗口 / profile warmup ?
```

### 第六步：对应哪个源码字段

```text
RateLimiter counter
Batch::size()
token_ids.numel()
meta.num_sequences
meta.actual_num_sequences
dp_global_token_nums
q_seq_lens
graph bucket
```

回答完这六步，绝大多数“batch size 对不上”的问题都会自然消失。

---

## 51. 推荐源码阅读顺序

第一遍只建立对象和预算：

1. [service_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/service_config.h:69)：服务并发；
2. [scheduler_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/scheduler_config.h:62)：Sequence/token 预算；
3. [sequence.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.h:246)：Sequence 身份；
4. [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:216)：每步重组；
5. [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:469)：Decode 预算。

第二遍追真实 forward shape：

1. [batch_factory.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_factory.cpp:32)：DP 分 batch；
2. [batch.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch.cpp:75)：Batch 类型；
3. [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:671)：每 Sequence q length；
4. [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:734)：压平 token；
5. [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:1189)：ForwardInput metadata。

第三遍追模型、并行与 Graph：

1. [kimi_k3_text.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/models/kimi_k3_text.py:1537)：`[T,H]` 模型主干；
2. [npu_paged_attention.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/attention/npu_paged_attention.py:231)：Attention 分阶段执行；
3. [linear.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/layers/linear.py:25)：TP Linear；
4. [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1208)：DP token counts；
5. [decode_cuda_graph.py](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/python/model_executor/runners/decode_cuda_graph.py:33)：Graph bucket；
6. [acl_graph_persistent_param.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/acl_graph_persistent_param.cpp:99)：actual/padded batch。

最后用 [batch_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/batch/batch_test.cpp:620) 的 `[9,1,1,4]` 例子把整条链手算一遍。

---

## 52. 读完本文应该能回答的问题

1. 为什么 `max_concurrent_requests` 不是模型 batch size？
2. Request 和 Sequence 为什么不是一一对应？
3. `max_seqs_per_batch` 与 `max_tokens_per_batch` 分别限制什么？
4. 为什么一条 Prefill Sequence 可以形成很大的模型输入？
5. 为什么 128 条 Decode Sequence 的 query length 都是 1，但模型输入第一维是 128？
6. Mixed Batch 的 `T_q` 怎样求和？
7. packed token 后 Attention 如何防止跨请求串上下文？
8. 为什么只有少量 hidden rows 进入 lm_head 和 sampling？
9. Continuous Batching 如何在每步改变 batch composition？
10. DP 为什么分 Sequence，而 TP 不分请求？
11. CP 与 SP 为什么不能简单理解为把 Request batch 做除法？
12. EP 的 `T*top_k` 为什么不是新的 Sequence 数？
13. Graph 为什么需要很多 bucket？
14. actual batch=13 与 graph batch=16 为什么能同时成立？
15. Prefix Cache、投机解码、beam、PD 与 overlap 分别改变哪一层计数？
16. 怎样从 `q_seq_lens` 推导真实模型 forward shape？
17. 怎样为 batch 正确性设计跨 eager/Graph/DP 的测试？

如果这些问题都能独立回答，就已经建立了理解 xLLM 在线推理数据面的核心坐标系。

---

## 53. 最终总结

xLLM 中没有一个脱离语境的“真实 batch size”。

最接近模型计算量的数字通常是：

```text
T_q = token_ids.numel() = sum(q_seq_lens)
```

最接近调度并发度的数字通常是：

```text
B_seq = Batch::size()
```

最接近服务压力的数字是：

```text
C_req = 尚未完成的在途 Request 数
```

最接近 Graph 静态执行容量的是：

```text
G = 覆盖 actual token/Sequence 数的 bucket
```

四者必须分开。

Prefill 的特点是一条 Sequence 可贡献很多 query token；普通 Decode 的特点是每条 Sequence 贡献一个 token，但可以同时有很多条 Sequence；Chunked Prefill 与 Mixed Batch 让二者在同一步求和；DP 把 Sequence 固定分到不同模型副本；TP 切模型维而不切请求；CP 重排/协同上下文；EP 把 token 扩展为 token-expert 路由项；Graph 用 padding bucket 把动态 Continuous Batch 投射到静态 shape。

最后，Continuous Batching 能够正确工作的根本原因不是请求顺序固定，而是每一步都完整重建了三类映射：

```text
Sequence -> packed query rows
packed query rows -> 独立 Attention/KV 边界
selected output rows -> 原 Sequence / Request
```

这三条映射，加上 Scheduler 的 Sequence/token 双预算，构成了 xLLM 从在线并发到模型真实 forward 的完整 batch 机制。
