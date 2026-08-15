# xLLM 显存与内存管理完整链路：从启动预算到请求运行时的每一块字节

> 本文是 xLLM 知识库的显存专题。它回答一个比“KV Cache 占多少显存”更基础的问题：xLLM 一张卡上的显存到底由什么组成，启动时按什么顺序申请，为什么服务启动后看起来会逐渐把显存吃满，哪些配置会改变容量，为什么 `free`、`reserved`、`allocated` 和 xLLM 自己统计的容量不是同一个数。
>
> 源码基线：当前工作区 `xllm/`。本文讨论通用 LLM/VLM/混合注意力路径，DiT 只在需要说明 encoder/latent/工作区时提及。Kimi K3 的模型状态与 `research/PROJECT.md` 保持一致：当前已有五层 TP2 prefill 对齐和基础服务冒烟，但还没有把全层长上下文、Graph、chunked prefill、Prefix Cache 和显存稳定性写成完成验收。

建议先把本文当作容量总账，再按问题进入专项文章：KV 的物理组织见 [07 Paged KV](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/07-paged-kv-and-paged-attention-complete-chain.md:1)，缓存复用见 [08 Prefix Cache](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/08-prefix-cache-complete-chain.md:1)，异步生命周期见 [09 Async Schedule](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/09-async-schedule-and-overlap-complete-chain.md:1)，并行副本、FlashComm 与 Rolling Load 见 [10 分布式并行](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/10-distributed-parallelism-and-communication.md:1)，量化后的字节数见 [12 量化](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/12-quantization-complete-chain.md:1)。

## 1. 先给结论：显存是一张分层预算表

读 xLLM 显存最重要的一句话是：

> 启动阶段先驻留模型和运行时的常驻对象，再把每个 Worker 观察到的剩余显存换算成 KV/linear-state 容量；请求阶段主要从已经分配的 block 池中取页，只有 Graph、workspace、通信或模型特性需要时才会继续出现新的临时峰值。

可以把每张设备的物理显存粗略写成：

~~~text
HBM_used
  = weights
  + KV_cache_and_linear_state
  + activations_and_workspace
  + graph_persistent_buffers
  + communication_buffers
  + runtime_temporaries
  + allocator_reserved_but_not_active
~~~

而 xLLM 的容量规划更接近：

~~~text
cache_budget
  = min(
      observed_free_after_model,
      total_memory * max_memory_utilization - already_reserved_weight_budget,
      max_cache_size(if > 0)
    )
~~~

实际代码通过“先读取空闲显存，再扣掉总显存保留比例”实现，具体不是一个独立的“模型权重大小配置”。因此必须同时看真实 allocator 状态、Worker 观测和 xLLM 的 KV capacity 日志。

## 2. 先区分五个容易混淆的数字

### 2.1 Physical total memory

设备驱动报告的卡容量，例如 80 GiB。`Device::total_memory()` 通过后端 API 获取，不表示 xLLM 可以安全使用的额度。见 [device.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/platform/device.cpp:99)。

### 2.2 Driver free memory

驱动在某个时刻报告的空闲空间。它包含没有被当前进程映射、也没有被其它进程占用的空间；不同后端的 caching allocator 可能还持有可复用但未归还驱动的缓存。

### 2.3 Allocated / active

PyTorch 或后端 allocator 当前被活跃 tensor 使用的部分。KV cache、权重和临时 activation 释放后，块可能仍保留在 allocator cache 中，所以 driver free 不一定立刻增加。

### 2.4 Reserved

allocator 已经向驱动申请、但其中一部分暂时没有 active tensor 的空间。CUDA Graph private pool、PyTorch caching allocator、NPU caching allocator 都可能让 reserved 大于 allocated。

### 2.5 xLLM logical capacity

KV cache capacity 是“可供 BlockManager 使用的逻辑 block 数 × 每 block 字节”，不是驱动报告的物理 allocated 数。它还会扣除 linear state、padding、RDMA scale 对齐或 DeepSeek V4 固定池。

| 数字 | 谁产生 | 能回答什么 | 不能回答什么 |
| --- | --- | --- | --- |
| total | 设备 API | 这张卡物理上多大 | xLLM 能用多少 |
| free | 设备 API | 当前还未被驱动占用多少 | caching allocator 可否复用 |
| allocated/active | allocator snapshot/stat | 活跃 tensor 多少 | reserved 是否能立即还给驱动 |
| reserved | allocator | 进程占住多少地址/段 | 真实模型有效数据多少 |
| KV capacity | xLLM 估算器 | 可容纳多少 block/token | 所有 workspace 峰值是否足够 |

## 3. 显存组成总表

### 3.1 权重常驻

每个 TP rank 只持有本地分片，但它仍包含 embedding、decoder layer、norm、LM head、MoE expert 分片或副本。权重通常是 BF16/FP16/FP8/INT8 等 dtype；量化还需要 scale、zero-point 或反量化临时 buffer。

### 3.2 KV 和线性状态

普通 Attention 主要是 K/V；GQA/MQA 会减少本地 KV heads；MLA 可能只保存 latent 与 rope 部分；KDA/GDN 则增加 convolution state 和 recurrent SSM state。xLLM 将这些容量统一表达为 `KVCacheCapacity`，再由 `KVCacheShape` 展开成 tensor shape。

### 3.3 激活和模型工作区

包括当前 batch 的 hidden states、Q/K/V 临时张量、MoE dispatch/combine、attention plan、sampler、RMSNorm/GEMM workspace，以及 kernel 需要的 scratch。它们通常随 `max_tokens_per_batch`、hidden size、top-k、prefill/decode shape 增长。

### 3.4 Graph 持久化 buffer

CUDA/ACL Graph 为了固定地址和 shape，会预分配 tokens、positions、block table、hidden states、paged metadata 等 persistent tensor；多 shape capture 还可能产生 graph private pool。CUDA Graph 的 persistent 参数可在 [cuda_graph_executor_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/cuda_graph_executor_impl.cpp:90) 看到。

### 3.5 通信 buffer

TP/EP/CP/DP 会产生 all-reduce、all-gather、reduce-scatter、all-to-all 的发送和接收 buffer；FlashComm、MoE dispatch、PD KV transfer 和 HCCL/NCCL runtime 也会占用空间。通信 buffer 的峰值常在 prefill 或 MoE，而不是单 token decode。

### 3.6 Host、pinned 和统一虚拟内存

主存可能放：checkpoint mmap、rolling weight buffer、host KV offload、共享内存 request buffer、pinned D2H/H2D staging、Mooncake/RDMA 注册区。它们不计入卡上 HBM，但会影响启动、传输和是否能把更多 KV 留在设备。

### 3.7 Allocator 与碎片

驱动和框架都可能因为对齐、页粒度、Graph private pool 或长短 tensor 交错而产生内部碎片。显存“还有 4 GiB free”不代表能成功申请一个 4 GiB 连续 tensor；这也是 NPU huge page、VMM 和 block 化设计存在的原因。

## 4. xLLM 的显存状态图

~~~text
启动入口 xllm.cpp
  |
  +--> 设备上下文 / ProcessGroup / allocator 初始化
  |
  +--> LLMEngine::init
         |
         +--> Worker::init_model
         |      +--> ModelContext / Model object
         |      +--> load_model: 本 rank 权重进入设备
         |      +--> optional rolling / sleep / xtensor weight pages
         |
         +--> estimate_kv_cache_capacity
         |      +--> 每个 Worker 读 free/total
         |      +--> utilization / max_cache_size 限额
         |      +--> KVCacheCapacity: block/slot/linear/indexer
         |
         +--> allocate_kv_cache
                +--> KVCacheShape
                +--> BlockManagerPool / Composite leaves
                +--> Worker::allocate_kv_cache
                       +--> K/V / MLA / index / conv / SSM tensor
                       +--> transfer / huge page / sleepable / xtensor

请求进入后
  Request -> Scheduler -> BlockManager allocate/reuse
           -> BatchInputBuilder metadata
           -> model forward workspace / Graph bucket
           -> KV write + block usage update
           -> finish/cache/deallocate/offload
~~~

这张图的关键是：`BlockManager` 的 block 分配通常不再申请新的 HBM tensor；它只是把预先分配好的 KV tensor 切成逻辑页并记录 id。Graph/workspace/通信是运行时可能继续增长的另一条路径。

## 5. 源码地图：先看哪些文件

| 层次 | 文件 | 读什么 |
| --- | --- | --- |
| 配置 | [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:34) | block、cache 上限、dtype、prefix、linear、xtensor |
| 选项合并 | [xllm.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/xllm.cpp:120) | 配置如何进入 runtime Options |
| 设备测量 | [device.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/platform/device.cpp:99) | 各 backend total/free API |
| 初始化顺序 | [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:135) | model -> estimate -> KV allocate |
| 权重加载 | [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1371) | model context、load、rolling |
| 容量估算 | [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:342) | free memory、utilization、max cache |
| 公式实现 | [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:604) | slot、block、linear、DSV4 |
| 形状 | [kv_cache_shape.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:45) | K/V/index/conv/SSM tensor shape |
| 实际 tensor | [kv_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache.cpp:265) | 逐层创建 KV 实现 |
| block 池 | [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:22) | DP 池、Composite leaves、分配/释放 |
| flat block | [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:42) | free list、padding block、prefix eviction |
| host offload | [hierarchy_block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/hierarchy_block_manager_pool.cpp:1) | HBM/host block 对 |
| Graph | [cuda_graph_executor_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/cuda_graph_executor_impl.cpp:35) | persistent buffer、private pool |
| XTensor | [xtensor_allocator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/xtensor/xtensor_allocator.cpp:186) | 物理页池和共享映射 |
| 监控 | [device_monitor.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/device_monitor.cpp:20) | weight/KV/activation 账本 |

这些文件分别回答“预算从哪来”“物理 tensor 在哪申请”“请求如何消费既有容量”三个问题。

## 6. 配置入口和默认值

### 6.1 KVCacheConfig 的公共默认值

当前 `KVCacheConfig` 默认：

| 配置 | 默认值 | 直接影响 |
| --- | --- | --- |
| `block_size` | 128 | 每个逻辑 block 的 token 数和尾部浪费 |
| `max_cache_size` | 0 | 0 表示自动计算，否则限制 KV budget |
| `max_memory_utilization` | 0.8 | 总显存给模型推理（权重+KV）的比例 |
| `kv_cache_dtype` | `auto` | KV dtype，`int8` 当前只支持 MLU 且有限制 |
| `indexer_cache_dtype` | `auto` | DSA/indexer cache dtype |
| `enable_prefix_cache` | true | 共享 prefix block、LRU、eviction |
| `enable_in_batch_prefix_cache` | false | 同一 batch 的 prefill full block 是否入 cache |
| `max_linear_state_cache_slots` | 0 | 线性 attention state 槽位自动或显式容量 |
| `enable_xtensor` | false | 物理页池和权重/KV 的 VMM 映射 |
| `phy_page_granularity_size` | 2 MiB | XTensor 物理页粒度 |

定义和 gflag 见 [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:43) 与 [kv_cache_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.cpp:20)。

### 6.2 Runtime Options 有另一组默认值

`runtime::Options` 的 `max_memory_utilization` 默认是 0.9，`max_tokens_per_batch` 默认 20480；公共 gflag/config 默认分别是 0.8 和 10240。服务入口会把公共配置写入 Options，所以不要只读 [options.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/options.h:35) 就下结论。

### 6.3 其他直接相关配置

- `max_tokens_per_batch`、`max_seqs_per_batch`：Graph persistent buffer、attention workspace、线性 state 自动槽位上限。
- `max_tokens_per_chunk_for_prefill`：chunked prefill 的单拍激活/metadata 峰值，并且 KDA linear prefix cache 要求 block 对齐。
- `host_blocks_factor`：主机 KV block 池倍数，可能增加 pinned host 内存，减少 HBM 常驻。
- `enable_graph`、`max_tokens_for_graph_mode`、`enable_graph_vmm_pool`：Graph 常驻 buffer 和 private pool。
- `flashinfer_workspace_buffer_size`：CUDA FlashInfer workspace 上限，见 [model_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/model_config.h:45)。
- `num_speculative_tokens`：draft/target 的 KV、embedding、linear state 和临时 output。
- `dp_size`、`tp_size`、`cp_size`、`ep_size`：每 rank 权重分片、local heads、通信 buffer 与 KV capacity。

## 7. 启动时真正发生的顺序

### 7.1 进程和设备上下文

`xllm.cpp` 先读取配置、构造 Options，并在 Worker 中初始化设备上下文和 stream；`Device::init_device_context()` 对 NPU 还区分 Python model 是否已经初始化 runtime。这个阶段通常产生 runtime context、通信库句柄、默认 stream 和 allocator 状态，但尚未按请求分配 KV block。

### 7.2 Worker 建立模型对象

`LLMEngine::init()` 首先调用 `init_model()`，而不是先建 KV cache。见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:135)。

Worker 通过 [WorkerImpl::init_model](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1371) 读取 ModelArgs/QuantArgs，构造 ModelContext，创建 model/executor，再按 `WAKEUP` 或 sleep/rolling 状态加载权重。

### 7.3 本 rank 权重进入设备

普通模式最终调用 `model_->load_model(loader)`；TP/EP 由模型 loader/parallel args 决定每个 rank 实际持有哪些参数。权重大小不是 checkpoint 文件大小的简单等价：分片、dtype、量化、MoE expert、临时反量化 buffer 和 alignment 都会改变 HBM 峰值。

### 7.4 模型加载后计算 KV 预算

权重已经占用显存后，Engine 才调用 `estimate_kv_cache_capacity()`。它逐个 Worker 查询 available/total，取跨 worker 的最小可用预算，应用 utilization 和 max cache 限额，再进入 `estimate_kv_cache_capacity()` 公式。见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:342)。

### 7.5 实际分配 KV tensor

容量估算完后，Engine 构造 `KVCacheShape`、`BlockManagerPool`，并并行调用每个 Worker 的 `allocate_kv_cache_async()`。见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:475) 和 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1580)。

### 7.6 Graph 并不一定在启动时全部申请

很多 Graph 是按 bucket/key lazy capture；首次遇到 shape 时才创建 Graph private/persistent buffer。因而服务刚启动时显存可能低于稳定运行后，第一次大 prefill 或 decode bucket 可能让 reserved/allocated 突然上升。

## 8. `max_memory_utilization` 到底怎样计算

### 8.1 非 XTensor 路径

Worker 的容量探测逻辑是：先清理 caching allocator 可回收缓存，再读取 `free_memory` 和 `total_memory`，同时把 NPU caching allocator 的 cache bytes 加回 available。见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:542)。

Engine 随后做：

~~~cpp
if (max_memory_utilization < 1.0) {
  buffer_memory = total_memory * (1.0 - max_memory_utilization);
  available_memory -= buffer_memory;
}
if (max_cache_size > 0) {
  available_memory = min(available_memory, max_cache_size);
}
~~~

这是等价伪代码，原始实现见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:400)。

### 8.2 为什么要扣总显存的 reserve

如果一张 80 GiB 卡在权重加载后还剩 42 GiB，`max_memory_utilization=0.8` 会扣 16 GiB 的保留额度，KV 估算最多看见约 26 GiB，而不是把 42 GiB 全给 KV。这样为 workspace、通信、allocator 碎片和运行时峰值留空间。

### 8.3 `max_cache_size` 的单位

源码和 gflag 使用 bytes；CLI 文档有时用 MB/GiB 的人类单位展示。实际命令应确认启动配置 dump 和日志中的 `readable_size`，不要把“24”误当作 24 GiB。

### 8.4 多 worker 的最小值原则

Engine 取所有 Worker 可用预算的最小值。某个 rank 因 MoE 权重、旧进程、通信 buffer 或其它服务少了显存，整组 KV capacity 都按最小 rank 规划，避免某一 rank 先 OOM。

### 8.5 这不是严格的物理上限证明

容量估算发生在 Graph lazy capture、后续 workspace warmup 和某些 backend plan 创建之前；它只是可控的 KV budget，不保证整个服务任意请求都不会达到峰值 OOM。因此调 `max_memory_utilization=1.0` 风险很高。

## 9. XTensor 模式的不同预算逻辑

### 9.1 物理页池先于模型

启动时若 `enable_xtensor`，`xllm.cpp` 会调用 `XTensorAllocator::init_phy_page_pools()`。单卡模式按当前 available/total、utilization、max_cache_size 划出固定数量物理页；多节点模式取所有 worker 的最小 available/total 后广播同一个 page count。见 [xllm.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/xllm.cpp:518) 和 [xtensor_allocator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/xtensor/xtensor_allocator.cpp:186)。

### 9.2 权重和 KV 共享物理页池

LLMEngine 初始化时为 model 注册 page list，并按总权重大小/TP 分片预留 weight pages；KV tensor 通过 XTensorAllocator 映射同一个物理池。于是 XTensor 的“可用 KV”不是独立于权重的另一块显存，而是和权重共同竞争 page pool。

### 9.3 页粒度造成的内部浪费

默认 `phy_page_granularity_size=2 MiB`。实际权重、KV、映射对齐到 page 粒度，尾部不足 2 MiB 的空间会成为内部碎片；页粒度越大，管理开销越低，但小 tensor/小 cache 的浪费越大。

### 9.4 XTensor 与普通 KV block 的区别

普通模式：先分配一个大 KV tensor，再由 block id 索引。

XTensor：逻辑 tensor 通过 page allocator 取得 offset，物理 page 可以在模型/DP/进程间映射。BlockManager 仍然管理 token/block 语义，但 tensor storage 的生命周期由 PageAllocator/PhyPagePool 负责。

## 10. 权重显存：模型规模不等于单卡权重

### 10.1 估算下界

~~~text
weight_bytes_lower_bound
  = parameter_count_on_rank × bytes(weight_dtype)
~~~

实际常驻还要加：

- embedding/LM head 是否 tied、是否复制。
- MoE expert 本地分片与 redundant expert。
- quant scale/zero-point、packed weight layout。
- Python/C++ wrapper 的参数视图和转换临时量。
- loader 在加载期间的 CPU mmap、pinned staging 或短时双份 tensor。

### 10.2 TP/EP 对权重的影响

TP 通常让线性层输出/输入维度按 rank 分片，但某些 embedding、norm、LM head 或 replicated expert 不一定按同一比例下降。EP 将专家权重分散到 expert rank，却会引入 dispatch/combine buffer；EPLB redundant experts 会牺牲一部分显存换负载平衡。

### 10.3 Rolling Weight Loading

NPU rolling load 通过 host-pinned 权重和有限层缓存，把 effective device weight budget 从全部 decoder 权重变成 non-decoder 常驻加若干 layer slot；[llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:342) 的 `get_effective_xtensor_weight_size` 记录了这条预算逻辑，Worker 的 rolling runtime 初始化见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1560)。

它节省的是权重常驻 HBM，不是免费节省总内存：每层执行需要 H2D、host pinned buffer 和 load stream，overlap 还可能提高峰值。

### 10.4 Sleep mode

RL sleepable allocator 为权重或 KV 保留虚拟地址、按物理 chunk map/unmap；deep sleep 可以释放物理 HBM，wake_up 再映射。见 [sleepable_allocator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/platform/sleepable_allocator.cpp:40)。

Sleep mode 当前对 standard K/V 有严格限制，不支持 linear/indexer/quantized KV；不能把它当作所有模型的通用显存压缩。

## 11. KV Cache 的统一容量公式

### 11.1 普通 K/V

单 token、单 rank 的近似字节数：

~~~text
KV_bytes_per_token
  = 2(K,V) × n_layers × n_local_kv_heads × head_dim × bytes(dtype)
~~~

单 block：

~~~text
block_bytes = block_size × KV_bytes_per_token
~~~

这是 GQA/MQA 的关键收益：`n_local_kv_heads` 不是 query heads。

### 11.2 MLA

MLA 通常不是完整 K/V heads，而是 latent rank 与 rope head dim：

~~~text
MLA_slot_size = (kv_lora_rank + qk_rope_head_dim) × bytes(cache_dtype)
block_bytes   = block_size × full_attention_layers × MLA_slot_size
~~~

NPU 某些 model/layout 还会按 NZ alignment 向 16 对齐，实际 allocation 可能大于逻辑公式。

### 11.3 Indexer cache

DSA/indexer 可能为 full-attention 层额外存 index cache；INT8 index cache 还需要独立 fp32 scale。容量估算中的 `index_slot_size`、`scale_slot_size` 和 layer mask 见 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:75)。

### 11.4 线性 attention state

KDA/GDN 不按每个历史 token 保存完整 KV，但每个可并发 sequence 需要 conv state 和 SSM/recurrent state：

~~~text
linear_slot_size
  = conv_state_bytes
  + ssm_state_bytes × (speculative_slots + 1)
~~~

再乘 linear layers 和 state block 数。自动容量在 prefix cache 开启时大约把 0.9/(1+0.9) 约 47% 的 cache budget 作为 linear-state 估算上限；这是当前实现的 heuristic，不是所有模型的理论最优比例。见 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:220)。

### 11.5 DeepSeek V4 分组池

DeepSeek V4 使用 SWA/C4/C128 独立 pool，估算器先扣除 constant SWA bytes，再按压缩比换算 token unit。它不能套用“一个 KV block × 层数”的普通公式。见 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:280)。

## 12. `KVCacheCapacity` 是估算结果，不是 tensor

`KVCacheCapacity` 保存：

- `cache_size_in_bytes`
- `n_blocks`、`block_size`
- `slot_size`、`index_slot_size`、`scale_slot_size`
- `linear_slot_size`、`linear_cache_size_in_bytes`
- full/linear layer 数
- linear state blocks
- DeepSeek V4 的 SWA/C4/C128 count

定义见 [kv_cache_capacity.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_capacity.h:20)。它把“预算”与“形状/实现”解耦，便于同一份容量用于 Engine、BlockManager 和 Worker tensor allocation。

## 13. `init_standard_counts` 如何把预算变成 block 数

### 13.1 先分类层

估算器遍历 `n_layers`，根据 `is_full_attention_layer` 把层分成 full 和 linear；indexer cache 如果启用，还会用 layer mask 限制参与层数。

### 13.2 先为 linear state 保留空间

~~~text
linear_cache_bytes
  = num_linear_attention_layers
  × num_linear_state_blocks
  × linear_slot_size

full_cache_budget
  = cache_size_in_bytes - linear_cache_bytes
~~~

若 full budget <= 0，估算直接 CHECK 失败；这就是 `max_linear_state_cache_slots` 过大或 `max_seqs_per_batch` 过大的典型启动错误。

### 13.3 再计算普通 block

~~~text
full_block_bytes
  = block_size × (
      full_layers × (slot_size + scale_slot_size)
      + indexer_layers × index_slot_size
    )

n_blocks = floor(full_cache_budget / full_block_bytes)
~~~

MLU RDMA index scale 需要额外 registered padding，代码用二分搜索找实际可放的最大 block 数；见 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:440)。

## 14. `KVCacheShape` 如何展开 tensor

`KVCacheShape` 依据 model args、capacity 和 world size 构造：

- key cache shape
- value cache shape
- index cache shape/scale
- conv cache shape
- SSM cache shape
- DeepSeek V4 grouped pool shape

构造入口见 [kv_cache_shape.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:45)。`world_size` 会决定 local KV heads，因此 TP 改变 tensor shape，但不一定按 TP 倍数减少所有辅助 buffer。

## 15. Worker 怎样真的申请 KV tensor

### 15.1 创建每层实现

`create_kv_cache_impl()` 按 layer mask 选择：

- `LinearAttentionKVCacheImpl`
- `QuantizedKVCacheImpl`
- `IndexedKVCacheImpl`
- 标准 `KVCacheImpl`

见 [kv_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache.cpp:35)。

### 15.2 普通 allocator

默认 `KVCacheTensorAllocator` 对每个 shape 调 `torch::zeros(... device)`；这是实实在在的设备 tensor allocation，不是只有 block id。见 [kv_cache_tensor_allocator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_tensor_allocator.cpp:20)。

### 15.3 NPU huge page

PD transfer 或指定路径可使用 `aclrtMalloc(... ACL_MEM_MALLOC_HUGE_ONLY)`，再包装为 Torch storage。优点是物理连续/注册行为更稳定，代价是大块申请对碎片更敏感。实现见 [kv_cache_utils.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_utils.cpp:70)。

### 15.4 Sleepable KV

standard K/V 可以放进单个 VMM-backed region，按 layer 为 K/V 建 view；sleep 时 unmap 物理 chunk，虚拟地址仍保留。实现见 [kv_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache.cpp:130)。

### 15.5 XTensor KV

XTensor 不再由每层普通 `torch::zeros` 申请独立大块，而通过 XTensorAllocator 创建映射到 page pool 的 K/V tensor；当前实现明确不支持 linear/indexer cache 组合。见 [kv_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache.cpp:300)。

## 16. BlockManager 不是显存 allocator

### 16.1 它管理逻辑页

`BlockManagerImpl` 初始化 free block id、usage bitmap、prefix cache，并预留 block 0 作为 padding block。见 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:42)。

### 16.2 allocate 只取 id

`allocate(num_blocks)` 从 free list 取 id，增加 `num_used_blocks_`，不会在这一步重新申请 KV tensor。tensor 已由 Worker 在启动阶段建好。

### 16.3 CompositeBlockManager 扩展多种资源

BlockManagerPool 为每个 DP rank 创建 Composite；它可以同时包含 KV、SWA、C4/C128、LINEAR、EMBEDDING 等 leaf。见 [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:22)。

### 16.4 请求增长

新请求通常经过 `allocate_shared_for_sequence`、beam copy-on-write 和 `allocate_sequence`；如果某 leaf 没有足够 block，整个 admission 失败并回滚已分配资源。运行中看到 used block 增长，代表逻辑容量消耗，不代表 driver 又 malloc 了同样大小。

## 17. Prefix Cache 如何让显存“看起来没释放”

开启 prefix cache 后，finished request 的完整 block 可能进入 LRU，而不是立即回 free list；多个 request 可以 alias 同一物理 block。`has_enough_blocks()` 不足时才调用 prefix eviction。见 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:120)。

所以：

~~~text
request finished
  != device KV immediately reusable
  unless block has no prefix-cache / alias owner
~~~

这解释了为什么服务空闲后 `nvidia-smi` 仍显示高显存：KV tensor 本身常驻，prefix cache 只改变 block 的逻辑 owner。

## 18. 请求运行期显存变化的真实模式

### 18.1 第一个请求

第一个请求通常只做：

1. 从 free list 取 prompt 所需 block。
2. 构造 host/device input metadata。
3. 首次 Graph bucket、attention plan 或 workspace lazy allocation。
4. 写入已存在 KV tensor。

因此首请求可能同时出现 block used 增长和 workspace/Graph allocation 峰值。

### 18.2 后续 decode

每生成一个 token，Sequence 只有跨过 block 边界时才取新 block；例如 block_size=128，已有 127 个 token，再生成一个 token 可能从 free list 取第二 block，否则只是写现有页。

### 18.3 结束与回收

无 prefix cache 时，Sequence deallocate 会让 block 回 free list；有 prefix cache 时，block 进入 cache 并等待 LRU eviction。Host offload 还可能把 device block 内容复制到 pinned host block，再释放 HBM alias。

### 18.4 抢占

Scheduler 可能 deallocate/evict 请求的 block，恢复时重新分配并重新计算；显存峰值下降，但计算量和延迟上升。Zero-Eviction 会更保守地拒绝 admission，以换取不重算。

## 19. Host offload：用主存换 HBM

`host_blocks_factor > 1` 时，LLMEngine 创建 `HierarchyBlockManagerPool` 并为 host KV 建 pool；见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:600)。

### 19.1 它增加什么内存

~~~text
host_KV_capacity
  = device_KV_blocks × host_blocks_factor
~~~

实际还包括 pinned allocation、block metadata 和 H2D/D2H transfer queue。

### 19.2 什么时候 offload

Hierarchy manager 只把已经完成、独占（ref_count 合适）的 device block 排入 offload 队列；共享 beam/prefix block 不会盲目搬走。见 [hierarchy_block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/hierarchy_block_manager_pool.cpp:180)。

### 19.3 什么时候回灌

新的请求命中 host prefix cache 时，host blocks 通过 H2D transfer 重新填入设备 block，并将 `kv_cache_tokens_num` 前移；这会增加首 token 延迟和 PCIe/HCCS 带宽压力。

### 19.4 当前限制

DeepSeek V4 的 SWA/C4/C128 grouped cache 当前 host offload 会 `CHECK` 失败，因为 host path 仍只处理 flat KV；不能把 `host_blocks_factor` 写成所有模型通用支持。

## 20. Graph 显存：为什么首请求会突然涨

### 20.1 Persistent tensors

CUDA Graph persistent 参数按 `max_tokens_per_batch`、`max_seqs_per_batch`、`max_position_embeddings/block_size` 建立 token、position、block table、hidden state、paged index 等 tensor。其尺寸直接受 batch 和最大上下文配置影响。

### 20.2 Capture private pool

Graph capture 可能把中间 activation 的内存保存在 private pool；不同 shape/bucket 可能形成多个 pool。`enable_graph_vmm_pool` 通过 VMM 共享物理内存的设计，目标是让多 shape 物理显存更接近 `max(shape)` 而不是 `sum(shape)`，详见 [shared_vmm_allocator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/platform/shared_vmm_allocator.cpp:180)。

### 20.3 Graph double buffer

NPU ACL Graph double buffer 会为两个 slot 保留 persistent param/graph 实例；它改善 overlap，但通常增加静态 buffer。`enable_graph_double_buffer` 默认 true 的定义见 [execution_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/execution_config.h:45)。

### 20.4 no-padding 的权衡

decode no-padding 可以减少小 batch 的无效 token buffer，却增加 shape/graph 数量和 capture 复杂度；prefill 通常仍需要 bucket/padding 才能复用 Graph。

## 21. Attention workspace 与激活峰值

### 21.1 Prefill 比 decode 更危险

Prefill 的 Q/hidden 数量约为本轮 token 数，workspace 通常随 `max_tokens_per_batch` 线性或超线性增长；decode 每 sequence 一 token，主要峰值来自 batch size、MoE、Graph 和通信。

### 21.2 Chunked Prefill 的显存意义

chunked prefill 不减少最终 KV，但限制单拍激活和 attention workspace。长 prompt 被切成多轮后，单拍峰值更低，代价是更多调度轮次、更多 metadata 和可能的 Graph bucket。

### 21.3 `DeviceMonitor` 的 activation 不是所有激活

DeviceMonitor 在模型同步执行后记录 `monitor_buffer` 的最大临时 buffer，计算 `active_activation_memory`。见 [device_monitor.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/device_monitor.cpp:35)。它是 xLLM 的监控估计，不是 allocator snapshot 的完整 active bytes。

## 22. MoE 显存和通信的特殊关系

MoE 主要新增两类显存：

- 本 rank expert weight、redundant expert weight。
- token dispatch/combine、sorted token、router top-k 临时 buffer。

EP 减少单 rank 权重但增加 all-to-all buffer；EPLB 复制热门 expert，可能提升吞吐但直接吃 HBM。`max_tokens_per_batch` 和 `num_experts_per_tok` 会共同影响 dispatch workspace。

因此不能只用“参数量/TP/EP”计算 MoE 显存，还要把一次最大 token batch 的 dispatch buffer 加入峰值账本。

## 23. Speculative Decoding 的双份状态

投机解码可能同时存在 target 与 draft model：

~~~text
resident = target_weights + draft_weights
        + target_KV + draft_KV
        + candidate_embeddings/probs
        + accepted/rejected token metadata
~~~

`KVCacheEstimateOptions` 对 DeepSeek V4 MTP 会把 target/draft constant 和 token unit 合并后重新折算 capacity；见 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:360)。

投机 token 数增大可能提升吞吐，却会提高 linear state stride、embedding block 和验证 activation。不能只看接受率。

## 24. PD 分离的显存账本

Prefill 和 Decode 分开后，每个实例都可能持有：

- 本地权重。
- 本地 KV/linear cache。
- 发送或接收 staging buffer。
- RDMA/HCCS 注册内存。
- transfer metadata、block id 和 route state。

PUSH 模式 transfer buffer 可能在模型计算期间存活；PULL 模式 decode 侧需要临时目标 block。跨节点传输降低单实例压力，却不会消灭 KV 总字节数，还可能产生双份短暂峰值。

## 25. MLA/KDA：Kimi K3 的显存特殊点

### 25.1 MLA 不等于普通 K/V

Kimi K3 Full MLA cache 使用 latent/position 两类 cache，而不是完整 K/V heads。TP2 五层验证中已确认 Full MLA write cache 通过 `torch_npu._npu_reshape_and_cache` 完成，但这只证明五层 prefill 写入链路，不能推出全长 decode 的 HBM 稳定性。

### 25.2 KDA 需要线性 state pool

KDA layer 的 recurrent state 和 conv state 由 LINEAR cache 叶子管理；它们按 slot/block 预留，不随历史 token 线性增长，但随并发 sequence、speculative stride 和 layer 数增长。

### 25.3 Prefix Cache 的额外成本

Kimi K3 如果启用 linear-state prefix cache，Engine 要求 chunked prefill 和 chunk size 对齐 block；否则不能安全保存/命中 recurrent checkpoint。这个约束会影响调度配置，也会把一部分 KV budget 转成 linear-state budget。

## 26. 通信 buffer 的拓扑分析

### 26.1 TP

TP 的 activation collective buffer 通常按本轮 token × hidden/intermediate dimension × dtype；prefill token 越多，通信 buffer 峰值越明显。

### 26.2 EP

EP dispatch/combine buffer 可能按 token × top-k × hidden，并且 all-to-all 的 send/recv 可能同时存在，峰值不是单份 tensor。

### 26.3 CP

CP 对 attention context 和 metadata 做分片/交换；KV cache 可能按 local context 变小，但 collective/通信 staging 变大。

### 26.4 DP

DP 不一定复制通信 buffer到所有请求同样大小，但每个 DP rank 独立拥有权重和 KV block pool；容量最终取最小 rank。

### 26.5 PD

PD 增加跨实例 KV block 传输 buffer；PUSH/PULL、RDMA 注册和 host pinned 方式会改变它位于 HBM、主存还是设备 BAR 映射。

## 27. Overlap 对显存的影响

异步调度把 host scheduling 与设备计算重叠，但会延长 in-flight Request、input tensor、handoff output 和 KV block 的生命周期。`last_batch_` 尚未提交时，KV 不能提前 deallocate；Worker `retained_input` 也必须活到 event 完成。详见 [09-async-schedule-and-overlap-complete-chain.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/09-async-schedule-and-overlap-complete-chain.md:228)。

所以 overlap 的显存成本通常不是“再分配完整一份 KV”，而是：

- 一拍延迟的 block ownership。
- cloned host views。
- retained input/output。
- Graph double buffer 的另一槽。
- 更晚的 prefix eviction/release。

## 28. Zero-Eviction 与显存利用率

Zero-Eviction 不是增加显存，而是把 admission 条件从“可以抢占并重算”改为“必须有足够空闲 block”。结果是：

- used block 上限更保守。
- 运行中不发生 eviction/recompute。
- 空闲显存可能看起来更多。
- 吞吐/并发可能下降，但尾延迟和重复计算更稳定。

因此“显存利用率越高越好”并不总成立；服务容量要同时看重算率和 P99。

## 29. `block_size` 的双向影响

### 29.1 较大 block

优点：block table 更小、prefix hash/metadata 次数少、kernel 页表开销低。

代价：短请求尾部浪费更大，prefix 命中粒度粗，beam/fork 的 copy-on-write 更重。

### 29.2 较小 block

优点：显存按 token 更细粒度回收，短请求和 prefix cache 更灵活。

代价：block id、block table、hash、free-list 操作增多；某些 kernel/Graph 需要对齐，可能产生额外 padding。

### 29.3 Kimi K3 额外约束

线性 state prefix cache 要求 `max_tokens_per_chunk_for_prefill % block_size == 0`；因此不能只按普通 Attention 的尾部浪费选择 block size。

## 30. Prefix Cache 的显存收益不能按命中 token 直接相减

命中 prefix cache 省的是重复计算和新建 block 的逻辑 owner，但共享 prefix block 仍占用 HBM；cache LRU 还会延迟释放。真正的容量收益来自：

- 多请求共享同一物理 block。
- 减少新 suffix block 的增长速度。
- 提高 prefix hit 时的 admission 成功率。

如果请求前缀只命中很少 block，hash/LRU metadata 和常驻 block 可能抵消收益。

## 31. OOM 的四种不同类型

### 31.1 启动权重 OOM

模型构造或 load_model 时没有足够显存，KV 还没开始估算。降低 KV 配置无效，应换卡、量化、TP/EP、rolling 或减少模型。

### 31.2 KV capacity 为零

权重加载后 available 被 utilization reserve、max_cache_size 或 linear-state reservation 吃完，`estimate_kv_cache_capacity` CHECK 失败。应减小 `max_seqs_per_batch`、linear slots 或模型常驻内存。

### 31.3 运行时 workspace OOM

启动 KV 成功，但大 prefill、MoE、Graph capture 或 speculative verify 需要额外临时 buffer。应降低 batch/chunk、关闭高峰 feature 或增加 reserve。

### 31.4 碎片/连续大块 OOM

总 free 仍有空间，但无法满足 huge page、Graph pool、VMM chunk 或大 KV tensor 的对齐/连续性要求。应检查 allocator snapshot、历史 graph shape 和其他进程，而不是只看一行 free。

## 32. 运行时为什么会“逐渐吃满显存”

常见增长序列是：

~~~text
启动后：权重 + 初始 KV tensor
首个请求：attention/MoE workspace
首个 decode bucket：Graph persistent/private pool
并发上升：KV block used、prefix cache alias
长请求：block 跨界增长、linear state slot
复杂特性：spec/PD/host transfer staging
allocator：reserved segments 留在缓存
~~~

其中只有 KV block used 会随着 token 逻辑增长；workspace/Graph 可能在第一次遇到最大 shape 后稳定；allocator reserved 可能在请求结束后仍不下降。

## 33. 一个可计算的 80 GiB 示例

假设：

- 设备总显存 80 GiB。
- 权重和 runtime 初始化后驱动 free 约 42 GiB。
- `max_memory_utilization=0.8`，保留 `80×0.2=16 GiB`。
- `max_cache_size=24 GiB`。
- 普通 Attention 32 层、每 rank 8 个 KV heads、head_dim=128、BF16。
- `block_size=128`。

### 33.1 KV 单 token

~~~text
2(K,V) × 32 × 8 × 128 × 2 bytes
= 131072 bytes
= 128 KiB/token
~~~

### 33.2 单 block

~~~text
128 tokens × 128 KiB
= 16 MiB/block
~~~

### 33.3 可用 cache budget

~~~text
min(42 - 16, 24) GiB
= 24 GiB
~~~

### 33.4 逻辑 capacity

忽略 alignment、workspace、linear/indexer 后：

~~~text
24 GiB / 16 MiB ≈ 1536 blocks
1536 × 128 ≈ 196608 token slots
~~~

这不是服务保证值：如果模型是 MLA、KDA、MoE、Graph 或 PD，实际 block bytes 和额外峰值都会改变；如果某个 TP rank 只有 38 GiB free，最终 capacity 还会按该 rank 重新下降。

## 34. 另一个 Kimi K3 线性 state 例子

假设 5 层影子配置包含 4 个 KDA 层、1 个 MLA 层，cache budget 固定为 4 GiB，自动 linear state 使用约 47% heuristic。则不是简单地把 4 GiB 全除以 MLA block bytes：

~~~text
linear budget ≈ 4 GiB × 0.9 / 1.9 ≈ 1.895 GiB
full MLA budget ≈ 2.105 GiB
~~~

实际还要乘每个 KDA 层的 conv/SSM slot size、padding 和 speculative stride。这个例子只说明预算分治，不能替代从 Kimi K3 ModelArgs 计算出的精确容量。

## 35. 监控：服务中到底看哪些指标

### 35.1 xLLM 自己的账本

DeviceMonitor 记录：

- total memory
- weight memory
- total KV cache memory
- total activation memory
- active activation memory

设置与更新见 [device_monitor.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/device_monitor.cpp:20)。

### 35.2 Scheduler/BlockManager

应该观察：

- `num_free_blocks`
- `num_used_blocks`
- KV cache utilization
- prefix cache blocks
- preemption/recompute count
- host offload queue

这些数字解释“逻辑容量用尽”而不是物理 allocator 是否 OOM。

### 35.3 Driver/allocator

CUDA 可查看 `allocated_bytes`、`reserved_bytes`、Graph private pool snapshot；NPU/MLU 需使用相应 caching allocator 和设备工具。必须把时间点固定在：模型加载后、KV 分配后、首请求后、最大 prefill 后、并发稳定后。

### 35.4 推荐日志快照

每次实验保存：

~~~text
启动配置 dump
每 rank total/free/weight/KV capacity
KV shape 和 slot bytes
Graph bucket/private pool
block free/used/prefix count
host offload count
TTFT/TPOT/P99/OOM 时间点
~~~

## 36. 显存调优的正确顺序

1. 先确认旧 worker/其它进程没有占卡。
2. 固定模型、dtype、TP/DP/EP 和输入，记录启动后 weight memory。
3. 调 `max_memory_utilization` 留出 workspace，而不是直接设 1.0。
4. 用 `max_cache_size` 给 KV 设置可解释的硬上限。
5. 再调整 `block_size`、`max_seqs_per_batch`、`max_tokens_per_batch`。
6. 对混合模型检查 `max_linear_state_cache_slots` 和 prefix/chunk 对齐。
7. 需要容量时考虑 host offload、rolling、量化或 TP/EP，而不是只加 utilization。
8. 最后叠加 Graph、spec、overlap、PD，并重新测峰值。

## 37. 配置影响矩阵

| 配置 | 常驻显存 | 峰值显存 | 逻辑 KV 容量 | 典型风险 |
| --- | --- | --- | --- | --- |
| `max_memory_utilization` ↑ | 不直接改变权重 | reserve 变小，峰值风险 ↑ | KV budget ↑ | workspace OOM |
| `max_cache_size` ↓ | KV 上限 ↓ | 峰值更可控 | block/token ↓ | admission/抢占 ↑ |
| `block_size` ↑ | tensor 总量近似不变 | metadata ↓ | 尾部浪费 ↑ | prefix 粒度粗 |
| `max_tokens_per_batch` ↑ | Graph buffer ↑ | activation/workspace ↑ | 调度吞吐可能 ↑ | prefill OOM |
| `max_seqs_per_batch` ↑ | Graph metadata ↑、linear slots 可能 ↑ | decode/beam ↑ | 并发 ↑ | capacity 估算失败 |
| `kv_cache_dtype=int8` | KV ↓ | scale/workspace变化 | token capacity ↑ | 当前后端限制/精度 |
| `enable_prefix_cache` | 常驻 cache blocks ↑ | eviction/alias变化 | 共享后有效容量 ↑ | cache 不立即释放 |
| `host_blocks_factor` ↑ | HBM 可下降 | staging/pinned ↑ | host capacity ↑ | 传输尾延迟 |
| `enable_graph` | persistent/private pool ↑ | capture 峰值 ↑ | 不直接改变 KV | shape/bucket 累积 |
| `enable_xtensor` | page metadata/映射 ↑ | page 对齐 ↑ | 与权重共享池 | feature 组合限制 |
| `num_speculative_tokens` ↑ | draft/linear state ↑ | verify activation ↑ | 接受率可能 ↑ | 双模型 OOM |

## 38. 精度与显存布局风险

显存优化不只影响容量，也可能影响正确性：

- KV dtype/量化：低精度表示，可能有数值误差。
- MLA layout/NZ alignment：shape/stride 错误会导致 attention 读错。
- block table/page mapping：逻辑 token 到物理 slot 错位会产生逻辑错误。
- prefix alias/refcount：提前复用或释放会污染请求。
- host offload：H2D 未完成就读会产生状态错误。
- Graph/VMM：虚拟地址复用错误会让 replay 访问旧 buffer。
- linear state slot：KDA restore 错位会导致后续 token 全部漂移。

因此“减少显存”必须分成布局、低精度、生命周期和逻辑映射四类风险分别验证。

## 39. 通信与显存的重叠关系

通信 overlap 常常需要双份或多份 buffer：

~~~text
current_compute_buffer
next_send_buffer
recv_or_reduce_buffer
event / staging metadata
~~~

当 overlap 真实发生时，峰值可能增加；当通信没有足够独立计算可隐藏时，新增 buffer 只增加显存，不提升性能。应在 trace 中确认 stream/event 和实际 concurrent kernel，而不是看到 async API 就宣称成功。

## 40. 组合特性状态矩阵

| 组合 | 显存变化 | 当前源码判断 |
| --- | --- | --- |
| Paged KV + Prefix Cache | tensor 常驻、block alias/LRU | `已支持`，请求级容量仍需压测 |
| Chunked Prefill + Linear state | 单拍 activation 降，state pool 增 | `条件支持`，需 block 对齐 |
| Async overlap + Graph | retained/in-flight + 双槽 | `条件支持`，backend/shape 相关 |
| Speculative + MTP | target/draft KV、embedding/state 双份 | `条件支持`，Kimi K3 未验证 |
| PD + host offload | transfer/staging + host KV | `条件支持`，模型/设备相关 |
| XTensor + linear attention | page pool 与 linear state | `已禁用/不支持组合`，代码有 CHECK |
| Sleep mode + quant/indexer/linear | 需要复杂 state map | `已禁用`，当前实现拒绝 |
| DeepSeek V4 + flat host offload | grouped pool 无 flat KV | `已禁用`，明确 CHECK |

## 41. 常见误区

### 41.1 “`max_memory_utilization=0.8` 就只用 80% 显存”

它是 KV capacity 规划中的 reserve 逻辑，实际还受权重、allocator cache、workspace、Graph 和其它进程影响。

### 41.2 “KV cache 是请求来了才 malloc”

默认路径在启动后已按 capacity 申请整组 KV tensor；请求只取 block id。Graph/workspace 才可能首次请求 lazy allocation。

### 41.3 “used blocks=0 就代表 KV tensor 释放了”

错误。block id 空闲不等于底层 K/V tensor销毁；tensor 通常服务整个进程生命周期。

### 41.4 “prefix cache 命中就是免费显存”

错误。共享 block 仍常驻 HBM，LRU 还会延迟回收；它节省的是重复计算和新 block 增长。

### 41.5 “nvidia-smi 的显存就是模型有效显存”

错误。它通常包含 allocator reserved、Graph pool、通信和其它进程，不能直接等于权重+KV。

### 41.6 “增大 max_tokens_per_batch 一定提高吞吐”

它可能提高设备利用率，也会放大 activation、Graph persistent、MoE dispatch 和通信 buffer；最终可能降低并发或触发 OOM。

### 41.7 “host_blocks_factor 越大越好”

主存容量增加会带来 H2D/D2H 带宽和尾延迟成本；并且当前对 grouped cache 有明确限制。

## 42. OOM 排查决策树

~~~text
启动就 OOM？
  |-- 模型构造/load_model 期间 -> 权重/量化/TP/rolling
  |-- KV estimate 期间 -> utilization/max_cache/linear slots/max_seqs
  |-- KV allocate 期间 -> shape、huge page、碎片、rank 最小值
  |
服务首请求后 OOM？
  |-- Graph capture/private pool -> 降 max_tokens/shape、VMM pool
  |-- Prefill/MoE workspace -> chunk/batch/top-k/通信 buffer
  |-- Spec verify -> draft/target、spec tokens
  |
运行一段时间后 OOM？
  |-- prefix cache/LRU -> eviction/refcount
  |-- host/PD transfer -> staging/pinned queue
  |-- overlap/Graph retained -> drain/event/lifecycle
  |-- 其它进程/碎片 -> allocator snapshot/设备监控
~~~

## 43. 建议的分层实验

### E0：配置和容量静态检查

保存启动 config dump、ModelArgs、TP/DP、total/free、估算的 slot/block bytes；不宣称性能。

### E1：Worker 路径可达

确认 model load、KV shape、KV allocation、BlockManager pool 创建完成；检查日志中的 `kv cache capacity`。

### E2：基础分配释放正确

运行 block manager/KV unit tests，验证 free/used、padding block、prefix refcount、deallocate 回滚。

### E3：请求 smoke

短 prompt + `max_tokens=1/2`，比较请求前后 block used、显存快照和服务健康。

### E4：数值/容量组合

固定输入比较 sync/feature 模式 token/logits；测长 prompt、跨 block、prefix hit、chunked、Graph 和 Kimi MLA/KDA。

### E5：稳定性与性能

长时间并发、P99、allocator reserved、Graph bucket growth、OOM/recovery、host offload 和 PD transfer；当前 Kimi K3 显存稳定性尚未达到 E5。

## 44. 读源码的推荐顺序

1. 读 [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:43) 和 [options.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/options.h:35)，确认参数和默认值。
2. 读 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:135)，确认 model -> estimate -> allocate 顺序。
3. 读 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1371)，确认权重和 KV 的 Worker 生命周期。
4. 读 [kv_cache_estimation.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_estimation.cpp:604)，掌握容量公式。
5. 读 [kv_cache_shape.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp:45) 和 [kv_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/kv_cache.cpp:265)，把容量变成 tensor。
6. 读 [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:90) 和 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:85)，理解请求只取逻辑页。
7. 最后读 Graph、XTensor、host offload、spec/PD 和 DeviceMonitor，解释特殊峰值与边界。

## 45. 读完后应能回答的问题

1. xLLM 为什么要先 load model 再 estimate KV cache？
2. `max_memory_utilization` 具体扣的是什么，为什么不是简单的 `total×0.8`？
3. 为什么不同 TP rank 最终使用同一个最小 KV capacity？
4. `KVCacheCapacity`、`KVCacheShape`、`KVCache` 和 `BlockManager` 分别拥有哪一层职责？
5. 为什么请求增长通常不会再次 malloc 整块 KV tensor？
6. MLA、GQA、KDA 的每 token 显存公式有什么不同？
7. `max_linear_state_cache_slots` 为什么可能让普通 KV block 变少？
8. prefix cache 为什么会让显存保持高位却提高有效容量？
9. Graph 首次 capture 为什么会导致显存阶跃？
10. XTensor 如何让权重和 KV 竞争同一物理页池？
11. 哪些 OOM 属于权重，哪些属于 KV，哪些属于 workspace/Graph？
12. Kimi K3 当前显存证据证明到哪一级，哪些长上下文组合仍未验证？

## 46. 当前支持状态

| 主题 | 状态标签 | 当前证据 |
| --- | --- | --- |
| 普通权重加载与 KV capacity 估算 | `已支持` | 当前 LLMEngine/Worker 主链可达 |
| Paged block pool 与 prefix refcount | `已支持` | BlockManager/PrefixCache 源码和既有测试代码 |
| MLA latent/position cache | `条件支持` | Kimi K3 五层 TP2 prefill 已通过写 cache，长服务未完成 |
| KDA linear state pool | `条件支持` | capacity/shape/restore 代码可达，Kimi K3 长压测未验证 |
| Graph/VMM memory reuse | `条件支持` | CUDA/ACL/XTensor 源码路径，目标设备 trace 未齐 |
| host KV offload | `条件支持` | flat KV host pool 可达，grouped cache 有限制 |
| KV INT8 | `条件支持` | 当前只允许 MLU，VLM/PD 明确拒绝 |
| XTensor weight/KV physical pool | `条件支持` | 代码可达，Kimi K3 完整服务未验证 |
| Kimi K3 全层显存稳定性 | `代码可达、未验证` | 当前仅五层/TP2 prefill 和短 smoke |

## 47. 最终总结

xLLM 的显存管理可以按三个问题来读：

1. **常驻什么？** 权重、KV/linear state、Graph、通信和 allocator pool。
2. **容量如何定？** Worker 测量 free/total，Engine 应用 utilization/max cache，再按模型结构算 block/slot。
3. **运行中怎样复用？** Scheduler 通过 BlockManager 取逻辑页，Prefix Cache/host offload/抢占改变 owner，Graph/workspace/通信决定峰值。

“显存用满”不是一个单一事件，而是：权重先占住一部分，KV tensor 按预算预分配，逻辑 block 随请求增长，Graph/workspace 在 shape 到达时懒分配，allocator 还可能保留已经释放的 segment。只有把这几层拆开，才能知道该调 `max_memory_utilization`、`max_cache_size`、`block_size`、batch/chunk、linear slots、host offload、TP/EP、Graph 还是模型 dtype。

## 48. 证据边界

本文确认的是当前源码中的预算和生命周期逻辑，不代表任意模型、任意设备和任意组合已经通过稳定性性能验收。特别是：

- 当前 Kimi K3 五层 TP2 prefill 的 cache 写入正确性，不等于全层长上下文 HBM 稳定性。
- 当前代码可构造 Graph/VMM，不等于目标设备所有 shape 都命中且显存收敛到理论值。
- 当前 host offload flat KV 可达，不等于 DeepSeek V4 grouped cache 可用。
- `free/allocated/reserved` 的一次快照不能替代长时间压力测试。

后续显存专题整改应沿着本文的 E0-E5 阶梯，把容量日志、allocator snapshot、block usage、Graph bucket 和 Kimi K3 请求级稳定性逐项补齐。

## 49. Allocator 生命周期：tensor 释放后为什么显存不降

### 49.1 驱动分配和框架缓存是两层

模型调用 `torch::empty/zeros` 时，通常先向 PyTorch/设备 caching allocator 请求；allocator 没有合适空闲块时才向设备驱动申请 segment。tensor 析构后，块可能回到 allocator free list，而不是直接归还驱动。

~~~text
device driver segment
  +-- active tensor block
  +-- inactive reusable block
  +-- alignment / split remainder
~~~

所以 device tool 看到的是进程持有的 segment，allocator snapshot 才能区分 active 和 inactive。

### 49.2 xLLM 容量探测会主动清 cache

Worker 估算 KV 前调用 `Device::empty_cache()`；NPU 路径还读取 caching allocator 的 cache size，再把这部分视为可用于 KV 的 available。`empty_cache` 的各 backend 实现见 [device.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/platform/device.cpp:130)。

这一步尽量让“权重加载后的剩余显存”更接近真实值，但不能清理仍被 model、communication runtime 或 Graph 引用的 allocation。

### 49.3 为什么运行期不频繁 empty cache

频繁归还驱动会增加重新 malloc、同步和碎片整理开销；服务更倾向保留 allocator pool 以快速复用。因此空闲期显存保持高位通常是性能设计，不是自动等同于泄漏。

### 49.4 如何判定泄漏

至少同时满足以下信号才值得怀疑：

- 相同 shape、相同并发多轮后 active/reserved 持续单调增长。
- Graph bucket、prefix block、response queue 数已稳定。
- 请求结束后 logical used blocks 回到预期，但 active tensor 数仍增长。
- allocator snapshot 显示新 segment 的 owner/stack 持续增加。

## 50. VLM Encoder Cache：另一套按字节 LRU

### 50.1 它和 KV Prefix Cache 不同

VLM encoder cache 保存图片/视频 encoder 产生的 embedding，而 KV Prefix Cache 保存 decoder 历史状态。前者按 embedding tensor bytes 做 LRU，后者按 token block/refcount 管理。

### 50.2 配置单位和创建点

`max_encoder_cache_size` 的公共配置单位是 MB，默认 0 表示禁用。`VLMExecutorImpl` 将其乘 `1024×1024` 后创建 `EncoderCache`；见 [vlm_executor_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/vlm_executor_impl.cpp:38)。

### 50.3 VLM KV 预算会主动预留 encoder cache

VLM Engine 在估算 KV available 时扣除 `max_encoder_cache_size × MiB`，避免 KV 先把 encoder cache 的目标空间吃掉。见 [vlm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/vlm_engine.cpp:214)。

### 50.4 LRU 行为

EncoderCache 插入 embedding 前计算 tensor 大小并 `evict_until_fit`；大于 cache 总上限的单项不会无条件撑爆预算。实现见 [encoder_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/encoder_cache/encoder_cache.cpp:24)。

### 50.5 多模态峰值

即使 encoder cache 有上限，一次请求仍可能短时同时持有：原始 pixel tensor、vision encoder activation、projector output、cache insertion tensor 和 decoder input embedding。`limit_image_per_prompt` 限制输入数量，但不能替代实际图像分辨率/patch 数的峰值测量。

## 51. FlashInfer/XAttention Workspace 是显式常驻成本

### 51.1 FlashInfer float workspace

`flashinfer_workspace_buffer_size` 默认 128 MiB。初始化会创建 device float workspace、device int workspace 和 page-locked host int workspace；见 [flashinfer_workspace.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/flashinfer_workspace.cpp:20)。

### 51.2 Page-locked host 不是 HBM

page-locked int workspace 占用主机 pinned memory，而对应 device int/float workspace 占 HBM。排查系统内存不足时也要记录它，尤其是每 worker/每进程都构造 workspace 的部署。

### 51.3 XAttention 的额外 workspace

XAttention 可能维护 shared/unshared 两套 int/page-locked workspace；执行器根据 plan 选择 FlashInfer 或 XAttention buffer。见 [xattention_workspace.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/xattention_workspace.cpp:20) 与 [xattention_planinfo.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/cuda/xattention_planinfo.cpp:86)。

### 51.4 调小的风险

workspace 过小可能导致 plan/kernel 不可用、fallback 或运行时错误；它不是单纯可回收的“浪费”。应根据最大 batch/context 的 planner 要求验证，不应盲目设成 0。

## 52. Beam Search 的显存放大

### 52.1 Sequence 数放大

beam width 为 B 时，一个请求会展开为多个 Sequence；Graph persistent seq capacity 在推荐模式下还可能乘 beam width。KV shared prefix 起初可以 alias，但每个 beam 继续生成后会分叉。

### 52.2 Copy-on-write

如果多个 beam 共享最后一个未满 block，其中一个继续写时必须申请新 block并复制旧内容。BlockManagerPool 在 `need_swap` 时分配一个新的 KV block并记录 transfer info；见 [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:265)。

### 52.3 峰值公式

粗略上界：

~~~text
beam_KV_tokens
  ≈ shared_prefix_tokens
   + B × divergent_suffix_tokens
   + copy_on_write_tail_blocks
~~~

尾部 block 对齐会让短 divergent suffix 的浪费更明显，因此 beam workload 的 block_size 选择不同于单样本 greedy。

## 53. DP、多实例和同卡部署

### 53.1 DP pool 的语义

BlockManagerPool 为每个 DP rank 建独立 Composite manager；Scheduler 可把新 Sequence 路由到 free blocks 最多的 rank。见 [block_manager_pool.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_pool.cpp:100)。

### 53.2 每个 DP 副本的权重

标准 DP 副本通常复制模型权重；若同一物理卡运行多个进程/副本，驱动 free 会被所有进程共同消耗。xLLM 的每个 Engine 只能看到自己的 allocator 和设备总 free，无法替其它进程保证预算。

### 53.3 容量取最弱 rank

某 rank 因额外 expert、Graph bucket或遗留 worker 少 4 GiB，整组 KV capacity 会按它下降。这是集群稳定性的正确选择，但也意味着“平均空闲显存”对容量规划没有意义。

### 53.4 Kimi K3 已出现的实际教训

项目记录曾因目标卡已有约 50 GiB 占用，逻辑 rank 在 MoE 构造阶段仅剩约 1 GiB，而后续还需 2.30 GiB 直接 OOM；切换到空闲 NPU 并使用 0.7 utilization、4 GiB cache cap 后服务才稳定启动。这证明预检其它进程比微调 block size更优先。

## 54. Shutdown、Sleep 和重启的释放语义

### 54.1 正常进程退出

进程结束后 allocator、通信 context、权重、KV 和 Graph pool由操作系统/驱动统一回收。若 master 退出但 worker 成为孤儿，worker 仍持有 HBM 和 HCCL 端口；HTTP 端口不可用不等于设备已释放。

### 54.2 RL deep sleep

`WorkerImpl::rl_sleep()` 调 SleepableAllocator unmap 权重与 KV 的物理 HBM，wake_up 再 map；见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1157)。

### 54.3 XTensor sleep

XTensor 通过 PageAllocator 对 model pages 做 sleep/wakeup；轻睡/深睡和是否保留 weight pages 的语义不同。服务编排必须等待所有 rank完成状态切换，不能单 rank释放。

### 54.4 重启前检查

至少核对进程树、设备显存、通信端口和 worker 日志。直接重复执行启动脚本容易同时存在两组 worker，导致下一次容量估算和 process group 都失真。

## 55. 碎片来源分解

| 来源 | 粒度 | 表现 | 处理方向 |
| --- | --- | --- | --- |
| KV block 尾部 | token/block | 短序列最后一页未满 | 调 block_size/混批 |
| allocator split | bytes/segment | reserved >> active | 稳定 shape、allocator 配置 |
| Graph pool | bucket/private pool | 新 shape 阶跃增长 | bucket/VMM/no-padding策略 |
| VMM page | 2 MiB/运行时粒度 | 小 allocation 尾页浪费 | 调页粒度，谨慎 |
| huge page/连续块 | 大物理块 | free 足够却申请失败 | 预留、早分配、减少碎片 |
| RDMA registration | 注册页/对齐 | logical < registered | 使用 capacity 实际日志 |
| MoE padding | expert/token block | dispatch token 补齐 | token batch/top-k/kernel |

“碎片率”必须说明分母：逻辑 token capacity、allocator reserved，还是物理设备总显存。三个口径不可互换。

## 56. 容量规划工作表

部署前可以按下面顺序填表：

~~~text
1. device_total                       = ____ GiB
2. other_processes_and_runtime        = ____ GiB
3. local_weight_resident              = ____ GiB
4. communication_and_fixed_workspace  = ____ GiB
5. graph_persistent/private_pool       = ____ GiB
6. safety_reserve                     = ____ GiB
7. requested_KV_budget                = ____ GiB
8. KV_bytes_per_token                 = ____ KiB
9. linear/indexer/fixed_pool           = ____ GiB
10. usable_token_blocks               = ____
11. target concurrent tokens          = ____
12. observed P99 peak                 = ____ GiB
~~~

第 7 项应由 `max_cache_size` 明确控制；第 12 项来自实际 workload，不应由理论式臆测。

## 57. 文章验收用的三组对照实验

### 57.1 Cache cap 对照

固定所有配置，只把 `max_cache_size` 从 4 GiB 改成 8 GiB。预期：权重不变，KV capacity/block 近似翻倍，首请求 workspace不应随之翻倍；如果总显存增长远超 4 GiB，要检查 Graph或 allocator。

### 57.2 Batch peak 对照

固定 cache cap，逐步提高 `max_tokens_per_batch` 或实际 prefill chunk。预期：KV tensor常驻不变，activation/workspace/communication峰值增加，Graph persistent可能因最大 shape增加。

### 57.3 Prefix lifecycle 对照

固定相同 prompt，依次关闭/打开 prefix cache。预期：底层 KV tensor总量不变；打开后 finished request 的 logical cache blocks保留、第二请求命中增加、需要新 block 时 LRU eviction发生。

这三组实验分别隔离“预分配容量”“运行时峰值”“逻辑 block owner”，可以避免把三个问题混成一个显存数字。

## 58. 最后再压缩成一张图

~~~text
                  配置与模型结构
                        |
                        v
权重加载 --> Worker free/total --> KVCacheCapacity
   |                               |
   |                               v
   |                         KVCacheShape/tensors
   |                               |
   +--------- 常驻 HBM ------------+
                                   |
请求 --> Scheduler --> block ids --+--> KV write
  |          |                           |
  |          +--> prefix/evict/offload --+
  |
  +--> batch shape --> activation/workspace/communication
                         |
                         +--> Graph bucket/private pool

结束：block owner 释放或转入 prefix cache
空闲：allocator segment 可能继续 reserved
退出/深睡：进程或 VMM 才归还物理 HBM
~~~

这就是判断任何显存问题的顺序：先问属于常驻、逻辑容量、运行时峰值还是 allocator 保留，再去找对应配置和源码，而不是从一张设备监控截图反推所有原因。
