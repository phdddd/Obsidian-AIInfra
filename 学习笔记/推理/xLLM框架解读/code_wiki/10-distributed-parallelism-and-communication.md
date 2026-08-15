# xLLM 分布式并行全解：DP、TP、CP、KV Split 与通信组

## 1. 先分清“复制什么、切分什么”

- DP：复制模型，把不同请求分给不同副本。
- TP：切一层的权重/hidden/head，同一 token 在多卡协作。
- CP：切上下文 token 维度，同一 Attention 的序列分布在多卡。
- KV Split：进一步描述 KV 在并行 rank 间的切分度。
- EP：切 MoE experts，单列在 [11](11-moe-expert-parallel-complete-chain.md)。

它们可以组成多维 rank 网格，而不是互斥选项。

## 2. 当前源码入口

| 层次 | 权威入口 | 这一层回答什么 |
|---|---|---|
| 公共配置 | [parallel_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.h:59) | DP/EP/CP/KV Split/DiT TP 的默认值 |
| flag 合并 | [parallel_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.cpp:23) | CLI/JSON 怎样写入单例配置 |
| 通信组构造 | [collective_communicator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/collective_communicator.cpp:150) | world rank 怎样分解成 DP/TP/EP/CP 域 |
| NPU 域映射 | [mapping_npu.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/mapping_npu.cpp:170) | Attention、MLP、MoE、LM head 使用哪个 group |
| 模型参数 | [parallel_args.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_args.h:38) | rank、group 和 kv split 怎样交给 layer |
| 同步 collective | [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:124) | gather/reduce/reduce-scatter/all-to-all 的公共封装 |
| 异步 collective | [parallel_state_async.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state_async.cpp:22) | launch 与 finish 如何拆开 |
| FlashComm 上下文 | [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:39) | 长 Prefill 是否进入 sequence shard |
| MMRS 消费者 | [linear.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/common/linear.cpp:1537) | RowParallel Linear 怎样融合 Matmul+ReduceScatter |
| Multi-Stream guard | [master.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/master.cpp:356) | 当前为何必须标成已禁用 |
| Rolling 配置 | [load_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/load_config.h:53) | 常驻层数和滚动 slot |
| Rolling 生命周期 | [rolling_load_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.cpp:29) | H2D/compute event 和 slot 复用 |

NPU ATB 和 Torch backend 的 communicator 所有权不同：ATB 可拥有 TP/DP/EP，CP 仍可能单独建 HCCL process group。读代码时不能假定所有 collective 都经同一封装。

## 3. Rank 布局

源码中的 CP 组使用如下布局：

```text
global_rank = dp_rank * (cp_size * attn_tp_size)
            + cp_rank * attn_tp_size
            + tp_rank

attn_tp_size = world_size / (dp_size * cp_size)
```

固定 `dp_rank` 和 `tp_rank`，枚举 `cp_rank` 得到 CP group。任何 rank 公式错误都会造成 group 成员不一致，最终不是结果略差，而是 collective hang。

## 4. TP 的两类 Linear

列并行把输出维切开：每 rank 计算一部分 QKV 或 MLP 中间维，后续若天然消费分片可不立即通信。

行并行把输入维切开：各 rank 得到部分和，通常需要 all-reduce 恢复完整 hidden：

```text
Y = Σ_rank X_rank W_rank
```

Attention 的 head 切分与 MLP 中间维切分能显著减少每卡权重，但每层往往引入 collective。Decode 消息小，延迟敏感；Prefill 消息大，带宽敏感。

## 5. DP 不等于零通信

纯推理 DP 副本计算互不依赖，但服务层仍需：请求路由、负载反馈、结果汇聚。若一个 DP 组内部又有 TP，实际拓扑是多个 TP replica。

DP 的主要内存代价是每个副本拥有完整模型 shard 和独立 KV pool；优点是不同请求不在 token 级互相同步，扩吞吐通常比继续增大 TP 更线性。

## 6. CP 与 KV Split

CP 沿 sequence/context 维切分长 Prefill 或长上下文 Attention。每 rank 保存/处理部分 token，但 Attention 需要获得全局语义，通常伴随 all-gather、all-to-all 或特定 ring/kernel 协议。

`kv_split_size` 允许 KV 分片度与外层 CP 映射显式关联。源码对 world、DP、CP 的整除关系做检查；默认值还可能回退到 CP size。它不是随便填的性能旋钮。

## 7. Collective 实现细节

`parallel_state.cpp` 的 `reduce_scatter` 当前沿 dim 0 工作。当 token 数不能整除 group size 时先补零，再按 rank 切片并去掉尾部 padding。

这意味着性能账本必须加入：

```text
padded_tokens = ceil(tokens / world_size) * world_size
padding_overhead = padded_tokens - tokens
```

`all_to_all_4D` 用于在不同布局间变换四维 tensor，常见于 sequence/head 维交换。异步 API 返回 completion closure；真正 overlap 取决于 closure 何时被调用。

## 8. 通信量直觉

Ring all-reduce 每 rank 近似传输：

```text
2 * (P - 1) / P * message_bytes
```

reduce-scatter 或 all-gather 各约一半阶段。实际延迟还包括 hop、协议、拓扑和小包启动成本。

EP all-to-all 与 token 路由分布相关；CP 通信与上下文大小相关；TP 通信与 hidden/token batch 相关。三者不能用一个“带宽利用率”指标概括。

## 9. Overlap

源码有 launch/finish 异步 collective、multi-stream 配置和 SP overlap 路径，但“API async”只代表 host 不阻塞。要隐藏通信，后续必须存在不依赖 collective 输出的计算。

典型机会：

- 将大 tensor reduce-scatter 后，每 rank 继续计算自己的 sequence shard。
- MoE dispatch 后并行处理本地已到达 token。
- 下一 micro-batch 计算与上一 micro-batch collective 重叠。

典型失败：立刻调用 finish、默认流隐式同步、通信/计算争用相同设备资源、Graph 捕获改变 stream 关系。

## 10. 内存影响

TP 降每卡权重，通常也按 KV head 降每卡 KV；但需要通信 buffer 和有时的 gathered hidden。DP 不降单副本权重，增加全局 KV 容量。CP 降每卡 context KV/激活，但引入重排和 gather buffer。

并行方案的峰值显存不是各 shard 大小简单相加：某些边界会短暂同时保留 shard 与 full tensor。

## 11. 精度影响

并行分片数学等价，浮点归约顺序不同可能带来末位漂移。BF16/FP16 下 TP degree 改变时更明显，但不应出现系统性大误差。

验收应包括：单卡/多卡 logits 误差、greedy token 一致率、长上下文、非整除 padding、不同 batch shape。collective group 错误通常表现为 hang 或大错，不是正常浮点误差。

## 12. 选型原则

- 模型单卡放不下：先 TP；MoE 可用 EP 降专家权重。
- 模型能放下、目标是吞吐：优先增加 DP replica。
- 超长 context 使单卡 KV/Attention 不可承受：考虑 CP/KV split。
- 长 Prefill 的 TP 通信占比高：评估 FlashComm。
- 热门专家成为瓶颈：评估 EPLB，而不是盲目增加 EP。

## 13. 必测指标

- 每个 process group 的成员与 rank 映射。
- 各 collective bytes、次数、P50/P99。
- compute/communication overlap 比例。
- 每 rank 权重、KV、temporary buffer 峰值。
- rank 间 token/专家负载偏差。
- 跨节点与节点内带宽分别统计。

## 14. 初始化链路：配置怎样变成真实 group

进程启动后并不是模型层自己根据 `world_size` 猜 group。公共配置先得到 `dp_size`、`ep_size`、`cp_size` 和 `kv_split_size`；CollectiveCommunicator 再构造 MappingNPU/ProcessGroup，并把结果装入 `ParallelArgs`。NPU 构造时会先算 `attn_tp_size = world_size / (dp_size * cp_size)`，再分别设置 attention TP、MoE TP/EP、CP 和 KV-split 域，见 [collective_communicator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/collective_communicator.cpp:238)。

CP 大于 1 时还会单独构造 CP ranks，校验 group size、本地 rank 和 global rank 的对应关系，见 [collective_communicator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/collective_communicator.cpp:338)。DP 和 EP 也分别创建 TP-replica、DP 和 MoE 组；模型不能把 `tp_group`、`moe_ep_group`、`cp_group` 混用，否则轻则 shape 错，重则不同 rank 进入不同 collective 而挂死。

### 14.1 配置不变量

- `world_size` 必须能被所选并行域整除；
- `attn_tp_size × dp_size × cp_size = world_size`；
- `moe_tp_size × ep_size = world_size`；
- `kv_split_size` 为 0 时使用 CP 兼容语义，为 1 时每 rank 独占 KV，其他值必须满足 CP 的分组约束；
- 同一 communicator 的所有 rank 必须以相同顺序创建 group。

KV split 的默认/有效值定义见 [parallel_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.h:65)，NPU 映射给它建立独立 HCCL 域见 [mapping_npu.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/mapping_npu.cpp:208)。

## 15. 一个请求经过 DP+TP 的完整链路

以 `world_size=4, dp_size=2, cp_size=1` 为例，每个 DP replica 内有两个 TP rank：

```text
请求 A -> Scheduler 选择 DP0 -> global rank 0,1 同步执行
请求 B -> Scheduler 选择 DP1 -> global rank 2,3 同步执行

每个 replica 内：
embedding -> TP-sharded Attention/MLP -> AllReduce/ReduceScatter -> logits
```

Scheduler 只把请求分给一个 DP rank 的逻辑 batch；Engine 再把该 DP batch 发给属于同一 replica 的 Worker。每个 Worker 加载本 rank 的权重 shard并持有独立 KV pool。DP replica 之间通常没有逐层模型 collective，但服务端仍有请求路由、结果归并和容量反馈。

当前 Kimi K3 的五层 `DP2 + TP2` 已完成基础请求级验证：两个 replica 各自使用 TP2 组，DP replica 间没有 MLA collective；这只证明基础 MLA 服务链路，不能推广到 KDA DP、MoE DP+EP、Graph 或全层稳定性。该证据边界以 [research/PROJECT.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/research/PROJECT.md:240) 为准。

## 16. TP 的模型侧：Linear、Attention 和 LM Head

ColumnParallel Linear 把输出维切开。QKV 或 MLP gate/up 若后续算子能继续消费 shard，可以延迟通信。RowParallel Linear 把输入维切开，每 rank 得到局部部分和，通常必须 reduce 才恢复完整 hidden。公共 reduce 的同步封装最终调用 launch+finish，见 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:176)。

Attention 通常按 query/KV heads 切分；GQA/MLA 下本地 query heads 与本地 KV 表示未必同维。LM Head 可以使用独立 TP domain，NPU mapping 对其 group size 有专门限制，见 [mapping_npu.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/mapping_npu.cpp:177)。因此“TP=4”不是每个 layer 都机械使用同一个四卡布局。

## 17. CP 与 KV Split 的真实数据布局

CP 切 token/context 维，KV Split 切 KV cache 的归属宽度。两者相同时，可让每个 CP rank 保存相应上下文 shard；两者不同时，token 计算域与 KV 保存域需要额外重排。

`all_to_all_4D` 支持在 sequence/head 维之间交换四维 tensor，并显式检查可整除性，见 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:301)。ReduceScatter 则沿 dim 0 补齐后再分发，实际 padding 分配与调用见 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:194)。

这意味着 CP/KV Split 的正确性不只是 collective 返回成功，还要验证 position、mask、block table、每 rank sequence length 和最终去 padding都一致。

## 18. FlashComm1：为什么并入本篇

FlashComm1 的一句话模型是：长 Prefill 的 row-parallel 输出不再让每个 TP rank立即拥有 `[T,H]` 全量结果，而是 ReduceScatter 后每 rank 继续持有约 `[T/P,H]` 的 token shard。

资格条件由 [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:39) 直接给出：开关启用、当前是 Prefill、`cp_size==1` 且 token 数达到阈值。公共默认阈值 8192 见 [kernel_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kernel_config.h:75)。Decode、小 Prefill 和 CP>1 都应回退。

### 18.1 Padding 和布局不变量

token 数先向 alignment 取整，记录 `padded_num_tokens`、`pad_size` 和每 rank 的 `padded_local_num_tokens`，见 [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:104)。每 rank切出本地行，语义边界再 gather 并裁掉 padding，见 [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:116) 和 [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:143)。padding 行不能进入 position、KV 有效长度或采样。

### 18.2 MMRS

在 dtype、bias、shape、process group 等 predicate 满足时，RowParallelLinear 可以调用 Matmul+ReduceScatter，让 partial matmul 输出直接进入通信，见 [linear.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/common/linear.cpp:1594)。失败时回退普通 matmul/reduce；开关打开不等于每一层都命中。

FlashComm 不减少模型数学，也不必然减少端到端 collective 总字节。它的主要收益是延迟 full tensor materialization、减少中间激活驻留和融合 HBM 往返。

## 19. FlashComm 数字例子

设 Prefill `T=9000`、hidden `H=7168`、BF16、TP=4、alignment=16：

```text
padded_T = 9008
local_T  = 2252
full hidden bytes/rank  = 9008 * 7168 * 2 ~= 123.2 MiB
shard hidden bytes/rank = 2252 * 7168 * 2 ~= 30.8 MiB
padding rows            = 8
```

如果 shard 布局跨过多段计算，每 rank 可少常驻约 92.4 MiB 中间激活；若下一算子立刻 AllGather，则又付出恢复全量 tensor 的时间和峰值。阈值应从真实形状的交叉点测得，而不是固定认为 8192 对所有模型最优。

## 20. Multi-Stream：设计链路与当前状态

Multi-Stream 试图把 batch 切成两个 micro-batch：

```text
compute stream 0: [MB0 compute]       [MB0 next compute]
comm stream 0:           [MB0 reduce]
compute stream 1:        [MB1 compute]       [MB1 next compute]
comm stream 1:                    [MB1 reduce]
```

可重叠的是 MB0 的通信与 MB1 的独立计算，同一 micro-batch 的消费者仍必须等待自己的 collective。它通常不减少 bytes，反而把消息切小并增加 launch/event，同时让两套激活和 metadata 在途。

当前源码在 `enable_multi_stream_parallel && nnodes>1` 成立后直接 `LOG(FATAL)`，见 [master.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/master.cpp:356)。因此统一状态是 **已禁用**。文章只能解释设计和未来验收门槛，不能提供启用建议。

恢复前至少需要：micro-batch 输入/输出顺序测试、Graph on/off、allocator 跨流生命周期、异常取消、每种 backend 的 event 语义，以及 E5 timeline 证明。

## 21. Rolling Load：把 Host-to-Device 也放进通信账本

Rolling Load 不属于 rank collective，但属于推理关键路径上的数据搬运。Worker 在启用时取得独立 load stream，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:306)；初始化保留若干 device slot，其余 decoder 权重驻留 pinned Host，再逐层 H2D。

配置默认关闭，默认缓存 2 层、rolling slot 为自动值 `-1`，见 [load_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/load_config.h:53)。Worker 只在 NPU rolling 路径初始化 runtime，并把 cached layers/rolling slots 交给模型，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1584)。

### 21.1 Slot 映射

前 `N` 层先装入 N 个 slot；后续层循环使用前 K 个 rolling slot。K 为负时当前实现最多选 2 个，映射和 guard见 [rolling_load_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.cpp:55)。

### 21.2 Event 时间线

每层都有 `h2d_event[i]` 和 `compute_event[i]`。计算流先等待本层 H2D；本层计算后记录 compute event；load stream 等待该 event 才能覆盖同一 slot并装入下一层，见 [rolling_load_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.cpp:150) 和 [rolling_load_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.cpp:164)。

`finalize` 在正常或提前退出后恢复 dirty startup slots，避免下一请求把后层权重当成前层权重，见 [rolling_load_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.cpp:232)。

## 22. Rolling Load 的性能上界

设每个未缓存层 shard 300 MiB，Host-to-Device 有效带宽 24 GiB/s，则每层裸传输约 12.2 ms。若到下一次复用 slot 前只有 8 ms 独立计算，至少暴露约 4.2 ms stall。Decode 每生成一个 token 都走所有 decoder 层，因此这些传输会逐 token重复。

它的第一目标是“模型能放下”，而非保证低 TPOT。TP/EP 可让每 rank 只搬自己的 shard，但所有 rank 在下一 collective 处仍会等待最慢 H2D。

## 23. 三类通信的统一账本

| 类型 | 载荷 | 频率 | 主要瓶颈 | 本篇机制 |
|---|---|---|---|---|
| Rank collective | hidden、token shard、expert token | 每层/每 step | 小包延迟或大包带宽 | TP/CP/EP/FlashComm |
| Host-to-Device | decoder weight shard | 每层/每 forward | PCIe/HCCS、NUMA | Rolling Load |
| 控制面 | route、batch、rank/shape metadata | 每 step/请求 | CPU/RPC 延迟 | DP routing、Engine/Worker |

PD 的跨实例 KV 是另一套所有权协议，单独见 [13 PD 分离](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/13-pd-disaggregation-complete-chain.md:1)；MoE AllToAll 和 EPLB 则在 [11 MoE](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/11-moe-expert-parallel-complete-chain.md:1) 展开。

## 24. Overlap：证明条件而不是看函数名

`launch_gather`/`launch_reduce` 返回异步上下文，只有把 `finish_*` 延后到真正消费者前，且中间存在独立计算，才能形成真实 overlap。Launch/finish API 见 [parallel_state_async.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state_async.cpp:22) 和 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:127)。

静态代码最多证明依赖允许；E5 仍要用设备 timeline 观察 compute kernel 与 HCCL/H2D 时间区间重叠，并检查两者争用同一 DMA/片上资源后关键路径是否真的缩短。

## 25. 内存账本

| 项目 | 常驻/峰值 | 并行度影响 |
|---|---|---|
| 权重 shard | 常驻 | TP/EP 降低每 rank；DP 复制副本 |
| KV pool | 常驻 | DP 每副本独立；TP/GQA/MLA 按模型切分 |
| full/shard hidden | step 峰值 | FlashComm 可降稳态 shard，gather 边界可能双持 |
| collective buffer | collective 生命周期 | group/message/concurrency 增大而增加 |
| Rolling slots | 常驻 | cached slot 越多 HBM 越大、H2D窗口越长 |
| pinned weights | Host 常驻 | 近似未驻留设备层的完整 shard |
| Multi-Stream state | 设计峰值 | 两份 micro-batch 激活/metadata；当前已禁用 |

容量统一预算见 [17 显存篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/17-memory-and-vram-management-complete-chain.md:1)。

## 26. 精度和正确性

TP/CP/FlashComm/原样 Rolling H2D 的理论数学不变，但 collective 归约顺序、kernel shape 和 padding 会造成浮点末位差异。下面这些不是正常误差：

- rank group 不一致导致 hang；
- padding token 进入 position/KV/采样；
- AllToAll head/sequence 维反转；
- Rolling slot 在 compute event 前被覆盖；
- quantized weight 与 scale 不是同一层版本；
- DP 空 shard 构造出没有 cache metadata 的 fake input。

验证应从 shape/layout 断言、单层单卡/多卡对齐、greedy token 一致，逐步走到长上下文、多并发和 E5 压测。

## 27. 特性组合矩阵

| 组合 | 状态 | 边界 |
|---|---|---|
| TP + DP | 条件支持 | 每个 DP replica 内独立 TP group和 KV pool |
| CP + KV Split | 条件支持 | 需要整除、布局和 backend 支持 |
| FlashComm + CP | 已禁用 | eligibility 明确要求 `cp_size==1` |
| FlashComm + Decode | 已禁用 | eligibility 只接受 Prefill |
| Multi-Stream | 已禁用 | Master 入口 FATAL |
| Rolling Load + NPU manual loader | 条件支持 | pinned weights、load stream、模型 guard必须接通 |
| Rolling Load + Graph | 代码可达、未验证 | 固定地址与 slot覆盖需要专项 capture 验证 |
| Kimi K3 DP2+TP2 MLA | 条件支持 | 五层基础请求级通过；KDA/MoE/Graph未覆盖 |

## 28. 测试阶梯

- E0：检查 ParallelConfig、group mapping 和 feature guard；
- E1：日志打印每个 global rank 的 DP/TP/EP/CP/KV-split 成员；
- E2：运行 ReduceScatter padding、CP ranks、FlashComm eligibility、rolling slot/event 单测；FlashComm 资格用例见 [flash_comm1_context_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/common/flash_comm1_context_test.cpp:64)，ReduceScatter 多进程测试见 [parallel_state_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/parallel_state/parallel_state_test.cpp:90)；
- E3：固定模型完成 TP、DP+TP、长 Prefill FlashComm 和 Rolling 多请求 smoke；
- E4：与单卡/eager 基线比较 hidden、logits 和 greedy token；
- E5：扫 token 长度、并行度和节点边界，报告吞吐、TTFT、TPOT、P99、HBM、collective/H2D exposed time。

### 28.1 Process-group 验收矩阵

以 `world_size=8, dp=2, cp=2, ep=4` 为例，不能只打印一个 `rank=3`。每个进程应输出并核对：

| 域 | 预期 group size | 用途 | 错误症状 |
|---|---:|---|---|
| Attention DP | 2 | 请求/Attention 数据并行 | KV pool 归属或 batch 分发错误 |
| Attention CP | 2 | 上下文 token 分片 | 长上下文错位或 collective hang |
| Attention TP | `8/(2×2)=2` | QKV/输出投影 | hidden shape 错或归约失败 |
| MoE EP | 4 | expert placement/dispatch | token 发错 expert rank |
| MoE TP | `8/4=2` | expert 内矩阵分片 | expert output 归约错误 |
| KV Split | 配置有效值 | KV shard/gather | cache head或 token shard 错位 |

每个 group 还应验证成员顺序。集合相同但 rank 顺序不同，也会让 ReduceScatter 的第 r 片落到错误进程。CP 创建代码对 `cp_ranks[cp_local_rank] == global_rank` 做强校验，正是为了阻止这种“通信成功、语义错位”，见 [collective_communicator.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/collective_communicator.cpp:343)。

### 28.2 Collective 成本不只是一条带宽公式

Ring AllReduce 每 rank 网络字节近似：

```text
B_ar ~= 2 * (P - 1) / P * M
```

其中 P 是 group size，M 是 tensor 字节。ReduceScatter 与 AllGather 各约对应一个阶段：

```text
B_rs ~= (P - 1) / P * M
B_ag ~= (P - 1) / P * M
```

但服务延迟更接近：

```text
T_collective ~= alpha * phases + beta * bytes + T_queue + T_contention
```

`alpha` 是每阶段启动/同步延迟，`beta` 是有效带宽倒数。Decode 的 `[batch, hidden]` 小消息常由 alpha 主导；长 Prefill 的 `[tokens, hidden]` 大消息更多由 beta 和链路拥塞主导。FlashComm 把 full output 变成 shard 后，必须同时计算减少的驻留、增加的 padding和最终 gather，不能只比较一次 ReduceScatter 与 AllReduce。

### 28.3 非整除输入的正确性例子

设 dim-0 token 为 5、group size 为 4。公共 ReduceScatter 会把 token 行补到 8，每 rank获得 2 行；最后一个或多个 rank 的尾行可能是 padding。实现先计算 padded size，再构造 padded input 和 output，见 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:194)。

```text
real rows:     [0 1 2 3 4]
padded rows:   [0 1 2 3 4 0 0 0]
rank shards:   [0 1] [2 3] [4 pad] [pad pad]
```

后续消费者必须知道每 rank真实 token 数。若只看 tensor shape，padding 会被当成合法 token，污染 residual、position 或最终输出。

### 28.4 AllToAll 不是 AllGather 的别名

AllGather 让每 rank获得所有 rank 的片段；AllToAll 让不同目的 rank获得不同切片。CP 常在 sequence/head layout 之间做转置式交换，MoE EP 则按 expert destination 发送不等长 token。

`all_to_all_4D` 当前只接受特定 scatter/gather 维组合，并分别检查 head_num 或 seqlen 可整除，见 [parallel_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/parallel_state.cpp:318)。文章中的通信公式只能作为近似；真实不等长 MoE AllToAll 还受热点 rank 决定。

### 28.5 控制面失败与数据面失败要分开

控制面失败包括 Worker 未加入、rank table/端口冲突、配置不一致；数据面失败包括某个 collective 未被所有 rank以相同顺序调用、tensor shape 不一致、stream/event 等待环。前者通常在初始化或 RPC 日志暴露，后者可能表现为设备 hang。

排查顺序应是：

1. 确认所有 rank 进程和设备映射；
2. 打印每个 group 的 ordered rank list；
3. 为每个 collective 打印 layer、step、shape、dtype 和 group id；
4. 比较最后一个共同完成的 collective；
5. 再检查 stream/event 和 kernel。

不能只重启服务，因为孤儿 Worker 或占用端口会让新的 group 构造再次失败。

### 28.6 Rolling Load 的内存精算

设 decoder 共 L 层，每 rank单层权重 shard 为 W，常驻 slot 为 N，rolling slots 为 K：

```text
HBM_weight ~= non_decoder_weight + N * W_max
Host_pinned ~= sum(all decoder layer shards)
H2D_per_forward ~= sum(weights of layers not already in their needed slot)
```

`Host_pinned` 不是 `(L-N)×W` 的简单临时区：当前设计以 Host pinned 权重作为 source of truth，Worker 的注释明确了这一点，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1566)。节点主存、NUMA locality 和 locked memory limit都必须纳入部署预算。

K 增大通常拉长预取距离，却不会无限降低等待：如果 H2D 总流量已超过互联可提供的稳态带宽，更多 slot 只能改变 burst，无法消除平均 stall。

### 28.7 Rolling Load 的异常路径

提前退出、exception、请求取消或模型只执行部分层时，最后已执行层不一定是 L-1。`RollingLayerGuard` 析构调用 `finalize(last_executed_layer)`，见 [rolling_load_manager.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu/loader/rolling_load_manager.h:130)。Finalize 只恢复 dirty slot，并在覆盖前等待该 slot 最后的 compute event。

专项测试应给每层写入可识别常数权重，依次覆盖：完整 forward、在不同层抛出异常、连续两个请求、sleep/wakeup 后地址刷新、量化 weight+scale 同步迁移。只验证首轮完整 forward 会漏掉最危险的旧权重复用问题。

### 28.8 FlashComm 的模型接入条件

资格判断只是入口。要让 shard 布局跨层存活，模型 residual、norm、Attention/MLP 和输出边界都必须知道当前 tensor 是 full 还是 sequence-sharded。`maybe_shard_residual` 接受 full real-token 或 padded local shard，并检查行数不变量，见 [flash_comm1_context.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/flash_comm1_context.cpp:217)。

因此新模型接入 FlashComm 的最低检查项是：

- 哪里建立/销毁 FlashComm context；
- 第一个 ReduceScatter 在哪一层；
- residual 是否与 hidden 使用相同 layout；
- 哪个语义边界执行 gather；
- position/KV metadata 是否仍按真实 token；
- quantized RowParallelLinear 是否命中 MMRS或正确回退；
- Graph capture 是否固定 padding shape。

### 28.9 性能 A/B 设计

并行性能实验应固定模型、请求集、batch policy、KV budget 和随机种子，只改变一个轴：

| 实验 | A | B | 主要指标 |
|---|---|---|---|
| TP 扩展 | TP1 | TP2/4 | 单层 compute、collective、TPOT、每卡 HBM |
| DP 扩展 | 1 replica | 2 replicas | request throughput、P99、每副本 KV |
| FlashComm | off | on | RS/AG bytes、full/shard peak、TTFT |
| MMRS | off | on | kernel launch、HBM bytes、collective exposed time |
| Rolling | 全常驻 | N slots | HBM、H2D bytes、compute wait、TPOT |

跨节点测试必须单独报告节点内与节点间流量。把 TP2 单机和 TP4 跨机画在同一条“卡数扩展曲线”上，会把拓扑跳变误认为并行度规律。

### 28.10 可观测性清单

最小线上指标包括：

- 每个 group 的 collective 次数、字节、P50/P99；
- device compute、collective、H2D 三条 timeline 的 exposed/overlapped 时间；
- 每 rank权重、KV、激活、通信 workspace 峰值；
- DP rank 的 waiting/running/token 数和 free blocks；
- FlashComm eligible、actual-hit、fallback、padding rows；
- Rolling 每层 H2D time、slot wait、dirty/refill 次数；
- collective timeout、rank failure、group rebuild 数。

“设备利用率提高”必须和 request/token throughput 一起看。通信 busy 或无效 padding 也可能抬高设备占用，却没有改善服务 SLO。

### 28.11 Kimi K3 最小后续验证

当前最有价值的下一步不是直接把所有通信开关叠上，而是：

1. 固定五层影子模型、输入和 TP2 基线；
2. 对 DP0/DP1 的 Full MLA prefill/decode 中间张量做 rank 内/副本间比较；
3. 接入 KDA DP 后验证 linear state ownership；
4. 接入 MoE DP+EP 后记录每个 MoE group 和 dispatch matrix；
5. 最后再分别打开 Chunked Prefill、Graph 或其他 overlap，确保首个偏差可定位。

现有五层输出相似度结论是阶段性 E4，不是逐元素严格一致；并行文章不能把“请求输出相同”提升成所有中间层和所有拓扑已经等价。

## 29. 常见误区与源码阅读顺序

常见误区：DP 等于没有通信；TP 越大越快；async API 等于 overlap；FlashComm 会减少所有 collective；Rolling Load 释放多少 HBM就加速多少；Multi-Stream 有 flag 就能开启。

建议顺序：

1. ParallelConfig 默认值；
2. CollectiveCommunicator rank 分解；
3. MappingNPU 各 domain；
4. ParallelArgs 模型消费者；
5. parallel_state collective 和 padding；
6. FlashComm context 与 RowParallelLinear；
7. Master Multi-Stream guard；
8. Worker rolling 初始化与 RollingLoadManager event。

## 30. 读完后应能回答的问题

1. 为什么 DP2+TP2 的 world size 4 不是一个四卡 TP group？
2. CP 与 KV Split 不同宽度时为什么需要额外布局协议？
3. FlashComm 在什么边界必须恢复 full token tensor？
4. `launch_reduce` 之后立刻 `finish_reduce` 为什么没有可见 overlap？
5. Rolling slot 增多怎样同时改变 HBM 和 H2D 暴露时间？
6. 为什么当前 Multi-Stream 必须写“已禁用”？
7. 多卡末位误差与 group/layout 错误怎样区分？

## 31. 结论

并行不是“卡越多越快”，而是用 collective 换单卡计算和内存；FlashComm 是 TP 长 Prefill 的布局优化，Rolling Load 是 Host 容量换 HBM 的权重流水，Multi-Stream 当前只是被 guard 阻断的设计。正确决策要同时画出 rank 拓扑、tensor layout、event 依赖和内存生命周期：只有减少的计算/驻留时间大于新增通信、同步、padding和搬运，扩卡或叠加快路径才真正有效。

把全文压成部署选择：

```text
模型单卡放不下
  -> 先用 TP/EP 分片权重
模型能放下但吞吐不足
  -> 增加 DP replica
上下文状态或 Attention 单卡不可承受
  -> 评估 CP/KV Split
长 Prefill 的 TP full hidden 成为瓶颈
  -> 在满足资格的模型上评估 FlashComm/MMRS
设备连权重 shard 都放不下
  -> 把 Rolling Load 当容量兜底，并接受 H2D TPOT 成本
希望直接开启 Multi-Stream
  -> 当前停止：运行时明确已禁用
```

每一步都要重新计算最弱 rank 的 HBM、消息大小、同步频率和端到端 SLO，上一层成立不代表下一层组合仍成立。
