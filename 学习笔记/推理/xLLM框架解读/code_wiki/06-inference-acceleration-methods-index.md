# xLLM 推理加速方法全景：源码入口、收益来源与组合边界

> 本文是加速专题的卷首和唯一导航入口。分析基于当前工作区源码，而不是只根据功能文档推断。知识组织按“调度与执行、KV 与容量、分布式通信、MoE、数值与模型、生成式视觉”六个知识域展开，不再按每个 flag 建一篇平级文章。

## 1. 先建立正确的性能模型

推理耗时不是一个数字，而是下式中若干瓶颈的最大值与串行部分之和：

```text
T_request ≈ T_queue + T_prefill + Σ T_decode_step + T_transfer

T_layer ≈ max(T_compute, T_HBM, T_collective, T_H2D) + T_unhidden
```

因此，“加速”至少有六种不同含义：

1. 少算：Prefix Cache、投机解码、DiT Cache、稀疏注意力。
2. 算得更便宜：量化、融合算子、专用 Attention/MoE kernel。
3. 少搬数据：Paged KV、MLA、量化 KV、融合算子。
4. 把通信藏到计算后面：FlashComm、多流、滚动加载、异步调度。
5. 提高设备占用率：连续批处理、Chunked Prefill、Graph、批次混排。
6. 改变系统拓扑：TP/DP/EP/CP、PD 分离、分层 KV。

同一个开关可能提高吞吐却恶化 TTFT，也可能降低显存却增加总线压力。后续文章不使用“性能更好”这种无条件结论。

## 2. 六卷知识体系

| 卷 | 核心问题 | 权威文章 | 收纳的方法 |
|---|---|---|---|
| A 调度与执行 | 一轮算谁、算多少、怎样降低提交等待 | [02 Scheduler](02-scheduler-complete-chain.md)、[04 Chunked Prefill](04-chunked-prefill-complete-chain.md)、[09 Async Schedule](09-async-schedule-and-overlap-complete-chain.md)、[03 Graph](03-graph-execution-complete-chain.md) | Continuous Batching、Zero-Eviction、Chunk、CPU/Device overlap、Graph |
| B KV 与容量 | 历史状态怎样分页、复用、迁移和计费 | [07 Paged KV](07-paged-kv-and-paged-attention-complete-chain.md)、[08 Prefix Cache](08-prefix-cache-complete-chain.md)、[17 显存](17-memory-and-vram-management-complete-chain.md) | Paged Attention、Prefix、Host Offload、远端边界、XTensor、容量规划 |
| C 分布式通信 | 模型、token、上下文和权重怎样跨 rank/Host 切分与搬运 | [10 分布式并行与通信](10-distributed-parallelism-and-communication.md)、[13 PD 分离](13-pd-disaggregation-complete-chain.md) | DP/TP/CP/KV Split、FlashComm、Multi-Stream 状态、Rolling Load、PD |
| D MoE | token 怎样路由到专家，热点怎样治理 | [11 MoE 与 EPLB](11-moe-expert-parallel-complete-chain.md) | Router、EP、AllToAll、Grouped GEMM、冗余专家、在线迁移 |
| E 数值与模型 | 每 token 必须保存/计算多少，单算子怎样变便宜 | [12 量化](12-quantization-complete-chain.md)、[14 Attention 架构](14-attention-architecture-acceleration.md)、[15 算子与 Kernel](15-operator-fusion-and-custom-kernels.md)、[05 投机解码](05-speculative-decoding-complete-chain.md) | W8A8/FP8/KV INT8、GQA/MLA/线性/稀疏、融合、自定义 kernel、Spec Decode |
| F 生成式视觉 | 多 diffusion step 和高分辨率 latent 怎样加速 | [16 DiT](16-dit-acceleration-complete-chain.md) | DiT Cache、SP、稀疏 Attention、VAE Tiling |

这套结构刻意让“方法数”多于“文章数”。例如 FlashComm 是 TP 长 Prefill 的布局/collective 快路径，Rolling Load 是 Host 与设备之间的权重流水，都必须先建立分布式 rank、stream 和所有权模型，因此统一收进第 10 篇；EPLB 若脱离 MoE dispatch/combine 就无法解释其收益和正确性，因此统一收进第 11 篇。

### 2.1 方法状态速查

| 方法 | 主要收益 | 主要代价 | 当前状态 | 权威位置 |
|---|---|---|---|---|
| Continuous/Zero-Eviction | 利用 batch；减少抢占重算 | admission 更保守或调度更复杂 | 条件支持 | 02 |
| Chunked Prefill | 控制长 Prompt 单轮占用 | 多轮状态与边界复杂 | 条件支持 | 04 |
| Async Schedule | CPU 调度与设备执行重叠 | 状态滞后一拍、在途内存 | 条件支持 | 09 |
| Paged KV/Prefix/Host Offload | 降碎片、复用历史、扩展容量 | block/LRU/传输管理 | 条件支持 | 07、08 |
| TP/DP/CP/KV Split | 分摊权重、请求或上下文 | collective、同步和拓扑约束 | 条件支持 | 10 |
| FlashComm1 | 长 Prefill 减少完整激活驻留 | padding、布局边界、NPU 限制 | 条件支持 | 10 |
| Multi-Stream | 设计目标是 micro-batch 通信计算重叠 | 双份在途状态 | 已禁用 | 10 |
| Rolling Load | 用 Host 容量换 HBM 权重常驻 | 每次 forward 的 H2D | 条件支持 | 10 |
| MoE EP/EPLB | 分片专家权重、缓解热点 | AllToAll、冗余和迁移 | 条件支持 | 11 |
| 量化 | 降权重/KV字节和带宽 | scale、kernel 与质量误差 | 条件支持 | 12 |
| PD 分离 | 独立扩缩 P/D 池 | KV 交接和故障协议 | 条件支持 | 13 |
| MLA/线性/稀疏 Attention | 从模型结构降低状态或计算量 | 模型专用状态与近似风险 | 条件支持 | 14 |
| 融合/专用 Kernel | 减少 launch 和 HBM 往返 | capability matrix、workspace | 条件支持 | 15 |
| DiT Cache/SP/Sparse/VAE | 跨 step 少算或降低 latent 峰值 | 质量误差、通信和边界 | 条件支持 | 16 |

## 3. 四类“精度影响”不能混为一谈

### 3.1 数学等价，但浮点顺序可能变化

TP、reduce-scatter、Graph、融合算子通常属于此类。模型公式不变，但 collective 归约顺序、融合后的累加顺序或 kernel 实现可能产生末位差异。验证目标应是容差和生成一致率，不是逐 bit 相等。

### 3.2 精确复用历史计算

Prefix Cache、Paged KV、主机 KV offload 原则上只保存并恢复同一份 KV，不主动近似。前提是 cache key 覆盖模型、适配器、token、位置、多模态输入等全部语义。

### 3.3 低精度表示

W8A8、FP8、INT8 KV 明确改变表示。它们可能改善吞吐和容量，也必须以任务集、困惑度、首 token/长上下文误差和生成一致率评估。

### 3.4 算法近似

投机解码在接受/拒绝校正正确时保持目标分布；而 DiT Cache、运行时稀疏注意力会主动跳算或裁剪连接，质量是参数的一部分。

## 4. 通信分析的统一口径

后续每篇都区分：

```text
设备内：HBM <-> compute core
设备间：HCCL/NCCL all-reduce、all-gather、reduce-scatter、all-to-all
主机设备：PCIe/HCCS 上的 H2D/D2H
节点间：RDMA/TCP/共享存储上的 KV 或请求传输
```

“通信量下降”也不等于“延迟下降”。小消息受启动延迟影响，大消息受带宽影响；通信只有在存在足够独立计算并且流与依赖关系正确时才能被隐藏。

## 5. 内存分析的统一口径

显存至少分成：权重、KV/状态、激活、Graph 静态池、通信 buffer、workspace、临时 padding。主存还可能包含 pinned 权重、host KV 和请求元数据。

常用近似式：

```text
KV_bytes/token ≈ 2 × L × N_kv_heads × D_head × bytes(dtype)

Dense_weight_bytes ≈ parameter_count × bytes(weight_dtype)

MoE_local_weight_bytes ≈ active_parameter_layout / EP_degree
                         + redundant_expert_bytes
```

MLA、GQA、KV 量化改变第一式；EP、量化、Rolling Load 改变权重常驻量；Graph 和多流增加激活/静态 buffer。

## 6. 组合特性时的正确顺序

1. 先确定模型语义：MHA/GQA/MLA、MoE、线性层、量化格式。
2. 再确定并行拓扑：DP/TP/CP/EP 与每组 rank。
3. 再算静态内存：权重、Graph 池、通信 buffer。
4. 再把剩余显存给 KV，并选择 block size、prefix/offload。
5. 最后调调度：batch token、chunk、zero-eviction、overlap。
6. 用真实流量联合观察 TTFT、TPOT、吞吐、P99、重算率和命中率。

反过来盲目叠加开关，最常见的结果是某一项 microbenchmark 变快，而服务 P99 或稳定容量变差。

## 7. 源码事实与文档事实的边界

本专题采用以下优先级：

```text
当前构造/校验代码 > 当前运行路径 > 单测 > CLI 文档 > 历史设计说明
```

两个已确认的例子：

- `master.cpp` 对启用 multi-stream 的配置直接 `LOG(FATAL)`，所以它目前不是可部署能力。
- 旧 Global KV 文档提到 `enable_cache_upload`，当前配置实际使用 `enable_kvcache_store`、`host_blocks_factor` 等，且 worker 初始化里的远端 store 仍被关闭；不能把主机 offload 与远端全局缓存混成一个已完成特性。

## 8. 阅读路线

第一次阅读不再按编号顺扫。建议按“服务状态从哪里来、怎样被计算、怎样跨设备”推进：

```text
02 Scheduler
  -> 04 Chunked Prefill
  -> 07 Paged KV
  -> 08 Prefix Cache
  -> 17 Memory
  -> 03 Graph
  -> 10 Parallelism + Communication
  -> 11 MoE + EPLB
  -> 05 Speculative / 14 Attention Architecture
```

遇到线上容量问题时则按“内存 -> 通信 -> 调度”逆向定位。

## 9. 合并决策与连续编号

| 被吸收的主题 | 权威目标 | 原因 |
|---|---|---|
| Zero-Eviction | 02 Scheduler 第 27 节 | 它是 admission policy，不形成新的模型/backend 链路 |
| Hierarchical/Global KV | 07 Paged KV 第 28 节 | Host block、D2H/H2D 与 Paged KV 所有权不可分 |
| FlashComm、Multi-Stream、Rolling Load | 10 分布式并行与通信 | 三者分别是 collective 布局、通信 overlap 状态和 Host-to-Device 权重流水 |
| EPLB | 11 MoE 与 EPLB | 统计、冗余、迁移最终都服务于 MoE dispatch |

合并后不保留旧文件或迁移页。专题正文从 07 到 17 连续编号，所有知识库内部链接只指向当前权威文件。

## 10. 为什么有些开关没有再拆一篇

本文按“可独立解释和验收的加速机制”拆篇，不按每个 bool 字段拆篇。以下配置已归入其真正所属机制：

- `enable_double_buffer`、`enable_no_padding`、piecewise graph、Python graph backend、VMM pool：属于 Graph 的输入稳定化、buffer 和捕获策略，见 [03](03-graph-execution-complete-chain.md)。
- mix batch、priority、max batch tokens/sequences：属于连续批处理和调度策略，见 [02](02-scheduler-complete-chain.md)。
- fused MC2、AddNorm、RMSNorm-RoPE、ACLNN matmul/SwiGLU、MMRS：属于算子/通信融合，见 [15](15-operator-fusion-and-custom-kernels.md)。
- `enable_xtensor`、physical page granularity：属于 KV 物理页与 PD 数据通路的具体 allocator/backend，见 [07](07-paged-kv-and-paged-attention-complete-chain.md) 与 [13](13-pd-disaggregation-complete-chain.md)。
- sleep mode 是模型驻留状态管理，Beam Search 是生成搜索算法；它们可能改变容量或输出质量，但不作为本轮“保持同一服务语义的推理加速方法”单列。
- 各设备的同类 Attention/GEMM kernel 不重复拆篇，在算子融合专题中按 capability matrix 理解；模型架构定义的 Attention 则在 [14](14-attention-architecture-acceleration.md) 单列。

这个边界既避免把一个机制拆成十几个开关，也不会漏掉配置层已经暴露的加速能力。
