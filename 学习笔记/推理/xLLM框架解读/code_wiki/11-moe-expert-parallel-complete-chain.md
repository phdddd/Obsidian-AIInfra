# xLLM MoE Expert Parallel：路由、Dispatch、Grouped GEMM 与 Combine

## 1. MoE 为什么需要专门加速

MoE 每个 token 只激活少量 experts，FLOPs 低于同参数量 dense 模型，但全部专家权重仍可能放不进单卡。EP 把 experts 分给不同 ranks，并把 token 发送到其目标 expert 所在 rank。

它用动态 all-to-all 换取专家权重分片，性能上限往往由负载均衡和通信而非理论 FLOPs 决定。

## 2. 一层完整链路

```text
hidden states
  -> router logits
  -> top-k expert ids + weights
  -> token count / layout metadata
  -> dispatch all-to-all
  -> local experts grouped GEMM
  -> combine all-to-all
  -> 按 routing weights 累加回原 token 顺序
```

shared experts、routed experts、expert bias/score correction等模型差异会插入这条主线。

## 3. 源码入口

| 层次 | 当前入口 | 证明什么 |
|---|---|---|
| 并行配置 | [parallel_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/parallel_config.h:59) | EP degree 与其他并行域的关系 |
| EPLB 配置 | [eplb_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/eplb_config.h:45) | 冗余、interval、threshold 和 rank table 默认值 |
| 通信域 | [mapping_npu.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/mapping_npu.cpp:192) | MOE_TP/MOE_EP/dispatch-combine group |
| NPU Torch MoE | [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:354) | router、权重 shard、local/EP 快路径 |
| Top-k | [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:750) | ids/weights 的生产和 shape 校验 |
| Local experts | [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:853) | 排序后 Grouped GEMM 与 combine |
| EP2 dispatch | [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1431) | token 跨 rank 发送 |
| EP2 combine | [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1568) | expert output 回原 rank |
| EPLB Manager | [eplb_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_manager.cpp:37) | 负载汇总和控制状态机 |
| EPLB Policy | [eplb_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_policy.cpp:43) | 何时更新和怎样放置冗余 expert |
| EPLB Worker | [eplb_executor.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_executor.cpp:47) | 异步准备权重和 ready 回执 |
| Engine 汇总 | [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1182) | 多 Worker load 怎样回到 Manager |

## 4. EP Level 1 与 Level 2 的直觉

文档中的 Level 1 更接近较粗粒度的跨域 gather/同步；Level 2 使用 all-to-all，只把 token 送到真正需要的 expert rank。后者减少无用计算与权重复制，但元数据和不规则通信更复杂。

当前是否命中某种融合路径必须以 backend/model 分支为准，不能只按 Level 名称推断。

## 5. Grouped GEMM

每个 expert 收到的 token 数不同。逐 expert 启动 GEMM 会产生大量小 kernel；grouped GEMM 把多个不同 M 的矩阵乘合并调度：

```text
for expert e:
  Y_e = activation(X_e W_gate_up_e) W_down_e
```

性能依赖 token 按 expert 排列、每 expert count、空 expert 处理和量化 scale 布局。热点 expert 会形成大 GEMM，冷 expert 形成低效小 GEMM。

## 6. 通信分析

若 hidden dtype 字节数为 `b`，top-k 为 K，粗略 dispatch payload：

```text
tokens × K × hidden_size × b
```

combine 还要返回 expert output。实际协议可能发送去重/压缩后的 token、routing metadata 和 counts。all-to-all 的困难在于每个 peer 消息不等长，最慢/最热 rank 决定层尾延迟。

跨节点 EP 通常远贵于节点内 EP，应让拓扑映射与专家布局协同。

## 7. Overlap

可尝试：counts/metadata 与 token payload 分阶段、dispatch 与 shared expert 计算重叠、分块到达后启动 local experts、combine 与后续独立工作重叠。

源码中的 async/fused dispatch-combine 或 LCOC 受 backend 与 Graph 限制。异步线程或 stream 存在不等于端到端已重叠，必须从 timeline 证明。

## 8. 内存影响

EP 的主要收益是每卡 routed expert 权重约降到 `1/EP`。代价包括：

- dispatch/send/recv buffer。
- token permutation 与反向索引。
- per-expert count/offset。
- grouped GEMM workspace。
- shared experts 仍可能复制。
- EPLB 冗余 experts 额外占权重。

大 batch/top-k 会让 token buffer 成为显著峰值。

## 9. 精度影响

EP 本身保持路由与 expert 数学。浮点差异来自 token 重排后的累加顺序、量化 expert、融合 kernel。必须确保：

- top-k ids 与 weights 在 dispatch 后不串 token。
- duplicate expert route 正确累加。
- empty/local/remote expert 都处理。
- combine 恢复原 token 顺序。

## 10. 负载不均

平均每 expert token 数并不能描述尾部。应观察 max/mean、rank send/recv matrix、每 expert GEMM M 值和空 expert 比例。

当热点长期稳定，可用 EPLB 重新布局或增加冗余副本；当热点随 prompt 快速变化，迁移可能追不上，需更平滑的路由或更短统计窗口。

## 11. TP 与 EP 组合

同一 MoE 层可同时有 EP 和 MoE TP：EP 切 experts，MoE TP 切 expert 内矩阵。它降低超大 expert 单卡计算/权重，但引入 expert 内 collective。

映射代码分别维护 Attention TP、MOE_TP、MOE_EP。world size 的分解必须与模型层实际读取的 group 一致。

## 12. 测试矩阵

- top-k=1/2/更多、均匀/极端热点路由。
- token 数小于/大于 expert 数。
- 单机与跨机 EP。
- BF16、W8A8/FP8 expert。
- Graph on/off 与融合 dispatch/combine on/off。
- EPLB 更新前后输出和负载。
- 任意 rank 无 token、请求取消和空 batch。

## 13. 配置与初始化

`ep_size` 决定 routed experts 的基本分片宽度；`enable_eplb` 默认 false，`redundant_experts_num` 默认每设备 1 个，更新 interval 默认 1000，阈值默认 0.8，见 [eplb_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/eplb_config.h:45)。这些默认值只说明配置，不证明目标模型已接通 EPLB。

Engine 启动时在开关成立后创建 `EplbManager`，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:151)；Worker 创建 `EplbExecutor`，见 [llm_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/llm_worker_impl.cpp:119)。Manager 负责全局策略，Executor 负责本 Worker 的权重准备，模型 MoE layer 负责产生实际 load 和消费 mapping。

## 14. 一轮 MoE forward 的数据契约

设 hidden 为 `[T,H]`，每 token top-k=K：

```text
router_logits  [T,E]
topk_ids       [T,K] int32/int64
topk_weights   [T,K] floating
dispatched_x   [sum routed copies,H]
expert_counts  [physical experts]
expert_output  [sum routed copies,H]
combined       [T,H]
```

top-k 权重和 id 的 token 维必须等于 hidden rows，当前 NPU Torch 路径有显式检查，见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:760)。Top-k 后 token 可能复制 K 份；Combine 必须按原 token和 routing weight 累加，不能只做逆 permutation。

## 15. Local MoE 与 EP MoE 的分岔

Local 路径在本 rank 筛出本地 expert route、重排 token、执行两段 Grouped GEMM，再用 combine index 恢复。Combine 的输入、weight 和 gather id 构造见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1077)。

EP2 路径则把 hidden、expert ids/weights 和通信域交给 distribute-dispatch，返回扩展 token、counts 与 combine assist metadata；本地 expert 计算完成后再调用 distribute-combine。这个 assist metadata 是跨算子的状态，不允许在 Graph replay 或异步流水中被下一 batch 提前覆盖。

## 16. 通信量与热点

粗略 dispatch payload：

```text
B_dispatch ~= T * K * H * bytes(dtype)
B_combine  ~= T * K * H * bytes(dtype)
```

实际协议还含 ids、weights、send counts、assist info，且本地 route 不一定经过互联。最关键的不是平均字节，而是 peer-to-peer 矩阵：热点 expert 所在 rank 收到更多 token，最终层延迟由最慢 rank 的 `communication + grouped GEMM` 决定。

跨节点 EP 会把不规则 AllToAll 放到较慢链路上。部署应优先让一个 EP group贴合高带宽域，并单独报告节点内/节点间流量。

## 17. Grouped GEMM 的 shape 和 workspace

每个 expert 的 GEMM M 维等于收到的 token 数。Grouped GEMM 用 counts/offsets把多个不同 M 的矩阵乘合并提交，降低大量小 kernel launch；但极端热点仍会让一个 expert 的大 M 成为尾部，很多冷 expert 的 M=0 也必须安全跳过。

量化 expert 还需要 weight scale、activation scale 和可能的 layout prepare。EP2 W8A8 路径明确要求 w13/w2 scale，见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1463)。

## 18. EPLB 为什么属于 MoE 主链

EPLB 的输入是每层每逻辑 expert 的 token load，输出是 `[layers, devices, local_experts]` 的物理分布和待更新 layer。Manager 接收各 Worker load，后台线程聚合后调用 Policy，见 [eplb_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_manager.cpp:137)。

Policy 将当前 load 与上次 load 归一化后计算余弦相似度；低于 update threshold 才更新该层，见 [eplb_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_policy.cpp:47)。因此 threshold 衡量“分布变化是否足够大”，不能简单解释成 max/mean 失衡比例。

## 19. 冗余 expert 的放置算法

Policy 为热点逻辑 expert 创建额外物理 id，并把原 load按副本数近似分摊；随后以贪心方式把副本/专家放到当前负载较低的 device，见 [eplb_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_policy.cpp:141)。

冗余改变的是物理 destination，不改变 router 的逻辑 top-k 语义。同一个逻辑 expert 的所有副本必须拥有同版本权重、scale 和 adapter；否则 EPLB 会把相同 token随机送到数学不同的函数。

## 20. Prepare、Publish、Retire 协议

安全更新至少分三步：

1. Manager 发送待准备 layer 和 expert ids；
2. Worker 后台线程调用模型准备权重并返回 ready layer；
3. 全局确认后发布新 mapping，旧权重之后才可退休。

Executor 的后台 worker在 [eplb_executor.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_executor.cpp:83) 执行 `prepare_expert_weight` 并设置 ready id。Engine 每轮把 `EplbInfo` 放进各 DP batch，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:1261)。

异步准备不等于迁移零成本：Host/设备 copy、HBM 写入和互联仍会与推理争用。

## 21. 内存账本

```text
local_expert_weight ~= base_experts_per_rank * bytes_per_expert_shard
redundant_weight    ~= redundant_experts_per_device * bytes_per_expert_shard
token_buffer        ~= T * K * H * dtype_bytes * in_flight_copies
```

还要计入 router logits、ids/weights、send/recv buffer、per-expert counts、permutation index、Grouped GEMM workspace，以及更新期新旧 expert 双持。共享 expert通常在各 rank复制，不随 routed EP 等比例下降。

EPLB 的 Host shared-memory expert buffer 由 [expert_buffer_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/expert_buffer_manager.cpp:20) 创建，也要纳入主存和共享内存限制。

## 22. Overlap 的真实边界

可以静态并行的候选包括 shared expert compute 与 routed dispatch、下一批权重 prepare 与当前推理、分块 token 到达后的本地 expert compute。硬依赖包括 top-k 后才能 dispatch、目标权重 ready 后才能消费、combine 完成后才能继续 residual。

是否真实重叠必须用 timeline 证明：dispatch/combine stream、Grouped GEMM、shared expert和 weight copy 在时间上同时存在，且层尾关键路径变短。单看后台线程或 fused operator 名称不够。

## 23. 精度和状态正确性

EP/EPLB 理论上不改变 router 或 expert 数学，浮点差异来自 token 重排、combine 累加顺序、量化和不同 kernel。逻辑错误则包括：token 重复/遗漏、top-k weight 与 id错配、空 expert 越界、mapping 版本不一致、scale 漏迁移、Graph replay assist buffer被覆盖。

最小对齐应比较 router logits、top-k ids/weights、每 expert counts、expert output、combined hidden 和最终 logits，而不是只看聊天文本。

## 24. 数字例子

设 T=256、H=7168、K=2、BF16、EP=8。两次 token payload理论总量约：

```text
one direction = 256 * 2 * 7168 * 2 ~= 7 MiB
dispatch + combine ~= 14 MiB/layer
```

若一个 rank收到平均值 3 倍的 token，层尾不会按 14 MiB 平均时间完成，而由该 rank 的收包和 GEMM决定。增加一个热点 expert副本只有在未来若干窗口节省的尾部时间大于权重准备和额外显存时才值得。

## 25. 特性组合矩阵

| 组合 | 状态 | 边界 |
|---|---|---|
| MoE TP + EP | 条件支持 | group 分解与权重 shard必须一致 |
| EP + W8A8/W4A8 | 条件支持 | scale/layout/backend 专用路径 |
| EPLB + EP | 条件支持 | 需完整 prepare/publish 回执 |
| EPLB + Graph | 代码可达、未验证 | mapping/weight 地址更新需专项验证 |
| EPLB + Rolling Load | 未确认 | 两种权重迁移/slot 所有权可能冲突 |
| Kimi K3 MoE EP/EPLB | 代码可达、未验证 | 当前五层 DP2+TP2 不覆盖 MoE DP+EP |

## 26. 性能指标和调参

必须观测 per-expert token、rank max/mean、AllToAll matrix、dispatch/combine 时间、Grouped GEMM M分布、空 expert比、冗余命中、迁移 bytes/time、更新期 TPOT/P99、HBM 和 shared-memory。

interval 越小响应越快但噪声/迁移更多；threshold 越高通常更少更新；冗余越多热点分摊潜力越大但权重内存线性增加。目标是最小化层尾时间，不是追求每个 expert token 完全相等。

## 27. 测试阶梯

- E0：配置、group、kernel 和 guard 静态存在；
- E1：日志证明目标 local/EP/EPLB 分支执行；
- E2：top-k、permutation、empty expert、dispatch/combine、policy 和更新协议单测；
- E3：均匀/热点真实 MoE 请求 smoke；
- E4：单卡/多卡及更新前后逐层对齐；
- E5：跨节点、长稳、热点突变和更新故障注入 benchmark。

### 27.1 Router 层测试

Router 单测至少覆盖 softmax/sigmoid scoring、grouped top-k、bias correction、是否 renormalize、相同分数 tie、NaN/Inf 和 token 数为零。测试输出不能只看 shape，还要逐 token比较逻辑 expert id 与 weight。

若模型允许预选 top-k，预选结果与内部 router 两条路径必须分别验证。当前实现会把预选 id转换为 int32、weight转为 hidden dtype，并检查 token 维，见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:750)。

### 27.2 Dispatch/Combine 守恒测试

为每个 token写入唯一编号，构造 top-k=2 且同时包含本地、远端、重复逻辑 expert 的 route。Dispatch 后检查：

- routed copy 总数等于有效 route 数；
- 每个 peer send count 与接收端 recv count一致；
- assist index 可逆；
- empty rank仍参与 collective；
- Combine 后每个原 token恰好得到 K 个加权贡献。

EP2 dispatch 参数把 ids、weights、EP rank/world 和 counts 一起交给设备算子，见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1431)。Combine 必须消费同一批 assist/counts，见 [fused_moe.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/layers/npu_torch/fused_moe.cpp:1568)。

### 27.3 Grouped GEMM 测试

用 expert counts `[0,1,2,17]` 覆盖空矩阵、极小 M 和正常 M；每个 expert使用不同常数权重，以便发现 token发错 expert。BF16、W8A8、W4A8 分别与逐 expert reference对齐，scale 缺失必须显式失败，不能静默使用错误默认值。

### 27.4 EPLB Policy 测试

构造稳定均匀、稳定热点、热点突然迁移和全零四种窗口。分别验证：

```text
稳定分布 -> 不反复更新
热点变化超过阈值 -> 目标 layer update flag=true
冗余副本 -> 优先分给高负载逻辑 expert
placement -> 每设备槽数固定且物理 id合法
```

Policy 的输出 shape 和 update flag 契约定义在 [eplb_policy.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_policy.h:38)。

### 27.5 EPLB 并发与故障注入

延迟一个 Worker 的权重 prepare，确认 Manager 不提前发布；让一个 Worker prepare 失败，确认旧 mapping继续可用；在更新时取消请求，确认不影响 mapping版本；连续触发两次热点变化，确认旧任务不会覆盖新版本。

Manager 内同时有 rebalance 和 manager 两个线程，构造位置见 [eplb_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/eplb/eplb_manager.cpp:74)。并发测试需要覆盖队列、ready layer和析构 join，不只是算法输出。

### 27.6 数值验收层级

```text
E2a router: ids/weights exact or tolerance
E2b expert: each expert output vs dense reference
E2c combine: hidden tolerance
E4 layer: single-rank vs EP hidden/logits
E4 update: old placement vs new placement hidden/logits
E5 service: fixed seed token + throughput/P99
```

只比较最终 token可能掩盖 router 和 expert 的误差抵消；逐层证据必须保留首个分叉位置。

### 27.7 性能实验的控制变量

比较 EP degree 时固定 token batch、top-k、模型 dtype、专家权重格式和节点拓扑。比较 EPLB 时固定同一份 route trace，避免流量随机变化比策略本身更大。至少报告：

| 变量 | 基线 | 实验 | 结论所需指标 |
|---|---|---|---|
| EP | EP1 | EP2/4/8 | HBM、A2A、layer latency |
| 冗余 | 0 | 1/2/... | max/mean、extra HBM、迁移摊销 |
| interval | 大 | 小 | 响应速度、更新次数、P99 |
| threshold | 高 | 低 | 触发率、抖动、层尾时间 |
| fused path | off | on | 命中率、workspace、数值误差 |

### 27.8 线上故障定位

MoE 层卡住时先比较所有 rank 的 send counts；输出大错时先比较 top-k 和 permutation；只有某些请求错时检查动态 shape、empty rank 和 mapping版本；更新瞬间 OOM时检查新旧权重双持、staging 与 KV 峰值；P99 周期性尖峰时把周期与 EPLB interval 对齐。

不要先把问题归因于“网络抖动”。AllToAll 负载不均、某个 expert大 GEMM、权重 prepare争用 HBM 都可能表现为通信尾延迟。

### 27.9 源码阅读中的对象所有权

| 对象 | 生产者 | 消费者 | 生命周期 |
|---|---|---|---|
| topk ids/weights | Router | Dispatch/Combine | 单层 forward |
| counts/assist info | Dispatch | Grouped GEMM/Combine | 单层 forward |
| expert load | MoE Worker | Engine/EplbManager | step/window |
| expert distribution | EplbPolicy | Engine/Worker/Model | mapping版本 |
| prepared layer id | EplbExecutor | Engine/EplbManager | 一次更新握手 |
| expert weight buffer | loader/EPLB | expert kernel | 模型/更新生命周期 |

EPLB info 作为 forward input的一部分跨 Engine/Worker 传递，字段序列化入口见 [forward_shared_memory_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/forward_shared_memory_manager.cpp:832)。这也是 Graph/多进程路径必须保持版本一致的证据。

### 27.10 Prefill 与 Decode 差异

Prefill token 多，Grouped GEMM 更容易形成高效大 M，但 dispatch bytes也大；Decode 每请求通常一个 token，动态 batch汇成的 expert M 仍可能很小，AllToAll 启动延迟和热点波动更突出。

因此 EPLB 的统计窗口不能只用 token总数解释。相同 expert load 在 Prefill 大块和 Decode 多步中的时间分布不同，对 TTFT、TPOT 和迁移可摊销窗口的影响也不同。

### 27.11 Scheduler 与 MoE 的边界

Scheduler 通常不知道每层将路由到哪些 expert，它只决定本轮 token和 sequence。MoE 热点在 model forward后才被观测，再反馈到未来 EPLB mapping。这是滞后控制，而不是 Scheduler 能提前保证的均衡。

动态 batch组成会改变 expert分布，所以调度策略、Chunked Prefill 和 EPLB 必须用同一真实流量联合测试；不能把离线均匀 token benchmark 的结论直接套到在线服务。

## 28. Kimi K3 当前边界

Kimi K3 五层前缀含一层 Dense 和四层 MoE，但现有阶段性 TP2 对齐的首个偏差出现在第 1 层 `KDA + MoE` 输出，尚未补齐层内 Attention/MoE 边界，因此不能把偏差归因锁死在 expert kernel。当前基础 `DP2+TP2` 也不等于 MoE DP+EP 已验证。

下一步应固定 top-k ids/weights，分别 dump dispatch 前 hidden、expert counts、SiTU 两段输出和 combine hidden，再判断偏差来自 router、量化 expert、通信还是累加顺序。

## 29. 常见误区与阅读顺序

误区：MoE 稀疏就自动省权重；EP 只通信一次；EPLB 会改变逻辑 top-k；有后台线程就无迁移尖峰；平均 expert load相同就没有尾部；请求返回就证明 token未错配。

阅读顺序：ModelArgs/router -> FusedMoE top-k -> local permutation/GEMM/combine -> EP dispatch/combine -> Engine load汇总 -> Manager -> Policy -> Executor prepare -> 下一轮 mapping。

## 30. 自测题

1. 为什么 payload 近似含 K 倍 token？
2. 为什么层延迟由最热 rank 而非平均 load 决定？
3. EPLB threshold 当前衡量的具体量是什么？
4. 新 mapping 为什么必须晚于所有目标权重 ready？
5. 冗余 expert 怎样同时影响 HBM 和 dispatch matrix？
6. Kimi K3 当前证据为什么不能宣称 EPLB 已支持？

## 31. 结论

MoE 的“稀疏”只减少被激活计算，不自动解决权重与通信。xLLM 的完整主链是 router top-k、token 重排、EP dispatch、Grouped GEMM、combine；EPLB 则在这条链上用在线负载、冗余副本和异步权重准备治理长期热点。所有收益都必须同时支付 AllToAll、token buffer、冗余权重和更新一致性成本。
