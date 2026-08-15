# xLLM PD 分离：Prefill/Decode 解耦、KV 传输与服务调度全链路

## 1. 为什么拆 P 和 D

Prefill 是大矩阵、计算/带宽密集，决定 TTFT；Decode 是小步反复、延迟和 KV 容量敏感，决定 TPOT。混在同一 batch 时，长 Prefill 会干扰 Decode。

PD 分离把两类工作放到独立实例池，让它们使用不同并行度、batch 策略和扩缩容比例。代价是 Prefill 生成的 KV/线性状态必须跨实例交给 Decode。

## 2. 源码地图

- `core/scheduler/disagg_pd_scheduler.*`：请求分发、P/D 队列、首 token 交接、容量控制。
- `core/distributed_runtime/disagg_pd_service*`：BRPC 控制面服务。
- `core/framework/kv_cache_transfer/*`：PUSH/PULL、transfer group、拓扑与状态。
- `proto/disagg_pd.proto`：RPC 消息。
- `core/framework/config/disagg_pd_config.*`：role、mode/type、heterogeneous PD、parallel shard pull。
- `xservice_client`：实例注册、发现和调度协作。

## 3. 两条数据流

控制面传：请求、实例信息、block ids、地址/端口、状态、错误和首 token。

数据面传：各层 KV blocks，混合模型还可能有 linear state/indexer 等。二者不能混淆：RPC 成功只说明控制消息到达，不说明设备 KV 已完成传输。

## 4. 请求完整时序

```text
Gateway/Master 选择 P 与 D
  -> 请求发给 P
  -> P 分配 blocks 并执行 prefill
  -> D 预留自己的目标 blocks
  -> P/D 交换 transfer metadata
  -> PUSH: P 推 KV；或 PULL: D 拉 KV
  -> transfer completion
  -> P 发送首 token/完成通知
  -> D 恢复 sequence committed 状态
  -> D 连续 decode 并流式返回
  -> 两侧按所有权释放 blocks
```

Chunked Prefill 下，不能依赖单一 `last_step_prefill` 标记；源码的 `step()` 每步检查首 generation token，内部再用生成数判断是否真正发送。

## 5. D 端 Prefix Cache 命中

D 端可能已经缓存 prompt 前缀。此时无需从 P 重传共享部分。源码在响应中携带 shared block count，并把各 transfer group cursor 推进到 D 端共享位置，再从第一个缺失 block 开始传。

这是节省网络的关键，但要求 P/D 对 block size、模型语义和分组布局一致。

## 6. PUSH 与 PULL

- PUSH：P 获得 D 目标地址后主动发送；适合 P 掌握生产进度。
- PULL：D 在目标 block 就绪后发起读取；控制权在消费端。

源码还处理 heterogeneous TP 和 grouped KV transfer。异构 P/D 并行度下，一个逻辑 KV 可能需要重新分片，不能简单复制同 rank 同 block id。

## 7. 通信量

每请求传输量近似 prompt 已生成 KV：

```text
bytes ≈ transferred_tokens × KV_bytes_per_token
```

MLA/GQA/量化可降低该值；Prefix hit 可减少 transferred_tokens。TP/kv_split 改变每 rank shard，但全局总字节由模型缓存表示决定，另有协议和对齐开销。

## 8. TTFT 归因

源码注释明确 P->D transfer 成本归入 TTFT，而不是后续 ITL。端到端：

```text
TTFT ≈ queue_P + prefill + allocate_D + KV_transfer + handoff
```

PD 能减少 decode 干扰，却不保证 TTFT 更低；短 prompt 时 transfer/调度固定成本可能超过隔离收益。

## 9. Overlap

理想情况下按 layer/block 流水：P 产生一部分 KV 后立即传，D 预分配并接收，最后边界到齐后开始 decode。是否支持取决于 transfer mode 和模型状态完整性。

可隐藏部分包括实例选择、D block 预留、metadata RPC、已完成层的传输。首个 Decode Attention 前所需的全部 KV 是硬依赖，无法凭异步 API消失。

## 10. 内存所有权

交接期间 P 和 D 可能同时持有 KV：

- P 在确认传输完成前不能释放源 block。
- D 在 transfer 完成前不能让 Attention 读取目标 block。
- 错误/超时必须两侧幂等回收。
- shared prefix block 的 refcount 与新分配 transfer block 分开管理。

因此 PD 峰值不是单侧 KV，而是系统级短时双份 KV 加传输 buffer。

## 11. 精度影响

同 dtype、正确重分片的 KV 传输不改变计算。风险来自层/头/shard 映射、block offset、量化 scale、线性状态和最后 prefill token 提交错误。

异构 TP 测试应覆盖非 MLA 与 MLA、linear attention、prefix hit、chunked prefill，不能只用普通 MHA 单请求验证。

## 12. 故障语义

必须考虑：P 失败、D 失败、RPC 超时但数据已到、重复通知、部分 block 到达、请求取消、实例信息过期。理想协议用 request/transfer id 保证幂等，并把资源回收与最终状态绑定。

## 13. 调度与扩缩容

`DisaggPDScheduler` 可注册实例的 TTFT/TPOT profiling data、KV capacity 和并行信息。路由不能只按队列长度：长 prompt 应看 P 预计时间，长生成应看 D KV 容量与 decode batch 状态。

P:D 的合理比例随 prompt/decode 比、模型 KV 大小和网络带宽变化。

## 14. 指标体系

- P queue/prefill、D queue/decode 分段延迟。
- KV transfer bytes/bandwidth/P99/失败重试。
- allocate-D 到 transfer-ready 的时间。
- D prefix shared blocks/tokens。
- P/D KV utilization、双份驻留时长。
- TTFT 与 TPOT 分位数。
- 每实例选中率和负载误差。

## 15. 与其他特性的边界

- Chunked Prefill：P 可分块算，但交接完成判断更复杂。
- Prefix Cache：P 减少计算，D 减少传输，两侧命中语义不同。
- KV quant：当前 worker 对 KV INT8 与 PD 有显式不支持限制。
- Speculative：通常主要发生在 D，候选状态不得污染交接前缀。
- Host/global KV：可能作为第三数据源，增加一致性状态。

## 16. 结论

PD 分离不是把两个进程启动起来，而是一个分布式 KV 所有权协议。真正的收益来自隔离两种完全不同的负载；真正的难点是 block 分配、传输完成、sequence 状态和异常回收在 P/D 两侧始终一致。

