# xLLM 异步调度与 Schedule Overlap 完整链路：从假 token 到设备事件

> 本文是 xLLM 知识库第 09 篇。目标不是把 `enable_schedule_overlap` 翻译成“异步执行”，而是回答：谁提前了哪一步、哪一个结果还是假的、真实结果在哪里提交、哪些内存必须保留、哪条通信会把重叠窗口重新堵住，以及当前源码对不同后端和模型到底支持到什么程度。

本文的源码基线为当前工作区 `xllm/`。代码链接均指向仓库内文件和行号；行号随后续提交变化时，应以附近函数名复核。

## 1. 先给结论

### 1.1 这不是“两条线程同时跑模型”

xLLM 的核心实现是一个深度为 1 的 producer-consumer 流水：

1. `ContinuousScheduler::step_with_schedule_overlap()` 先调度当前 batch。
2. `LLMEngine::step()` 把输入发到 Worker；WorkerService 立即返回一个负数假 token。
3. Scheduler 用假 token 把 batch 状态推进一拍，并保存为 `last_batch_`。
4. 下一次 `step()` 开始时，设备仍可能在计算上一拍；Worker 的单线程池按 FIFO 排队下一次 forward。
5. 下一拍真正需要上一拍 token 时，Worker 用 `get_last_step_result()` 取回真实输出，再把假 token 替换掉。

因此，源码中的“异步”包括三个不同层次：

| 层次 | 代码事实 | 是否等价于设备并行 |
| --- | --- | --- |
| RPC 异步 | `step_remote_async()` 返回 future | 否，后续 `collectAll().get()` 仍会等待 RPC |
| 调度重叠 | Engine 先拿假结果，Scheduler 继续建下一拍 | 是，前提是 Worker 内部 future 在设备计算完成前已返回 |
| 设备流重叠 | prepare stream、compute stream、`ready_event`、Graph 双缓冲 | 只在满足后端/Graph 条件时成立 |

把三者混在一起，会误判性能和正确性。

### 1.2 默认值必须从最终入口判断

底层 `runtime::Options` 和 `ContinuousScheduler::Options` 的属性默认值是 `true`，但公共配置 `SchedulerConfig`、Python API 和 CLI 默认是 `false`。最终服务通常由公共配置覆盖底层 Options，因此用户看到的默认行为是关闭。证据如下：

- [scheduler_config.h](../xllm/xllm/core/framework/config/scheduler_config.h#L43-L70)
- [scheduler_config.cpp](../xllm/xllm/core/framework/config/scheduler_config.cpp#L20-L31)
- [llm.py](../xllm/xllm/pybind/llm.py#L115-L128)
- [args.py](../xllm/xllm/pybind/args.py#L55-L68)
- [runtime/options.h](../xllm/xllm/core/runtime/options.h#L130-L143)

结论标签：`框架默认关闭；显式打开后进入能力门控`。

### 1.3 当前 Kimi K3 的结论

Kimi K3 是混合全注意力和 KDA/线性注意力的模型。Worker 侧已有线性状态恢复、`ready_event` 和 overlap 输入替换代码，因此属于“框架侧代码可达、需要条件验证”。本知识库已经完成 Kimi K3 TP2 五层 prefill 对齐，但该实验明确关闭了 schedule overlap；不能把 prefill 对齐结果写成 overlap 已验证。

结论标签：`Kimi K3：框架侧条件支持；本地请求级 overlap 尚未验证`。

## 2. 从问题出发：同步路径为什么有空洞

### 2.1 同步单步模型

同步路径的循环近似为：

~~~text
设备 forward(N)
    -> RPC 返回真实 token
    -> CPU 更新 Sequence/KV/停止条件
    -> Scheduler 选择 N+1
    -> 设备 forward(N+1)
~~~

如果设备解码耗时为 `T_device`，CPU 调度和输出处理为 `T_host`，单请求的理想周期接近 `T_device + T_host`。当 `T_device` 很短、batch 很小，host gap 会直接表现为 TPOT 抖动。

### 2.2 Overlap 的目标

overlap 不减少 FLOPs、KV 字节数和 TP/EP 集合通信量。它试图把 `T_host` 放入上一拍设备执行窗口：

~~~text
CPU:  schedule(N) -> schedule(N+1) -> commit(N) -> schedule(N+2)
NPU:               forward(N)   -> forward(N+1)
~~~

稳态周期的上限更接近 `max(T_device, T_host)`，但首拍填充、末拍排空、通信和真实 token 提交仍然存在。

### 2.3 为什么不能直接读取下一 token

Decode 的输入 token 来自上一轮采样，而上一轮采样尚未完成。若把未定义值写入下一轮，模型会读错 token；若 host 强行等待，又失去 overlap。因此 xLLM 使用一个可识别的负数占位符，并把“计划状态”和“提交状态”分开。

## 3. 配置与能力门控

### 3.1 配置值的传播链

典型传播链为：

~~~text
CLI / JSON / Python
  -> SchedulerConfig.enable_schedule_overlap
  -> xllm.cpp / Master 结合后端、角色、服务类型修正
  -> runtime::Options.enable_schedule_overlap_
  -> ContinuousScheduler / Engine / Worker / SequenceParams
~~~

配置合并和初始化见 [scheduler_config.cpp](../xllm/xllm/core/framework/config/scheduler_config.cpp#L75-L130)。Python LLM API 将参数写入 Options 的位置见 [llm.py](../xllm/xllm/pybind/llm.py#L185-L201)。

### 3.2 强制关闭矩阵

源码中以下路径会关闭或拒绝 overlap：

| 场景 | 代码行为 | 解释 |
| --- | --- | --- |
| MLU VLM | `xllm.cpp` 强制 `false` | 当前 VLM 组合没有走该协议 |
| DCU PD | `xllm.cpp` 强制 `false` | 设备侧 PD 路径未完成条件适配 |
| PD Prefill 角色 | `Master` 强制 `false` | Prefill 需要先产出并传输首 token/KV |
| embedding/mm_embed | `Master` 强制 `false` | 非自回归 decode，不适合负数 token 协议 |
| REC 模型 | `Master` 强制 `false` | 推荐模型调度器/输出语义不同 |
| `/v1/sample` | 返回 `UNAVAILABLE` | 服务接口明确拒绝 overlap |
| FixedStepsScheduler | 打印 not supported | 只支持同步 step |
| 多流并行实验开关 | `LOG(FATAL)` | 与 schedule overlap 是不同实验，当前未实现 |

对应证据：[xllm.cpp](../xllm/xllm/xllm.cpp#L320-L365)、[master.cpp](../xllm/xllm/core/distributed_runtime/master.cpp#L350-L380)、[master.cpp](../xllm/xllm/core/distributed_runtime/master.cpp#L490-L515)、[master.cpp](../xllm/xllm/core/distributed_runtime/master.cpp#L560-L580)、[sample_service_impl.cpp](../xllm/xllm/api_service/sample_service_impl.cpp#L145-L165)、[fixed_steps_scheduler.cpp](../xllm/xllm/core/scheduler/fixed_steps_scheduler.cpp#L405-L418)。

### 3.3 不要把文档示例当成当前支持矩阵

官方异步调度文档给出过约 200 微秒设备间隙和 17% 吞吐提升的案例，但没有给出硬件、提交版本、并发、prompt/生成长度和可复现实验脚本。本文把它作为历史性能叙述，不作为本地性能证据。见 [async_schedule.md](../xllm/docs/src/content/docs/zh/features/async_schedule.md#L15-L65)。

## 4. Scheduler：一拍超前的真正边界

### 4.1 初始化两个 batch 槽

`ContinuousScheduler` 构造时按 DP 数量初始化 `last_batch_`，并把 overlap 传给 ProfileManager 和 AsyncResponseProcessor；开启 overlap 时还把所需 speculative token 预算乘二。见 [continuous_scheduler.cpp](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L100-L175)。

为什么是两个槽而不是一个？

- `batch`：当前刚调度、设备正在处理或刚发出的 batch。
- `last_batch_`：上一拍已经用假 token 推进、现在等待真实输出提交的 batch。

这个深度为一的设计限制了额外 in-flight 计算，换来资源账本和序列状态比较容易推理。

### 4.2 非 overlap 分支

[ContinuousScheduler::step](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L345-L375) 在关闭时执行：

1. `schedule_request()` 选请求和 KV block。
2. `engine_->step(batch)` 等待真实 forward output。
3. `process_batch_output(false)` 直接提交真实 token。

PD OOC 会在第 2 步进入自己的迁移逻辑；这条路径没有假 token。

### 4.3 overlap 分支逐行解释

[step_with_schedule_overlap](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L377-L405) 的顺序是：

~~~cpp
auto batch = schedule_request(timeout);
if (!cur_batch_all_empty) {
  engine_->step(batch);
}
if (!is_first_step_ && !last_batch_all_empty) {
  engine_->update_last_step_result(last_batch_);
  process_batch_output(true);
}
last_batch_ = std::move(batch);
last_running_sequences_ = running_sequences_;
last_running_requests_ = running_requests_;
is_first_step_ = false;
~~~

关键点是“先提交当前 batch 的假结果，再提交上一拍的真实结果”。Scheduler 线程每轮最多提前一拍，不会无限制堆积 forward。

### 4.4 空 batch 不是无害的空操作

只有当前和上一拍都为空时才直接 return。若当前为空但上一拍不为空，仍然要调用 `update_last_step_result(last_batch_)`，否则上一拍的 fake token 永远不会被替换，finished、KV 和流式输出都不会收敛。

### 4.5 running 指针为什么要复制

`process_batch_output(true)` 使用 `last_running_sequences_` 和 `last_running_requests_`，而不是当前 `running_*`。在上一拍到当前拍之间，请求可能结束、取消、被 beam search 替换 Sequence 对象；函数开始会再次从 Request 刷新指针。见 [process_batch_output](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L478-L530)。

### 4.6 取消与结束的延迟

overlap 下，已取消或已完成请求可能仍位于 `last_batch_`。Scheduler 不会在收到 fake 输出的那一轮立即释放所有资源，而是在真实结果提交时过滤它们；最后一个 token 通过 `handle_last_token()` 补处理。这样可以避免设备仍引用 KV block 时提前 deallocate。

### 4.7 pause 是一个排空协议

WAIT 模式会把 `last_batch_` 是否在飞纳入判断；KEEP/ABORT 模式会先调用 `update_last_step_result(last_batch_)`，再清空流水状态，之后才释放或抢占请求。见 [try_complete_pause](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L648-L715)。

如果顺序反过来，设备还没完成读写就释放 KV block，恢复时会得到随机输出。这是调度暂停中最容易被忽略的生命周期约束。

## 5. Engine：为什么 `collectAll` 仍然不等于同步

### 5.1 Engine 的 RPC 聚合

[LLMEngine::step](../xllm/xllm/core/distributed_runtime/llm_engine.cpp#L1032-L1105) 为每个 Worker 调用 `step_remote_async()`，随后 `folly::collectAll(futures).get()`。表面上它仍等待所有 future，但等待的是 WorkerService 的 RPC 返回，不一定是设备完成。

### 5.2 WorkerService 返回的到底是什么

非 overlap 时，WorkerService 对 `worker_->step_async()` 调 `.get()`，把真实输出拷到 CPU，并同步服务 stream。overlap 时，driver worker 构造：

~~~cpp
next_tokens = torch::arange(-1, -(num_decode_seqs + 1), -1,
                            torch::TensorOptions().device(torch::kCPU));
~~~

然后丢弃 future 的返回值，不等待设备。见 [WorkerService::step](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L150-L305)。

所以 Engine 的 `collectAll` 很快完成，Scheduler 可以推进；真正 output 由下一轮 `get_last_step_result` 获取。

### 5.3 假 token 的编码

第 0 个 decode 行用 `-1`，第 1 行用 `-2`，以此类推。Worker 之后通过 `-fake_id - 1` 找回对应的真实输出行。这个编码同时解决了 batch 内多序列映射问题，不需要额外的 Python 字典。

### 5.4 Engine 如何分两次提交

当前拍：[LLMEngine::step](../xllm/xllm/core/distributed_runtime/llm_engine.cpp#L1060-L1102) 调 `batch.process_sample_output(result, false)`；`false` 的含义是“追加 fake token”。

上一拍：[LLMEngine::update_last_step_result](../xllm/xllm/core/distributed_runtime/llm_engine.cpp#L1108-L1155) 从 driver Worker 取真实结果，并调 `process_sample_output(raw, options_.enable_schedule_overlap())`；开启 overlap 时参数为 `true`，表示替换 fake。

### 5.5 EPLB 的特殊等待

EPLB 需要所有 expert worker 的 load 数据，因此 `update_last_step_result` 不能只取 driver；而 `step` 在 overlap 开启时跳过即时 EPLB 处理。这样做是为了让 expert load 与真实 output 在同一个 committed 边界处理，避免假结果污染负载统计。

## 6. WorkerService：RPC 之外的线程和事件

### 6.1 Host view 必须 clone

overlap 开启时，WorkerService 会复制 `token_ids_host`、`positions_host` 和 host block table。原因是 Scheduler 下一拍可能重写共享的 CPU view，而设备 prepare stream 尚未读取完上一拍的 view。见 [stabilize_schedule_overlap_host_views](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L100-L130)。

这几个 clone 是小的元数据张量，却是 correctness 必需的；不能仅凭“tensor 在 CPU”就认为可以复用。

### 6.2 SHM polling 与 Graph

共享内存 polling 线程在 `overlap + graph` 时选择 `DEFER_TO_WORKER_PREPARE`，不在 polling 线程直接 H2D。源码注释说明 NPU Graph task update 与 SHM 线程 H2D 同时入队会不安全；输入物化被延迟到 Worker 有序 prepare stream。见 [create_polling_shm_thread](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L315-L330)。

### 6.3 真实结果的 RPC

`GetLastStepResult` 调 Worker 的 `get_last_step_result_async()` 并等待；overlap 时先让服务 stream 等待 `ForwardOutput.ready_event`，再进行非阻塞 D2H 拷贝，最后同步服务 stream。见 [GetLastStepResult](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L820-L875)。

这说明 `ready_event` 的作用不是通知 Scheduler，而是把“设备输出已可被服务 stream 读取”传递到输出拷贝边界。

## 7. WorkerImpl：单线程 FIFO 保证状态顺序

### 7.1 为什么 threadpool 只有一个工作线程

WorkerImpl 头文件明确说明，overlap 下会有两个 step task 排队；线性注意力状态的 restore/save 和 compute stream 的入队顺序依赖 FIFO。见 [worker_impl.h](../xllm/xllm/core/runtime/worker_impl.h#L300-L325)。

多个线程虽然能提高 host 并行，却可能让下一拍 restore 先于上一拍 state write 入队，产生 silent corruption。

### 7.2 `step_async` 的真实流程

[WorkerImpl::step_async](../xllm/xllm/core/runtime/worker_impl.cpp#L1044-L1095) 分为两段：

1. 调用 `prepare_work_before_execute()`，在 prepare stream 准备设备输入。
2. 把已准备的 `ForwardInput` 投递到单线程 `threadpool_`。

工作线程中：

- 如果已有上一拍真实结果且当前 batch 含 decode，就调用 `update_input_by_last_step_output_for_schedule_overlap()` 替换负数 token。
- 调 `step_for_schedule_overlap()`，通常走 NO_SYNC。
- 把 output 放到一格 handoff 槽位；driver/EPLB 会等待消费者取走后才覆盖。
- 立即 resolve Worker future，使 WorkerService 可以返回 fake output。

### 7.3 handoff 槽位是单元素队列

`last_step_output_`、`last_step_output_valid_`、`is_recorded_` 由 mutex/condition variable 保护。driver 或 EPLB 需要结果时，`get_last_step_result()` 等待 `is_recorded_`，取走后清除并通知生产者。见 [worker_impl.cpp](../xllm/xllm/core/runtime/worker_impl.cpp#L1098-L1108)。

这不是无限缓存：如果 Scheduler 不调用 `update_last_step_result`，后续 Worker 可能在记录槽位处阻塞。也正因为槽位只有一个，Scheduler 的“最多提前一拍”不是风格选择，而是协议要求。

## 8. 输入替换：从负数到真实 token

### 8.1 CUDA/MUSA/通用 Torch 路径

通用路径在设备 token tensor 上构造负数 mask：

~~~cpp
auto neg_mask = (flatten_tokens < 0);
auto index = torch::clamp(-flatten_tokens, 0) - 1;
auto replacement = last_output.index({index});
inputs.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
~~~

MUSA 需要先把 index 转到 CPU 再回 MUSA；这是额外的设备/主机转换边界。证据见 [update_input_by_last_step_output](../xllm/xllm/core/runtime/worker_impl.cpp#L640-L676)。

### 8.2 NPU eager 路径

NPU 使用 `npu::replace_token`。非 Graph decode 会同步当前 stream，确保替换完成后 forward 读取；这会缩短 overlap 窗口，但保证输入可见。

### 8.3 NPU Graph 路径

满足 `can_prepare_npu_graph_decode_input()` 时，替换不立即同步，并把 tensor 写入 `input_tokens_override`。Graph replay 会消费这个 override；配合双 buffer，可把下一槽输入准备放到上一槽计算旁边。该条件要求 NPU、非 speculative、启用 Graph 和 Graph double buffer、backend 为 `llm` 且 batch 含 decode。见 [worker_impl.cpp](../xllm/xllm/core/runtime/worker_impl.cpp#L580-L620)。

## 9. LLMWorkerImpl：线性注意力的额外顺序

### 9.1 KDA/GDN 不只保存 KV cache

混合模型除了 full-attention KV block，还保存 recurrent/linear state。下一次 decode 必须先恢复属于该 sequence 的 checkpoint，再执行 KDA 更新；这使得“host 先准备、设备后计算”的普通顺序不够。

### 9.2 restore 放到 compute stream

overlap 下，`prepare_work_before_execute_on_stream` 不在 prepare stream 立即 restore linear state；[LLMWorkerImpl::step_for_schedule_overlap](../xllm/xllm/core/runtime/llm_worker_impl.cpp#L210-L245) 在 Worker 单线程中切到 compute stream，再 restore。这样上一拍写状态的 kernel 已经按 FIFO 入队，restore copy 自然排在其后，不需要额外跨流同步。

### 9.3 NO_SYNC 仍有一个不能省的等待

`step_internal(..., NO_SYNC)` 不做默认流同步，但会执行 `wait_kv_push()`，并把输入放进 `retained_input` 以延长 tensor 生命周期；如果启用 overlap 且需要记录 handoff，还会记录 `ready_event`。见 [llm_worker_impl.cpp](../xllm/xllm/core/runtime/llm_worker_impl.cpp#L377-L404)。

因此，NO_SYNC 不是“什么都不等”：KV push、输入生命周期和输出事件仍有硬约束。

### 9.4 event 的两端

`ForwardOutput.ready_event` 在 compute stream 上记录；读取端是 WorkerService 的 output copy stream。输入侧则有 `ForwardInput.metadata_ready_event`，prepare stream 完成元数据后由 compute stream 等待。两类 event 分别保护 output handoff 和 input materialization，不能互换。

## 10. ACL Graph 双缓冲：第二层重叠

### 10.1 两个 GraphSlot

`AclGraphExecutorImpl` 在 `enable_graph_double_buffer` 时创建两个 `GraphSlot`，每个槽维护自己的 persistent params 和按 bucket/key 缓存的 `AclGraph`。见 [acl_graph_executor_impl.cpp](../xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L380-L400) 与 [acl_graph_executor_impl.h](../xllm/xllm/core/runtime/acl_graph_executor_impl.h#L155-L170)。

### 10.2 replay 槽轮转

每次 `run()` 在 mutex 下取 `next_replay_slot_`，轮转索引并记录 `last_started_replay_slot_`；找到对应 graph 就 replay，否则 lazy capture。见 [acl_graph_executor_impl.cpp](../xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L530-L590)。

### 10.3 prepare 使用另一槽

`prepare_graph_input()` 只处理 decode 或 speculative verify 的特定 chunked-prefill，检查 DP token 数、最大序列长度和双缓冲是否可用，然后选择 `(last_started_replay_slot_ + 1) % graph_slot_count_`。如果目标槽已 prepared 或 graph key 不存在，就放弃这次提前准备。见 [prepare_graph_input](../xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L622-L690)。

### 10.4 为什么是“条件重叠”

Graph 双缓冲只有在以下条件同时成立时才扩大窗口：

- NPU、Graph 已启用、double buffer 已启用。
- backend 为 `llm`。
- 非 speculative decode。
- 当前 batch 有 decode，且 DP 各 shard 都满足 decode 形态。
- Graph bucket 已捕获，`kv_max_seq_len` 不超模型上限。
- KV transfer 不是会强制同步的 PUSH 路径。

任意条件不满足，代码会回退到 prepare 等待 compute stream，或直接 eager forward。

### 10.5 Graph persistent param 的 token 更新

Graph persistent param 更新接口提供 `skip_token_update`；overlap + replay 时，真实 token 可能尚未从上一拍 handoff 到达，因此可以先更新其它元数据，token copy 延迟到 replay。见 [acl_graph_persistent_param.cpp](../xllm/xllm/core/runtime/acl_graph_persistent_param.cpp#L615-L665)。

## 11. 输入准备与 host/device 事件

### 11.1 prepare stream 的职责

[prepare_work_before_execute_on_stream](../xllm/xllm/core/runtime/worker_impl.cpp#L750-L916) 在 prepare stream 上完成：

- CPU 到 device 的 input tensor 物化。
- KV block swap/copy。
- DP/EP padding 与 expert 元数据。
- CP plan prepare。
- MLA prefix-cache 输入修正。
- 满足条件时 Graph input prepare。

### 11.2 默认等待关系

overlap 开启且不能使用 Graph 独立槽时，prepare stream 会 `wait_stream(compute_stream_)`，避免写共享输入或线性状态时覆盖正在使用的数据。无论是否等待 stream，都要先等待 `metadata_ready_event`。

### 11.3 Graph double buffer 的例外

Graph double buffer 允许准备下一槽，而上一槽仍在 compute stream 执行。`can_prepare_without_compute_stream_wait()` 返回 true 时才走这个例外；它不会取消 Graph capture lock，也不会取消输入元数据 event。

## 12. Batch 与 Sequence：假 token 的两阶段提交

### 12.1 output_targets 冻结映射

Batch 构建 forward input 时生成 `output_targets_`，记录每个输出行对应哪个 Sequence 或 sample slot。Engine 后续两次处理必须复用这个映射，直到真实替换完成。

### 12.2 `process_sample_output(false)`

当前拍 fake output 进入 [Batch::process_sample_output](../xllm/xllm/core/framework/batch/batch.cpp#L630-L670)，最终调用 `append_token_for_sequence(..., false)`。Sequence 会把负数写入 tokens 并推进 `num_tokens_`，但不会增加真实生成 token 计数，也不会更新 logprob/unique-token 统计。

### 12.3 `process_sample_output(true)`

真实 output 到达后，Batch 调 `update_last_step_token(token, token_idx)`，覆盖 fake 位置；成功后清空 `output_targets_`，并允许处理 beam search。见 [batch.cpp](../xllm/xllm/core/framework/batch/batch.cpp#L660-L715)。

### 12.4 Sequence 的两个游标

`num_tokens_` 表示数组中已经占用的逻辑位置，`cur_generated_token_idx_` 表示下一个真实 token 应提交的位置。fake append 只推进前者；real update 同时写 token、更新 KV token 数、无效化 block/linear hash、更新 logprob，并推进后者。见 [sequence.cpp](../xllm/xllm/core/framework/request/sequence.cpp#L343-L440)。

### 12.5 负数 token 为什么不结束请求

负数 append 会把 `finish_status_invalidated_` 设为 true 后直接返回，因此 EOS、stop string 和 max-token 判断不会被 fake 误触发。流式输出生成时还会从后向前跳过负数 token。见 [sequence.cpp](../xllm/xllm/core/framework/request/sequence.cpp#L500-L520)。

### 12.6 统计口径必须扣除额外一拍

overlap 会让 `num_generated_tokens()` 包含一个尚未提交的 fake 位置，因此 Request 日志和 usage 分别减一。见 [request.cpp](../xllm/xllm/core/framework/request/request.cpp#L75-L95) 与 [request.cpp](../xllm/xllm/core/framework/request/request.cpp#L155-L175)。

## 13. Chunked Prefill 与 overlap 的交叉状态机

### 13.1 为什么需要额外队列

Chunked prefill 的一个 sequence 可能连续经历“prefill chunk N、prefill chunk N+1、decode”。overlap 的 fake/real 两次回调与 chunk stage 的时间并不相同，Batch 因此在 Sequence 上维护最多两个元素的 `pre_scheduled_step_prefill_queue`。

### 13.2 入队和出队

在 `replace_fake_token=false` 的阶段，若当前是 chunked-prefill stage，就压入 `true`；普通 decode fake 则压入 `false`。在真实替换阶段，若队首标记是 prefill，则只弹队列，不把 prefill 当作 token 替换。见 [batch.cpp](../xllm/xllm/core/framework/batch/batch.cpp#L671-L713)。

### 13.3 队列上限为 2 的意义

Scheduler 只提前一拍，所以队列不需要无限增长。超过 2 就弹旧状态，既防止内存增长，也把代码约束成“当前计划 + 上一拍提交”的模型。

### 13.4 Chunked prefill 的 overlap 风险

真正需要核对的是：chunk N 写入的 KV/linear state 是否在 chunk N+1 的 prepare/compute 之前可见；NPU KDA 路径通过单线程 compute stream restore 保证顺序。若换后端或自定义 model 没有同等 event/stream 约束，应该先关闭 overlap。

## 14. 投机解码：外层 overlap，内层同步

### 14.1 为什么不能嵌套两套 fake 协议

外层 schedule overlap 已经用负数 token 表示“上一拍真实 token 未提交”；投机解码内部还会产生 draft、accepted、bonus token。如果 target 和 draft worker 都各自维护 handoff 槽，接受长度和 KV 状态会出现两套提交时钟。

### 14.2 内部 Worker 明确关闭

MTP/Speculative Worker 构造 target/draft options 时设置 `enable_schedule_overlap(false)`，DFlash、Eagle、Suffix worker 也关闭内部选项。证据：[mtp_worker_impl.cpp](../xllm/xllm/core/runtime/mtp_worker_impl.cpp#L400-L430)、[dflash_worker_impl.cpp](../xllm/xllm/core/runtime/dflash_worker_impl.cpp#L55-L85)、[eagle3_worker_impl.cpp](../xllm/xllm/core/runtime/eagle3_worker_impl.cpp#L20-L45)、[suffix_worker_impl.cpp](../xllm/xllm/core/runtime/suffix_worker_impl.cpp#L50-L70)。

### 14.3 外层如何承接结果

MTP 在 overlap 下保留 target output 的 `ready_event` 和 retained input dependencies；接受 token 写入在同一 compute stream 完成，下一轮才由外层协议提交。内部代码会在非 overlap 时 flush pending target context，在 overlap 时把 handoff 留给外层。见 [mtp_worker_impl.cpp](../xllm/xllm/core/runtime/mtp_worker_impl.cpp#L1390-L1430)。

### 14.4 overlap 下的 speculative 预算

ContinuousScheduler 初始化时将 `min_speculative_tokens_required_` 设为 `num_speculative_tokens * 2`。这是为了同时容纳当前拍 fake/真实提交与投机额外 token 的空间需求，不是把模型的 speculative length 变成两倍。

### 14.5 验证重点

投机组合必须检查 accepted token 数、拒绝 token 数、bonus token、preempt 后无 KV block 的序列，以及真实输出替换后 block hash 是否失效。仓库已有对应 Batch 测试样例，见 [batch_test.cpp](../xllm/tests/core/framework/batch/batch_test.cpp#L1850-L2050)。

## 15. PD 分离：网络传输成为提交边界

### 15.1 Prefill 和 Decode 的不同职责

PD 分离中，Prefill 产生首 token、KV block 和可能的 linear state；Decode 收到迁移信息后再继续自回归。Prefill 角色被强制关闭 overlap，避免在 KV 尚未发出时提前调度 decode；Decode 端在支持的设备上仍可使用 overlap。

### 15.2 Decode 首 token 的 fake/real 处理

`DisaggPDScheduler` 收到 remote first token 后，如果 overlap 开启，会先 `append_token(-1)`，再 `update_last_step_token(first_token)`；PD OOC 迁移 token 也用同样模式。见 [disagg_pd_scheduler.cpp](../xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp#L950-L985) 与 [pd_ooc_scheduler.cpp](../xllm/xllm/core/scheduler/pd_ooc_scheduler.cpp#L1750-L1785)。

这不是为了网络异步，而是为了让 Decode 第一拍的状态形状与普通 overlap decode 一致。

### 15.3 KV transfer 的可隐藏程度

overlap 不会自动隐藏 P-D KV transfer：

- PULL 模式必须等远端 block 拉取完成后才能 decode。
- PUSH 模式可能让 NPU Graph decode 不能跳过同步。
- KDA/GDN 还需要传输并恢复 linear state slot。

因此 PD 的端到端周期应拆成 `prefill compute + network transfer + decode first step`，不能只看设备 forward timeline。

### 15.4 设备限制

MLU/DCU PD 路径在配置层关闭 overlap；NPU/其他后端的具体组合必须按 transfer mode、模型类型和 worker 实现验证。配置测试也会验证设备角色强制关闭，见 [disagg_pd_config_test.cpp](../xllm/tests/core/framework/config/disagg_pd_config_test.cpp#L30-L60)。

## 16. AsyncResponseProcessor：输出回调也有流水

### 16.1 三个线程池

构造函数创建：

- `response_threadpool_`：按 request 绑定线程处理流式/非流式输出。
- `rpc_threadpool_`：批量响应完成后统一调用 RPC callback。
- `generate_output_threadpool_`：tokenizer/文本生成的并行任务。

见 [async_response_processor.cpp](../xllm/xllm/core/scheduler/async_response_processor.cpp#L30-L60)。

### 16.2 为什么要按 request 固定线程

同一 request 的 streaming delta 必须保持顺序。Processor 第一次调度时记录 `response_thread_id`，之后使用 `schedule_with_tid`，避免多个 callback 交错。见 [process_stream_request](../xllm/xllm/core/scheduler/async_response_processor.cpp#L190-L250)。

### 16.3 批量 RPC 的 barrier

批量响应先把每个 Request 的输出放入 response threadpool，再由 `BlockingCounter` 等全部完成，最后在 rpc threadpool 调 `outputs_func`。这可以减少跨服务 RPC 次数，却会把最慢 request 变成批量回调的尾延迟。

### 16.4 Overlap 与流式输出

Scheduler 只把真实提交后的 request 放入 `process_stream_requests`。Processor 会跳过仍是负数占位的尾部 token；当 request finished 且最后 token 尚未处理时，先 `handle_last_token()` 再发送 delta。源码还注明 Default 服务路由下的 overlap streaming 支持需要进一步完善，见 [async_response_processor.cpp](../xllm/xllm/core/scheduler/async_response_processor.cpp#L300-L320)。

## 17. 通信分析：哪些能 overlap，哪些不能

### 17.1 TP/EP collective

Engine 仍向每个 TP/DP Worker 发 RPC；模型内部 TP/EP collective 数学量不变。CPU 调度可以与上一拍 collective 的设备执行并行，但 `collectAll` 需要所有 Worker 返回 fake/real 的协议结果，不能把 rank 间同步删掉。

### 17.2 RPC 通信

overlap 隐藏的是“Worker RPC 先返回 fake”带来的 host 等待，不是网络传输本身。跨节点 RPC 延迟若大于设备计算，`collectAll` 仍会成为可见瓶颈。

### 17.3 PD 网络

P-D KV 和 linear state transfer 是 decode 首步的硬依赖；只有 transfer 与其它独立 CPU 工作能并行时才有局部隐藏。KV copy 完成、remote first token 到达和 local decode commit 必须分别打点。

### 17.4 KV push barrier

`step_internal(NO_SYNC)` 在返回前仍 `wait_kv_push()`。如果层级 KV cache transfer 正在 push，overlap 的 future 可能在这里被拖住，形成“代码异步、时间线不异步”。

### 17.5 Graph/H2D

NPU Graph double buffer 可让下一槽 metadata/H2D 与当前计算并行，但 SHM polling 线程不能直接物化 Graph 输入；prepare stream 仍是唯一有序入口。设备事件比线程数更决定是否真正重叠。

## 18. 内存账本

### 18.1 Scheduler 层

额外保留：`last_batch_`、last running request/sequence 指针、output target 映射、chunk-prefill 状态队列。它们大多是 host 对象，但会延迟 Request 和 KV block 的释放。

### 18.2 Worker 层

额外保留：cloned host views、`last_step_output_`、`retained_input`、retained dependencies、ready event、Graph 双槽 persistent params 和 graph 实例。`retained_input` 的目的不是缓存结果，而是保证异步 kernel 仍持有合法 tensor storage。

### 18.3 KV block 的峰值

overlap 可能同时存在“当前设备正在读写的 block”和“下一拍已经 reservation 的 block”。Pause KEEP/ABORT 先 drain 就是为了不让 deallocation 与 in-flight 访问重叠。

### 18.4 Graph 双缓冲的代价

Graph slot 数从 1 变为 2，persistent parameter、输入地址、bucket graph 可能近似翻倍。是否值得取决于 decode batch bucket 数量；对大量动态 bucket 的 workload，Graph 内存会比 token tensor 更显著。

### 18.5 监控建议

至少同时观察 active activation memory、KV free blocks、Graph bucket 命中、eager fallback、in-flight batch 数和 response queue 深度。只看 GPU/NPU allocated bytes 无法发现 host view 被错误复用。

## 19. 精度与行为影响

### 19.1 数值精度

正确实现不会改变 forward 数学；假 token 不会进入模型计算，进入模型前已被真实 output 替换。Graph replay、NPU `replace_token` 和 ready event 只改变时序。

### 19.2 采样行为

采样 RNG 若是全局状态，线程/批次顺序改变可能改变随机序列；固定 seed 不能自动保证跨 overlap 模式逐 token 一致。Greedy、temperature=0、top-k/top-p、logprobs 都应分别对比。

### 19.3 停止条件

fake token 不更新 logprob 和 finished；真实替换才触发 EOS、stop string、max tokens。流式输出从后向前寻找最近的非负 token，防止把占位符发给客户端。

### 19.4 Beam search

Batch 在 fake 阶段不处理 beam search，只有真实替换阶段才清空 output targets 并处理 beam result。beam 可能替换 SequencesGroup 内的 Sequence 对象，所以每次提交前都要 refresh pointers。

### 19.5 统计指标

生成 token、usage total tokens、TPOT 都需要扣除 overlap 的额外 fake 位置；否则服务会报告多一个 token，TPOT 分母也会偏移。相关修正见 [request.cpp](../xllm/xllm/core/framework/request/request.cpp#L75-L95) 和 [request.cpp](../xllm/xllm/core/framework/request/request.cpp#L155-L175)。

## 20. 与 Graph、Chunked、Prefix、Spec 的组合表

| 组合 | 主要收益 | 主要新增约束 | 当前判断 |
| --- | --- | --- | --- |
| overlap + Graph | 隐藏 host prepare 和 launch | bucket、双槽、event、NPU 限制 | 条件支持 |
| overlap + chunked prefill | chunk 计算期间准备下一拍 | prefill 队列、KV/linear state 顺序 | 条件支持 |
| overlap + prefix cache | commit 延迟时保留共享 prefix 引用 | hash 失效、引用计数、真实 token 提交 | 代码可达，需组合压测 |
| overlap + speculative/MTP | 叠加 host/设备流水 | 内部 Worker 关闭 overlap、接受长度提交 | 代码可达，需模型验证 |
| overlap + PD | Decode 首拍和本地调度重叠 | transfer/linear state 完成边界 | Decode 条件支持，Prefill 关闭 |
| overlap + embedding/REC | 无明确 decode 依赖 | 服务接口/调度器语义不同 | 强制关闭 |

## 21. 测试证据

### 21.1 Batch fake replacement

[KeepTargetsForOverlapReplacement](../xllm/tests/core/framework/batch/batch_test.cpp#L1850-L1915) 验证 fake `-1` 先被写入，真实 `101` 再覆盖且触发 max-token finish。

### 21.2 MTP 预占和 composite KV

[OverlapMTPReplacementSkipsPreemptedSequenceWithoutKVBlocks](../xllm/tests/core/framework/batch/batch_test.cpp#L1915-L1985) 覆盖 preempt 后无 KV block 的替换安全性；[OverlapMTPReplacementKeepsCompositeKvBlocks](../xllm/tests/core/framework/batch/batch_test.cpp#L1985-L2050) 覆盖 composite manager 的 KV capacity 与多 token 替换。

### 21.3 Sequence 统计

[sequence_generated_tokens_test.cpp](../xllm/tests/core/framework/request/sequence_generated_tokens_test.cpp#L70-L115) 验证 fake token 不增加 generated count，真实 `update_last_step_token` 才计数。

### 21.4 配置降级

[disagg_pd_config_test.cpp](../xllm/tests/core/framework/config/disagg_pd_config_test.cpp#L30-L60) 验证 PD 配置会关闭 overlap；[continuous_scheduler_test.cpp](../xllm/tests/core/scheduler/continuous_scheduler_test.cpp#L560-L640) 提供同步调度基线，但目前没有覆盖完整的跨 Worker overlap timeline。

### 21.5 仍缺的测试

当前源码证据不足以宣称以下项目已经通过：

- 多请求、多 DP、TP/EP 的真实 overlap 端到端测试。
- NPU Graph 双槽 replay 与 bucket miss 混合压力。
- chunked prefill + KDA/GDN + overlap 的长序列恢复。
- PD transfer + overlap 的网络抖动和取消。
- streaming callback 顺序、客户端断开、服务路由场景。
- overlap 与 prefix cache 的 block 引用计数压力。

## 22. 可观测性与性能分析方法

### 22.1 先画四条时间线

建议在 trace 中分别画：Scheduler host、Engine/RPC、Worker prepare stream、Worker compute stream。再叠加 PD transfer stream 和 response threadpool。只有四条线都存在，才能判断“重叠”发生在哪里。

### 22.2 关键间隔

- `schedule_to_step`：Scheduler 建 batch 到 Engine 发 RPC。
- `worker_service_latency`：RPC service 处理时间。
- `execution_latency_seconds_model`：模型执行耗时。
- `engine_latency_seconds`：Engine 聚合和 fake/real Batch 处理。
- `responsing_latency_seconds_stream/non_stream`：输出回调耗时。
- `end_to_end_latency_milliseconds`：用户看到的请求总延迟。

指标定义和埋点可从 [metrics.h](../xllm/xllm/core/common/metrics.h) 及上述调用点追踪。不要只比较总吞吐，必须同时看 TTFT、TPOT P50/P99 和设备 gap。

### 22.3 识别“假异步”

出现以下现象时，通常只是接口异步而非时间线重叠：

- `WorkerService::step` future 很快返回，但 `step_internal` 在 NO_SYNC 前等待 KV push。
- `prepare_stream` 每拍都等待 `compute_stream`，Graph double buffer 未生效。
- RPC `collectAll` 耗时接近设备 forward，网络成为瓶颈。
- response threadpool 排队时间大于设备 gap。

## 23. 故障排查顺序

### 23.1 输出错位

先检查 fake id 是否从 `-1` 连续编码；再检查 `output_targets_` 是否在真实替换前被清空；最后检查 beam refresh 和 DP rank 映射。

### 23. KDA/GDN 输出漂移

检查 linear state slot 是否在 compute stream restore；确认 Worker threadpool 仍为单线程 FIFO；检查 chunk N 写入和 chunk N+1 restore 的 event 顺序。

### 23. 偶发非法访问

检查 WorkerService 是否 clone host views、`retained_input` 是否活到 ready event、pause/deallocate 是否先 drain `last_batch_`。

### 23. 吞吐没有提升

检查设备是否已被通信或 KV push 占满；Graph 是否 capture/replay 命中；Scheduler 是否频繁空 batch；response callback 是否阻塞；并确认 FixedStepsScheduler 或平台门控没有实际关闭。

### 23. 请求永不结束

检查空当前 batch 时是否仍提交 last batch；检查最后一个真实 token 是否执行 `handle_last_token`；检查 fake token 是否误触发 finished 或真实替换未发生。

## 24. 最小复现实验设计

### 24.1 同步基线

固定模型、设备、TP/DP、prompt token、max tokens、seed，关闭 overlap/Graph/prefix/spec，记录 TTFT、TPOT、设备 gap、KV 峰值。

### 24.2 只开 overlap

仅将 `--enable_schedule_overlap=true` 打开，确认：

- 第一个 Engine step 返回负数 token。
- 第二次 step 触发 `get_last_step_result`。
- 生成结果与同步模式在 greedy 下相同。
- usage 和 generated count 没有多一个 token。

### 24.3 再开 Graph

仅在 NPU 上打开 `enable_graph` 和 `enable_graph_double_buffer`，观察 Graph slot、bucket capture、prepare wait 是否减少。需要分别测 bucket hit 和 miss。

### 24.4 组合测试顺序

建议按 `overlap -> overlap+chunked -> overlap+Graph -> overlap+spec -> overlap+prefix -> overlap+PD` 逐项增加，不要一次打开所有开关。任何一项失败都回退到上一个已通过组合。

## 25. 代码阅读路线

1. 从 [scheduler_config.h](../xllm/xllm/core/framework/config/scheduler_config.h#L43-L70) 确认开关和默认值。
2. 读 [continuous_scheduler.cpp](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L345-L405) 建立一拍模型。
3. 读 [llm_engine.cpp](../xllm/xllm/core/distributed_runtime/llm_engine.cpp#L1032-L1155) 理解 fake/real 两次提交。
4. 读 [worker_service.cpp](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L150-L305) 找到负数 token 的来源。
5. 读 [worker_impl.cpp](../xllm/xllm/core/runtime/worker_impl.cpp#L640-L676) 和 [llm_worker_impl.cpp](../xllm/xllm/core/runtime/llm_worker_impl.cpp#L210-L245) 理解替换、event、linear restore。
6. 读 [batch.cpp](../xllm/xllm/core/framework/batch/batch.cpp#L630-L715) 与 [sequence.cpp](../xllm/xllm/core/framework/request/sequence.cpp#L343-L440) 理解两阶段提交。
7. 最后读 Graph、MTP、PD 和 AsyncResponseProcessor，确认每个特性如何改变边界。

## 26. 面向开发者的实现约束

### 26.1 新模型接入

如果模型有额外 recurrent state、跨拍 scratch buffer 或异步通信，必须明确：

- 哪个 stream 写入状态。
- 下一拍哪个 stream 读取状态。
- 是否需要 `retained_input` 或 ready event。
- fake token 是否会被模型侧采样/停止逻辑看到。
- prefill、decode、spec verify 的 batch type 是否都适配。

只实现 `step()` 而不实现 overlap 专用的输入恢复，通常是不安全的。

### 26.2 新后端接入

后端至少要实现：

- fake token 到真实输出行的映射。
- 非阻塞 output handoff。
- 输入和输出 stream event。
- host view 生命周期隔离。
- KV transfer 与 overlap 的同步策略。

没有这些条件时，应在配置门控中关闭，而不是让用户自行承担 silent corruption。

### 26.3 新服务接口

服务层要确认 streaming、batch callback、cancel 和 shutdown 都会等待 AsyncResponseProcessor 完成；否则 Scheduler 虽然停止，response threadpool 仍可能访问已销毁 Request。

## 27. 当前支持状态总表

| 能力 | 代码状态 | 证据强度 | 备注 |
| --- | --- | --- | --- |
| ContinuousScheduler overlap | 已实现 | 源码 + 单元测试 | 深度为 1 的 producer-consumer |
| Worker fake/real handoff | 已实现 | 源码 + Batch 测试 | 负数 token 协议 |
| NPU Graph double buffer | 已实现 | 源码 | 需真实设备 trace 验证收益 |
| Chunked prefill 组合 | 条件支持 | 源码 | 线性 state/队列是风险点 |
| MTP/speculative 组合 | 条件支持 | 源码 + Batch 测试 | 内部 Worker 关闭 overlap |
| PD Decode 组合 | 条件支持 | 源码 | transfer mode/设备相关 |
| PD Prefill overlap | 强制关闭 | Master/config | 角色约束 |
| MLU VLM / DCU PD | 强制关闭 | xllm.cpp | 当前门控 |
| FixedStepsScheduler | 不支持 | 源码日志 | 启用后报错 |
| Kimi K3 请求级 overlap | 未验证 | 模型代码可达 | 已有 TP2 prefill 对齐，但实验关闭 overlap |

## 28. 最终心智模型

可以把 xLLM overlap 看成一个两阶段提交系统：

~~~text
计划阶段（speculative / fake）
  Scheduler 选择 batch
  Batch 追加负数占位
  Worker 发起异步 forward

提交阶段（committed / real）
  Worker handoff 输出
  Engine 取回真实 token
  Batch 替换占位
  Sequence 更新 KV、hash、logprob、stop
  ResponseProcessor 发送客户端 delta
~~~

所有困难都围绕“计划对象什么时候可以释放、真实结果什么时候可见”展开：`last_batch_` 保护调度生命周期，fake token 保护序列形状，单线程 Worker 保护线性状态顺序，event 保护设备可见性，Graph 双槽保护输入地址，PD transfer 保护跨实例状态，AsyncResponseProcessor 保护客户端顺序。

读懂这些边界，就读懂了 xLLM 所谓异步调度；看到任何新模型、新后端或新特性时，也可以用同一组问题判断它是否真的兼容，而不是只看一个布尔开关。

## 29. 证据与审阅记录

本文重点引用的实现入口：

- [ContinuousScheduler](../xllm/xllm/core/scheduler/continuous_scheduler.cpp#L345-L715)
- [LLMEngine](../xllm/xllm/core/distributed_runtime/llm_engine.cpp#L1032-L1155)
- [WorkerService](../xllm/xllm/core/distributed_runtime/worker_service.cpp#L100-L330)
- [WorkerImpl](../xllm/xllm/core/runtime/worker_impl.cpp#L580-L700)
- [Worker prepare](../xllm/xllm/core/runtime/worker_impl.cpp#L750-L916)
- [LLMWorkerImpl](../xllm/xllm/core/runtime/llm_worker_impl.cpp#L210-L404)
- [Batch](../xllm/xllm/core/framework/batch/batch.cpp#L630-L715)
- [Sequence](../xllm/xllm/core/framework/request/sequence.cpp#L343-L440)
- [AsyncResponseProcessor](../xllm/xllm/core/scheduler/async_response_processor.cpp#L30-L380)
- [ACL Graph executor](../xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L380-L690)
- [PD Scheduler](../xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp#L950-L985)
- [PD OOC](../xllm/xllm/core/scheduler/pd_ooc_scheduler.cpp#L1750-L1785)
- [Overlap tests](../xllm/tests/core/framework/batch/batch_test.cpp#L1850-L2050)

审阅结论：`主链路源码可追踪；功能组合按条件支持；性能数字未在本文重新测量；Kimi K3 overlap 尚未完成请求级验证`。

## 30. 用一个两请求例子跑完整协议

为了避免只记住函数名，假设系统里有两个 decode 请求 A、B，每轮各产生一个 token，driver worker 只有一个 DP shard。

### 30.1 第 0 拍：填充流水

Scheduler 取到 A、B，建立 `Batch{A,B}`，`last_batch_` 仍为空。

~~~text
Scheduler: running = [A, B], last_batch = empty
Engine:    step([A, B])
Worker:    device forward([A, B]) 已入队
RPC:       返回 next_tokens = [-1, -2]
Batch:     A.tokens 追加 -1，B.tokens 追加 -2
Scheduler: last_batch = [A, B]
~~~

此时 A、B 的 `num_tokens_` 已增加一个逻辑位置，但 `generated_tokens_since_latency_` 仍为 0，EOS 检查也没有把负数当作真实 token。

### 30.2 第 1 拍：同时做两件事

设备可能仍在 forward(A,B)。Scheduler 重新检查队列；如果来了 C，也可以把 C 加入新 batch：

~~~text
CPU:  schedule([A, B, C])
NPU:  forward([A, B])
~~~

Engine 对新 batch 仍发 fake output。等 `engine_->step([A,B,C])` 返回后，Scheduler 才调用 `update_last_step_result(last_batch_)`，从 Worker 取回 A、B 的真实 token，例如 `[101, 202]`。

### 30.3 真实提交

Batch 对 A/B 调 `update_last_step_token`：

~~~text
A.tokens[prompt_len]     = 101
B.tokens[prompt_len]     = 202
A.generated_count       += 1
B.generated_count       += 1
A.kv_cache_tokens_num   -> prompt_len + 1
B.kv_cache_tokens_num   -> prompt_len + 1
~~~

C 的 fake 结果仍等待下一拍。输出回调只会看到 A/B 的新 token，不会看到 `-1/-2`。

### 30.4 A 提前结束

若真实 token 101 是 EOS，A 在第 1 拍提交时 finished。A 仍可能出现在当前设备 forward 的输入中，因为第 1 拍已经调度过它；Scheduler 只能在后续构造 batch 时移除它，并在确认上一拍不再使用后释放 KV。这个“延迟一拍离场”是正常行为。

## 31. 空 shard、DP 与 EP 的边界

### 31.1 为什么空 shard 不能随便补 token

DP/EP 不均衡时某个 worker 可能没有本地 sequence。Worker prepare 会判断 `num_sequences == 0`，只有在需要维持并行拓扑且 backend 不是 VLM 时才注入一个 fake input；VLM 空 shard 会直接返回，避免没有 cache metadata 的 token 进入 KDA 或 paged attention。见 [worker_impl.cpp](../xllm/xllm/core/runtime/worker_impl.cpp#L805-L840)。

### 31.2 DP global token 数

Graph 输入准备会把本地 token 数提升为 `max(dp_global_token_nums)` 的 bucket；但只有所有 DP shard 都是 decode 时才允许独立准备。若一个 shard 仍是 prefill，prepare graph input 会返回，防止不同阶段共享一个 decode Graph。

### 31.3 EP padding

EP 路径要构造 token size per DP group 和 raw token size，再生成 `DpEpPadding`。overlap 只改变 output commit 时机，不改变 padding 的数学契约；然而 fake token 的 batch 行数必须与 padding 后的有效行保持一致，否则 expert load 和 output row 会错位。

## 32. Request 生命周期的时序表

| 时刻 | Request 状态 | Sequence 状态 | KV 动作 |
| --- | --- | --- | --- |
| 入队 | waiting | prompt | 不占 decode 增量 |
| schedule 当前拍 | running | prefill/decode | reservation 已冻结 |
| fake 返回 | running | 逻辑长度 +1，真实生成数不变 | 保留 in-flight block |
| real commit | running 或 finished | 覆盖 fake，更新 stop/hash/logprob | 扩展或确认 cache |
| cancel | cancelled | set_cancel | 先等 last batch drain |
| finished | finished | last token handled 之前仍可在列表 | 最后一次 output 后释放 |
| pause KEEP | paused | preempt 后可恢复 | drain 后迁移/保留 |

这里的“逻辑长度 +1”仅指 tokens 数组游标，不等于已经确认的模型语义长度；这正是统计代码需要减一的原因。

## 33. 失败路径为什么必须和成功路径同样严谨

### 33.1 Worker future 异常

Engine 在 `collectAll` 后检查每个 Worker result，任何一个没有 value 都会 fatal；不能只使用 driver 的结果，因为其它 rank 可能已经改变了 KV 或 expert load 状态。overlap 的 fake 返回不能掩盖 Worker 失败。

### 33.2 设备 forward 被中断

Engine 按 DP group 检查空 output 和 `layer_forward_interrupted_`，再抛出 `ForwardInterruptedException`。Scheduler 应在此时进入 cancel/cleanup，而不是把 fake token 当作可提交结果。

### 33.3 请求在两拍之间取消

`process_batch_output(true)` 会忽略 cancelled request 的流式发送，但仍需要把真实 handoff 消费掉，才能让 Worker 的单元素槽位复位；“不发送输出”不等于“不取结果”。

### 33.4 Graph capture 失败

Graph capture 失败会记录 eager counter 并调用 `model_->forward` 回退。回退不应改变 fake/real 协议；只有设备执行实现变了，Scheduler 和 Batch 的提交边界不变。

## 34. `ForwardInput` 的所有权关系

### 34.1 host 输入

Scheduler/Batch 产生的 host view 可能引用共享内存或可复用 buffer。WorkerService 在 overlap 下 clone，保证 RPC 返回后下一拍不会覆盖地址。

### 34.2 device 输入

`prepare_work_before_execute` 生成 `input_on_device`，随后移动给单线程 Worker task。它不能指向调用栈里的临时 host tensor；否则 Worker 真正执行时 host 对象可能已经析构。

### 34.3 output retained input

NO_SYNC 返回的 `ForwardOutput.retained_input` 保持 input 和依赖 event 存活；消费者拿走真实 output 后才允许释放。这个字段是生命周期协议的一部分，不是调试信息。

### 34.4 Graph persistent buffer

GraphSlot 的 persistent param 持有静态地址，下一拍只能写入另一槽；如果把两个槽的 token/position buffer指向同一个 storage，双缓冲会退化成数据竞争。

## 35. 与 Prefix Cache 的精确交互

### 35.1 lookup 阶段

Prefix Cache 命中后，Batch 的 prompt 起点已经不是 0；overlap 仍可以为 decode 追加 fake，但 `cur_generated_token_idx_` 必须从完整 prompt 长度开始。共享 prefix block 不可在 fake commit 时重新分配。

### 35.2 hash 失效

真实 token 替换 fake 后，Sequence 会从该位置开始失效 block hash 和 linear state hash。否则 Prefix Cache 可能把包含占位符的 suffix 当作可复用前缀。

### 35.3 引用计数

当前 batch 和 last batch 可能同时引用同一 prefix block。Prefix Cache promotion 或 eviction 必须看到两者的引用；pause drain 的必要性同样适用于 prefix block，而不仅是新分配 block。

### 35.4 推荐验证

用同一 prompt 依次运行同步、overlap、overlap+prefix 三组，比较：命中 token 数、首 token、后续 token、block hash 数和释放时机。只比较最终文本，无法发现中间 cache 污染。

## 36. 与采样器的边界

### 36.1 采样发生在哪一拍

模型 forward 输出 logits 后，Worker 侧 sampler 生成 `next_tokens`。overlap 下这个采样仍在设备执行任务内完成，Scheduler 看到的只是 fake；真实采样结果在 handoff 中保持。

### 36.2 logprob/top-k 的归属

fake 阶段不能更新 logprob state；real commit 才把 token、logprob、top tokens 写入 Sequence。Batch 的 `build_token` 只负责把真实 output 转成 Token，不承担 fake 语义。

### 36.3 stop string 的延迟

stop string 依赖 detokenizer 的真实 token 序列。AsyncResponseProcessor 在真实提交后才调用 `generate_streaming_output`，因此 stop string 最多延迟一个 overlap 提交流程，不能在 fake 阶段提前结束。

## 37. 性能模型：收益和成本怎么量化

### 37.1 单请求 TPOT

设设备计算 `D`，调度 `S`，真实结果提交 `C`，可独立准备窗口 `W`：

~~~text
同步周期       ~= D + S + C
overlap 周期   ~= max(D, S + C - W) + visible_barrier
~~~

`visible_barrier` 包括 RPC、KV push、event wait 和 output copy；它不会因为函数名带 async 就自动为零。

### 37.2 多请求吞吐

当 batch 内请求数增大，`D` 往往变长、`S+C` 相对变短，overlap 收益下降；小 batch decode 和高并发短请求更可能受益。长 prefill 如果占满设备，host 工作本来就被隐藏，收益可能接近零。

### 37.3 尾延迟

overlap 会把真实 token 提交推迟一拍，TTFT/最后一个 token latency 可能增加一个小的 drain 成本；吞吐提高不代表 P99 一定提高。应分别记录首拍填充和末拍排空。

### 37.4 资源成本

用粗略账本表示：

~~~text
memory_peak
  = active_KV
  + planned_KV
  + cloned_host_views
  + retained_inputs
  + graph_slot_buffers
  + response_queue
~~~

其中 planned_KV 和 graph slot 是开关带来的增量，response queue 则取决于客户端消费速度。

## 38. 后端差异的阅读方法

### 38.1 CUDA

重点看 token replacement 是否触发隐式同步、输出 tensor 是否在独立 stream 上 ready、KV block copy kernel 是否和 compute stream 有序。不能因为 CUDA future 可以立即 resolve，就断定 D2H 已完成。

### 38.2 NPU

重点看 `can_prepare_npu_graph_decode_input`、ACL Graph persistent param、`ready_event` 和 `DeviceCaptureLock`。NPU 的 Graph/H2D 约束比普通 Torch tensor 更严格。

### 38.3 MUSA

负数索引会经过 CPU 辅助 tensor，可能缩短可隐藏窗口；需要单独 profile，而不能套用 CUDA 的 gap 数字。

### 38.4 MLU/DCU

当前高层配置会对 VLM/PD 等路径强制关闭。即使底层 Worker 出现 overlap 字段，也应以 Master 最终 Options 为准。

## 39. 代码审阅中的常见误读

### 39.1 “`future` 返回就代表 kernel 完成”

错误。overlap 的 future 是 Worker task 的协议 future；设备 output 通过 handoff 和 ready event 在之后才可读。

### 39.2 “负数 token 会进入下一次 forward”

错误。负数只存在于 host/Batch 的临时逻辑状态；Worker 在 decode 输入替换阶段覆盖成上一拍真实 token。

### 39.3 “`last_batch_` 是缓存 batch 的优化”

错误。它是提交协议的所有权锚点，决定哪些 Request、Sequence 和 KV block 仍在飞。

### 39.4 “Graph double buffer 等于两次 forward 并行”

错误。它主要允许下一槽输入和 Graph 参数准备与当前计算并行；模型 forward 仍按 compute stream 顺序执行。

### 39.5 “PD Prefill 也能直接打开”

错误。当前 Master/config 会把 Prefill overlap 关闭；必须先实现并验证跨实例首 token/KV 状态协议。

## 40. 给新开发者的断点清单

建议在以下位置设置日志或断点：

1. `ContinuousScheduler::step_with_schedule_overlap`：记录 current/last 是否为空。
2. `WorkerService::step`：记录 fake token 行数。
3. `WorkerImpl::update_input_by_last_step_output`：记录负数行到真实 output 行的映射。
4. `WorkerImpl::get_last_step_result`：记录 handoff 槽位等待时长。
5. `Batch::process_sample_output`：分别记录 `replace_fake_token=false/true`。
6. `Sequence::append_token/update_last_step_token`：记录 `num_tokens_`、`cur_generated_token_idx_` 和 KV token 数。
7. `LLMWorkerImpl::step_for_schedule_overlap`：记录 linear state restore 的 stream。
8. `AclGraphExecutorImpl::prepare_graph_input`：记录 replay slot、graph key 和跳过原因。
9. `AsyncResponseProcessor`：记录 response thread id 和 callback 排队时长。
10. PD service：记录 remote token、KV block 和 linear state transfer 的完成时间。

这些点可以把一个“输出偶发错位”还原为具体的阶段，而不是靠猜测线程竞争。

## 41. 一次完整的 correctness 对照

对同一组请求，至少保存以下两种 trace：

~~~text
sync:
  real token -> append -> next batch

overlap:
  fake token -> next batch -> real token -> replace -> next commit
~~~

逐拍比较：

- prompt token 不变。
- 每个真实 token id、logprob、top tokens 相同或在采样允许范围内。
- 每个 Sequence 的 KV cache token 数相同。
- block hash 在真实 token 改写后失效并重新计算。
- finished/last_token_handled 的先后关系相同。
- usage 的 prompt/generated/total/cached 数相同。

这套比较比只看最终字符串更能定位 overlap 的时序错误。

## 42. Kimi K3 的模型侧检查表

### 42.1 Full Attention/MLA 层

确认当前 decode 输入替换只改变 token id，不重建错误的 position、block table 和 MLA compressed KV 地址；Graph 路径还要确认 token override 在 replay 前已经可见。

### 42.2 KDA 层

确认：

- `has_linear_attention_layers()` 返回 true 时，linear cache slot 已分配。
- 当前 sequence 的 recurrent state slot 与 P/D transfer 广播的 slot 一致。
- restore 在 compute stream、FIFO worker 线程内执行。
- overlap 下 `wait_kv_push()` 不会在错误的 stream 上读未完成 state。

### 42.3 TP2 现有证据的边界

现有 Kimi K3 五层 TP2 prefill 对齐说明模型层计算和 cache 写入可用；它没有覆盖 fake/real handoff、Graph 双槽、连续 decode 或服务 streaming。因此本篇不把它升级为 overlap E4/E5 证据。

### 42.4 建议的下一次实验

在已经通过的五层 TP2 配置上，仅打开 overlap，先做 greedy 单请求 8 token decode，再做两个请求交错 decode；记录 `last_step_output_` 等待、linear restore event、KV push 等待和结果对齐。通过后再叠加 Graph、chunked prefill 和 PD。

## 43. 关闭 overlap 时应该恢复什么

关闭开关不是只把一个 bool 设为 false：

- Engine 应等待真实 Worker output。
- Batch 应直接 append real token。
- Worker 可以走 LEGACY sync policy。
- NPU manual loader 可使用默认 stream/ATB stream。
- Linear state restore 可在 prepare 阶段完成。
- Graph double buffer 不再提供 overlap 独立 prepare 窗口。
- Scheduler 不应访问 last batch handoff。

这正是同步模式作为排障基线的价值：它改变协议，不只是减少并行度。

## 44. 何时应主动禁用

出现以下任一情况，应先关闭并保留 trace：

- 新模型有未审计的跨拍 state。
- 新后端没有输出 ready event 或 host view 隔离。
- KV/linear state transfer 的完成事件不明确。
- 需要验证 logits 精确一致，且采样 RNG 顺序尚未固定。
- Graph capture 频繁 fallback，导致异步路径比同步更复杂却没有窗口。
- 内存接近上限，planned block 或双槽导致 OOM。

禁用是可观测性和正确性的工程开关，不应被视为性能失败。

## 45. 本文的阅读验收题

读完后，应该能不查文档回答：

1. `ContinuousScheduler` 为什么只提前一拍？
2. `LLMEngine::step` 的 `collectAll` 为什么仍可能很快返回？
3. `-2` 假 token 如何映射到真实 output 的第 2 行？
4. 为什么 WorkerImpl 必须使用 FIFO 单线程？
5. `ready_event` 分别在哪个 stream 记录、哪个 stream 等待？
6. fake token 为什么不增加 generated count、logprob 和 finished？
7. chunked prefill 为什么需要两元素状态队列？
8. 为什么投机解码内部 Worker 关闭 overlap？
9. PD Prefill 为什么被强制关闭，而 Decode 可以条件开启？
10. Graph 双缓冲准备的是另一槽的什么对象？
11. 哪些通信是 overlap 的硬 barrier？
12. Kimi K3 现有 TP2 prefill 对齐能证明什么、不能证明什么？

如果这些问题都能用源码函数和对象回答，才算真正理解 xLLM 的异步调度，而不是只记住启动参数。
