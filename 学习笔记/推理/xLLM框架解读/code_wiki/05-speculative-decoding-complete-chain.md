# xLLM 投机解码全量解读：从草稿模型到验证、回滚与在线服务

> 本文是 `code_wiki` 的第五篇核心文章。
>
> [第一篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/01-online-service-complete-chain.md:1)讲在线服务完整链路，[第二篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/02-scheduler-complete-chain.md:1)讲 Scheduler，[第三篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/03-graph-execution-complete-chain.md:1)讲组图机制，[第四篇](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/code_wiki/04-chunked-prefill-complete-chain.md:1)讲 Chunked Prefill。本文沿着一次请求的真实执行链路，彻底拆开投机解码，重点解释 MTP，并对 Eagle3、Suffix、DFlash、Scheduler、KV cache、PD 分离和采样正确性做统一说明。
>
> 源码基线：`xllm` 提交 `9768548dd359460d46e7fab32d3cd905407fff6d` 及 2026-08-13 当前工作区。文中严格区分“标准算法原理”“xLLM 当前源码行为”“已由测试或请求实测确认”三种结论。

---

## 1. 先建立最终认知

投机解码的目标不是少生成 token，也不是让目标模型变小，而是：

> 先让便宜的草稿侧猜出未来若干 token，再让昂贵的目标模型用一次并行 forward 验证整段候选；只提交目标模型认可的连续前缀。

### 1.1 先用“助理起草，专家审核”理解

普通 Decode 像让昂贵专家每次只写一个字：

```text
专家写第 1 个 token
专家写第 2 个 token
专家写第 3 个 token
```

投机解码像让便宜助理先起草一小段，再请专家一次审核整段：

```text
助理草稿: [d1, d2, d3]
专家审核: 从左到右确认，错到哪里就在那里改正
最终提交: 只提交前缀中已经被专家认可或修正的部分
```

助理不能决定最终答案。Target 模型仍然是权威模型；Draft 只提供候选，从而把多个位置凑成一次更宽的 Target forward。

### 1.2 Target 为什么能“一次验证多个未来位置”而不作弊

这是理解投机解码最关键、也最容易模糊的一点。

Draft 已经给出了候选链：

```text
当前上下文 C
d1 基于 C
d2 基于 C+d1
d3 基于 C+d1+d2
```

Target Verify 把 `[当前 token, d1, d2, d3]` 作为一段带因果 mask 的输入。Transformer 在一次 Prefill 风格的并行计算中，可以同时得到：

```text
位置 1: p(. | C)
位置 2: p(. | C+d1)
位置 3: p(. | C+d1+d2)
位置 4: p(. | C+d1+d2+d3)  # bonus
```

因果 mask 保证每一行只能看左边已有的候选，不能看到右边未来 token。因此是“多个条件概率并行计算”，不是让模型偷看正确答案。

若 `d2` 被拒绝，`d3` 的条件上下文已经建立在错误的 `d2` 上，所以后面的候选必须全部作废。这就是只能接受连续前缀的根本原因。

### 1.3 五个词先分清

| 术语 | 含义 |
|---|---|
| Draft model/side | 更便宜、用于提出候选的模型或算法 |
| Target model | 最终决定输出分布的权威大模型 |
| proposal | Draft 提出的候选 token |
| verify | Target 对整段候选计算条件分布并判断接受 |
| bonus token | 所有 N 个 proposal 都接受时，Target 多给出的第 N+1 个 token |

投机解码是运行时算法；MTP、Eagle3、Suffix、DFlash 是不同的候选产生方式。不能把“启用 MTP”与“投机解码的一般原理”视为同一个概念。

普通自回归 Decode 每次目标模型 forward 只前进 1 个 token：

```text
Target:  t0 -> t1 -> t2 -> t3 -> t4
forward:      1     2     3     4
```

若配置 `num_speculative_tokens=N=3`，一次投机 step 的理想过程是：

```text
当前真实 token: x0

Draft 1: x0 -> 猜 x1
Draft 2: x1 -> 猜 x2
Draft 3: x2 -> 猜 x3

Target 一次验证 [x0, x1, x2, x3]
                   |   |   |   |
                  y1  y2  y3  bonus
```

如果 `x1==y1`、`x2==y2`，但 `x3!=y3`，本轮提交：

```text
[x1, x2, y3]
```

`y3` 是第一个拒绝位置上目标模型自己的 token，因此即使草稿第一个 token 就错，本轮也至少提交 1 个目标 token。若所有草稿 token 都被接受，还能额外提交最后一行目标模型产生的 `bonus token`，最多一次前进 `N+1` 个 token。

这件事成立依赖五个关键事实：

1. 草稿必须明显比目标模型便宜。
2. 目标模型必须能并行验证 `N+1` 个位置，而不是再次串行跑 `N+1` 次。
3. 只能提交从第一个位置开始的连续有效前缀，不能跳过错误位置继续接受后面的 token。
4. 被拒绝候选写过的逻辑状态不能污染下一轮。
5. 随机采样时需要正确的拒绝采样公式，不能只做 token 相等比较。

xLLM 把这五件事分别落在：

```text
Draft Worker / Draft Model
  + Target Verify Input Builder
  + RejectionSampler
  + EmbeddingCache / Sequence / KV metadata 修正
  + Scheduler token 和 KV block 预算
```

---

## 2. 它为什么可能更快

### 2.1 Decode 的瓶颈通常不是“一个 token 算不动”

大模型 Decode 每一步只处理很少的 query token，却要读取大量模型权重和历史 KV。单请求或小 batch 下，设备经常受内存带宽和 kernel 启动开销限制，算力没有被完全吃满。

这里的“内存带宽受限”可以白话理解为：设备大部分时间在搬运大模型权重和 KV，而不是把计算单元完全用满。一次只处理一行 token 时，搬了很多数据，却只推进一个位置。

把未来 `N+1` 个位置放进目标模型的一次 forward，并不一定花费 `N+1` 倍时间。它能提高矩阵计算和访存复用效率：

```text
普通 Decode: 4 次目标 forward，每次 1 行
投机 Decode: 若接受率理想，1 次目标 forward，同时验证 4 行
```

### 2.2 粗略收益模型

设：

- `T_target(1)`：目标模型普通一步 Decode 时间；
- `T_target(N+1)`：目标模型一次验证 `N+1` 行的时间；
- `T_draft`：产生 N 个草稿 token 的总时间；
- `A`：一次投机 step 平均提交 token 数，范围 `[1, N+1]`。

则投机后的平均每 token 时间近似为：

```text
(T_draft + T_target(N+1) + T_validate) / A
```

只有它小于 `T_target(1)` 才真正加速。于是收益不是只由接受率决定：

```text
收益 = 草稿成本 + 目标多行验证效率 + 接受率 + 额外状态管理成本
```

### 2.3 为什么 N 不是越大越好

增大 N 会同时带来：

- 更多潜在提交 token；
- 更多草稿 forward 或更宽的草稿块；
- 更宽的 Target Verify；
- 更多临时 logits、采样参数、cache slot 和 Scheduler token budget；
- 越往后的候选越容易偏离目标，边际接受率下降。

因此 `num_speculative_tokens` 是性能调优参数，不是质量参数，也不是越大越先进。

### 2.4 一个可手算的收益例子

假设：

```text
普通 Target Decode 1 token = 10 ms
Draft 产生 3 个候选       = 3 ms
Target 一次验证 4 行       = 13 ms
其他验证/状态成本          = 1 ms
平均每轮提交               = 3 token
```

普通方法生成 3 token 约需：

```text
3 * 10 = 30 ms
```

投机方法约需：

```text
3 + 13 + 1 = 17 ms
平均每 token 约 5.7 ms
```

但若平均每轮只接受并提交 1 token，同样成本会变成 17 ms/token，反而慢于普通 Decode。真正要测的是平均接受长度、Target verify 时间、Draft 成本和端到端 TPOT，而不是只看 `N=3`。

---

## 3. 三组概念先分清

### 3.1 Draft token、accepted token、bonus token

假设 N=3：

```text
Draft proposals: [d1, d2, d3]
Target rows:      [p1, p2, p3, p4]
```

- `d1..d3` 是草稿提出的候选。
- `p1..p3` 用来判断对应候选是否接受，拒绝处使用目标分布补偿采样。
- `p4` 是前三个候选全接受时的 bonus token。
- 输出宽度因此是 `N+1`，不是 N。

### 3.2 物理写入与逻辑提交

Target Verify 为了并行计算，会为所有候选位置准备 position 和 cache slot，甚至把 K/V 写进去。但最终只提交第一个拒绝点之前的连续前缀。

所以要区分：

```text
物理上：本次 forward 可能计算并写过 N+1 个位置
逻辑上：Sequence 只前进 accepted_len 个 token
```

后续 step 会根据真实接受长度重新确定 token、position、KV 长度和写入位置。被拒绝位置属于可覆盖的推测区域，不能被当成已提交上下文。

### 3.3 投机解码与 MTP

投机解码是一类运行时算法；MTP 是提供草稿的一种模型结构。

xLLM 当前 Worker 工厂识别四种算法：

| 算法 | 草稿来源 | 候选产生方式 | 当前实现特征 |
|---|---|---|---|
| MTP | 训练好的多 token 预测层/浅层模型 | 通常串行提出 N 个 | 主实现，复用目标 embedding/LM head |
| Eagle3 | 目标中间层 hidden + 专用浅层模型 | 基于 MTP Worker 扩展 | 需要捕获多层辅助 hidden |
| Suffix | 历史 token 后缀树 | CPU 查表提出 | 不需要 draft model，固定贪心验证 |
| DFlash | block-diffusion 草稿模型 | 一次并行提出整个块 | NPU ATB 专用草稿体，非因果块注意力 |

算法分派见 [worker.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker.cpp:51)。

---

## 4. 标准投机解码算法

### 4.1 贪心验证

贪心生成时，目标模型在每个位置有唯一的 `argmax` token。验证规则很直接：

```text
从左到右比较 draft_id[i] 与 target_argmax[i]
相等：接受 draft token
不等：提交 target_argmax[i]，停止后续位置
全部相等：再提交 bonus token
```

例子：

```text
draft:  [17, 31, 42]
target: [17, 31, 99] + bonus 8
mask:   [ 1,  1,  1,  0]
output: [17, 31, 99, -1]
```

注意第三个输出是目标 token `99`，不是“什么也不输出”。这保证每个 step 至少推进一个 token。

xLLM 快路径直接使用目标 Sampler 已经生成的 token ID 比较，见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:2281) 和 [rejection_sampler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/sampling/rejection_sampler.cpp:341)。

### 4.2 随机采样为什么不能只比较 token

设草稿分布为 `q(x)`，目标分布为 `p(x)`，草稿采到候选 `x`。标准投机采样以：

```text
accept_prob(x) = min(1, p(x) / q(x))
```

接受该候选。若拒绝，则不能简单从 `p` 重新采样，而应从修正分布采样：

```text
r(x) = normalize(max(p(x) - q(x), 0))
```

这套“接受概率 + 拒绝后补偿分布”保证最终输出分布仍等于目标模型分布 `p`。因此标准投机采样是无损加速，不应改变模型的采样分布。

### 4.3 为什么只接受连续前缀

假设第二个候选被拒绝，第三个候选是基于错误的第二个 token 产生的：

```text
d1 -> d2 -> d3
       X
```

即使 `d3` 恰好等于目标第三行的 token，它的条件上下文已经不成立，也不能继续接受。因此接受 mask 必须在第一个拒绝点之后全部为 false。

[RejectionSampler::build_accepted_mask()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/sampling/rejection_sampler.cpp:172) 会把：

```text
逐位置比较: [1, 1, 0, 1]
提交 mask:  [1, 1, 1, 0, 0]
```

第三个 `1` 对应拒绝位置上的目标补偿 token，最后一位是 bonus 位置。

---

## 5. xLLM 投机解码源码地图

| 层次 | 关键文件 | 责任 |
|---|---|---|
| 配置 | `core/framework/config/speculative_config.{h,cpp}` | 算法、N、draft 路径和专项开关 |
| Master | `distributed_runtime/llm_master.cpp` | 判断是否创建 SSM/SpeculativeEngine |
| Engine | `distributed_runtime/speculative_engine.cpp` | 目标/草稿模型初始化和共享 KV 容量规划 |
| Scheduler | `scheduler/continuous_scheduler.cpp`、`scheduler_policy.cpp` | 为 N+1 验证保留 token 和 block |
| Worker 工厂 | `core/runtime/worker.cpp` | MTP/Eagle3/Suffix/DFlash 分派 |
| 公共 Worker | `core/runtime/speculative_worker_impl.cpp` | 只在纯 Decode 运行，构造通用 Verify 输入 |
| MTP 主链 | `core/runtime/mtp_worker_impl.cpp` | Draft 循环、Target Verify、状态接力 |
| Eagle3 | `core/runtime/eagle3_worker_impl.cpp` | hidden 捕获与 draft/target token 映射 |
| Suffix | `core/runtime/suffix_worker_impl.cpp` | 后缀缓存查找和贪心验证 |
| DFlash | `core/runtime/dflash_worker_impl.cpp` | 并行草稿块和 context KV 写入 |
| 采样 | `core/framework/sampling/rejection_sampler.cpp` | 贪心/随机接受拒绝 |
| 概率压缩 | `core/runtime/spec_input_builder.cpp` | 草稿概率缓存和 Verify tensor 构造 |
| 状态接力 | `core/framework/kv_cache/embedding_cache.{h,cpp}` | 保存最后已接受 token/hidden/offset |
| 异步重叠 | `core/runtime/mtp_async_state.cpp`、`mtp_async_input_builder.cpp` | 设备侧生成下一轮 draft 输入 |
| 模型侧 | `models/llm/mtp_model_base.h` 等 | MTP embedding/hidden 融合和浅层 forward |
| 输出提交 | `core/runtime/params_utils.cpp`、`core/framework/request/sequence.cpp` | `-1` 截断并提交多 token |

建议源码阅读顺序不是从模型开始，而是：

```text
SpeculativeConfig
 -> Worker::Worker
 -> SpeculativeWorkerImpl::step
 -> MTPWorkerImpl::step_decode
 -> prepare_validate_inputs
 -> validate / RejectionSampler
 -> EmbeddingCache
 -> Sequence output commit
 -> MTP model body
```

先看清运行时协议，再读模型层，变量会容易理解得多。

---

## 6. 配置如何打开整条链路

### 6.1 三个最核心参数

配置定义见 [speculative_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/speculative_config.h:79)：

```text
--draft_model=/path/to/draft
--num_speculative_tokens=N
--speculative_algorithm=MTP|Eagle3|Suffix|DFlash
```

- `draft_model` 是草稿权重路径；Suffix 例外，不需要草稿模型。
- `num_speculative_tokens=0` 表示关闭投机解码。
- `speculative_algorithm` 默认是 `MTP`。

Master 的引擎选择条件是：draft 路径非空，或者 Suffix 且 N>0，见 [llm_master.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_master.cpp:48)。非 Suffix 模式没有 draft path 会直接失败，见 [master.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/master.cpp:423)。

### 6.2 专项开关

| 参数 | 作用 | 需要特别注意 |
|---|---|---|
| `enable_opt_validate_probs` | Verify 直接使用 `[B,N]` 候选概率 | 不等于保留完整 draft 分布，见第 14 节 |
| `enable_mtp_draft_body_tp1` | MTP 草稿 body 用 TP=1 | LM head 仍留在目标 TP group |
| `enable_atb_spec_kernel` | 用 ATB speculative kernel 的分组输入 | Verify 会按 Chunked Prefill 形状构造 |
| `speculative_suffix_*` | Suffix 深度、候选数、概率阈值、缓存数和树模式 | 只影响 Suffix |

flags 先载入，JSON 后载入，因此同名 JSON 项覆盖 flag，见 [speculative_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/speculative_config.cpp:139)。

### 6.3 一个启动示例

具体二进制参数仍以第一篇中的服务启动方式为准，MTP 增量配置可写成：

```bash
./build/xllm/core/server/xllm_server \
  --model=/models/target \
  --draft_model=/models/target-mtp \
  --num_speculative_tokens=3 \
  --speculative_algorithm=MTP \
  --port=8000
```

请求协议不会因为启用投机解码而改变：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "target",
    "messages": [{"role": "user", "content": "解释一下投机解码"}],
    "temperature": 0,
    "stream": true
  }'
```

投机解码完全位于服务内部。客户端看见的仍是目标模型 token 流，不需要发送 draft 参数，也不应感知某个 token 来自草稿还是目标补偿。

---

## 7. 从 Master 到 Worker：对象到底怎样组织

### 7.1 为什么 EngineType 叫 SSM

`LLMMaster` 在启用草稿模式时选择 `EngineType::SSM`，随后 Master 创建 `SpeculativeEngine`。这里的命名是历史/架构命名，不应理解成只支持 State Space Model。

`SpeculativeEngine` 会把目标每步解码宽度设置为：

```cpp
num_decoding_tokens = num_speculative_tokens + 1;
```

见 [speculative_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/speculative_engine.cpp:38)。

### 7.2 目标和草稿当前共享设备

当前 `SpeculativeEngine` 要求 target 和 draft 使用相同 device list；不同设备会 `LOG(FATAL)`。草稿 Engine 自身关闭 graph，并标记 `is_draft_engine=true`。

初始化时还会：

- 分别加载 target 和 draft；
- 若最大上下文长度不同，取两者较小值；
- 同设备场景综合两份模型和 cache 开销计算可用 block 数；
- 目标和草稿都需要 KV cache 时，用共同可承受的容量。

所以投机解码并不是“白送性能”：草稿权重、草稿 KV 和中间状态会压缩目标侧可用 KV 容量，可能降低最大并发。

### 7.3 真正的组合逻辑在 Worker 内

`SpeculativeEngine::step()` 最终仍委托目标 `engine_->step()`。进程内 Worker 根据算法创建 `MTPWorkerImpl` 等组合 Worker；组合 Worker 内部持有：

```text
impl_       -> Target LLMWorkerImpl
draft_impl_ -> Draft LLMWorkerImpl（Suffix 除外）
```

因此不要把 `SpeculativeEngine` 想成每个 token 都在 C++ Master 层来回调用两个远端 HTTP 服务。Draft、Verify 和拒绝采样是 Worker 内的一次设备执行协议。

---

## 8. 哪些 batch 才会真正投机

[should_run_speculative_decode()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/speculative_worker_impl.cpp:44) 给出明确门槛：

1. `batch_forward_type` 必须是 Decode；
2. DP metadata 必须表明所有参与 rank 都是 Decode；
3. Prefill、Chunked Prefill、Mixed batch 不进入 steady speculative decode。

公共 `step()` 分派是：

```text
空输入                         -> step_empty
纯 Decode 且 DP 各分片均 Decode -> step_decode
其他                           -> step_prefill
```

这解释了两个容易误解的现象：

- 开启投机配置后，Prompt Prefill 仍然由目标模型正常计算；草稿侧只做状态预热。
- 某个 DP rank 若不是纯 Decode，不能擅自在该 rank 上跑不同宽度的投机路径，否则 collective 和张量形状会失配。

Scheduler 也会在 `num_speculative_tokens>0` 时关闭 Prefill/Decode Mix Batch，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:63)。Chunked Prefill 仍可用于 Prompt，但不会与投机 Decode 混在同一 batch。

---

## 9. MTP 模型到底是什么

### 9.1 它不是把完整目标模型复制一份

MTP，即 Multi-Token Prediction。训练阶段在目标模型后附加一个或多个较浅的预测层，让它根据目标模型 hidden state 和下一位置 token embedding，预测更远的 token。

xLLM 通用 MTP 层的核心输入融合见 [mtp_model_base.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/mtp_model_base.h:105)：

```text
e = RMSNorm(current token embedding)
h = RMSNorm(previous target/draft hidden)

普通路径:
  z = Linear(concat(e, h))

部分模型路径:
  z = e_proj(e) + h_proj(h)

z -> 一个或少数 MTP decoder layer -> norm -> shared LM head
```

它比重新跑完整目标模型便宜，是因为 MTP body 通常只有很少的 decoder layer。

### 9.2 为什么需要目标 hidden state

只给草稿模型 token ID，它需要自己重新编码完整上下文。MTP 直接接收目标模型已经算好的上下文表示，等于从昂贵目标模型那里继承语义状态，再用浅层网络向前猜。

`LLMWorkerImpl` 在 speculative mode 下会把：

```text
model_output.aux_hidden_states（若存在）
否则 model_output.hidden_states
```

放进 `sample_output.embeddings`，见 [llm_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/llm_worker_impl.cpp:360)。这里变量名叫 embeddings，但在投机协议中它经常承载的是目标 hidden state，不能只按“词嵌入表输出”理解。

### 9.3 权重共享

MTP Worker 加载 target 和 draft 后，通常把目标的 word embedding 和 LM head 共享给 draft，避免重复占显存，也确保输出词表映射一致，见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:527)。

Qwen3.5 是当前特殊情况：其 MTP checkpoint 可自带完整 embedding 和 LM head；源码检测该类型后不再强制共享。

### 9.4 草稿 body TP=1

`enable_mtp_draft_body_tp1` 会把草稿 transformer body 的 TP、DP、CP、EP 都改成单 rank group，但 `lm_head_group_` 仍使用目标 TP group，见 [MTPDraftParallelArgs()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:430)。

目的很实际：一个很浅的草稿层若跨很多卡做 TP，通信成本可能超过计算收益；body 放单卡跑，LM head 仍按大词表并行，可以降低草稿延迟。但这会改变每卡内存和通信结构，必须基于实测开启。

---

## 10. MTP Prefill：不是投机，而是给草稿建档

MTP 的 `step_prefill()` 做两次不同目的的 forward，见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:858)。

### 10.1 第一次：正常 Target Prefill

目标模型完整处理本轮 Prompt chunk：

```text
Prompt tokens -> Target -> 首个生成 token + target hidden states + target KV
```

这一步决定用户看到的首 token，仍然服从目标模型采样参数。

### 10.2 第二次：Draft Prefill

`prepare_prefill_inputs()` 对每条序列把 token 向左错一位：

```text
原 Prompt:      [t0, t1, t2, ..., tk]
Draft token:    [t1, t2, ..., tk, target_next]
Draft hidden:   [target 对应位置 hidden ...]
```

然后跑 draft model，建立草稿自己的 KV cache。它的采样结果不会替换目标首 token；这一轮主要是在使 draft cache 与目标上下文对齐。

### 10.3 EmbeddingCache 启动状态

目标 Prefill 结束后，[write_prefill_target_context()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/embedding_cache.cpp:45) 为每个 request 保存：

```text
target 产生的首 token
对应 target hidden state
position_offset = 0
request_id / embedding_id
```

这份状态是第一轮 Draft Decode 的起点。

---

## 11. MTP Decode 主循环：先串行猜 N 个

核心函数是 [MTPWorkerImpl::step_decode()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:987)。忽略异步优化后，主干可以压缩为：

```cpp
state = EmbeddingCache.read(last target state);
draft_input = prepare_draft_extend_inputs(state);

for (i = 0; i < N; ++i) {
  draft_output[i] = draft_model.forward(draft_input);
  cache_selected_draft_probability(draft_output[i]);

  if (i + 1 < N) {
    draft_input.token = draft_output[i].next_token;
    draft_input.hidden = draft_output[i].embedding;
    draft_input.position += 1;
  }
}

return run_validate(draft_outputs);
```

### 11.1 第一个 Draft 为什么可能需要两行

上一轮可能接受 1 到 N+1 个 token。MTP 需要“最后已接受 token + 对应目标 hidden”继续预测；某些全接受或异步修复场景还要携带前一个 token/hidden，保持固定 shape 并修复 draft KV。

因此 `EmbeddingCache::DecodeState` 不只保存最后一项，还保存：

- `token_id / embedding`；
- `prev_token_id / prev_embedding`；
- `position_offset`；
- `all_draft_accepted`；
- overlap 修正 token 和 offset。

字段定义见 [embedding_cache.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/embedding_cache.h:29)。

### 11.2 后续 Draft 怎样续上

每轮草稿 forward 只生成一个候选 token。除最后一轮外，源码提前准备下一位置的 position、KV 长度和 cache slot；等当前草稿输出到达，再填入：

```text
token_ids       <- 当前 draft next_tokens
input_embedding <- 当前 draft hidden
```

见 [prepare_draft_inputs()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:2170) 和 Draft loop 的 [token/embedding 接力](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:1160)。

### 11.3 MTP 不是“一次输出 N 个”的误区

xLLM 当前通用 MTP Worker 为得到 N 个候选，通常执行 N 次浅层 draft forward。加速来自：

```text
N 次很便宜的浅层 Draft
  + 1 次目标模型宽 Verify
替代
N 或 N+1 次完整 Target Decode
```

DFlash 才是一次 draft forward 并行产生整个候选块，后文单独说明。

---

## 12. Target Verify 输入是如何构造的

### 12.1 为什么宽度是 N+1

对 batch 中每条序列，`prepare_validate_inputs()` 构造：

```text
token:    [current_real, draft_1, draft_2, ..., draft_N]
position: [p,            p+1,     p+2, ..., p+N]
```

草稿结果尚未出来时，后 N 列先写入 `-1, -2, ...` 临时假值；`fill_validate_input_from_draft_outputs()` 再在 compute stream 上逐列填入真实 draft token，见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:1255)。

目标模型对第 i 行输出的是“读完该行 token 后的下一个 token 分布”：

```text
输入 current_real -> 验证 draft_1
输入 draft_1      -> 验证 draft_2
...
输入 draft_N      -> 产生 bonus
```

这就是输入宽度、目标输出宽度和最终最大提交宽度都为 N+1 的原因。

### 12.2 两种 metadata 布局

普通路径把每个验证 token 当成一条 q_len=1 的 Decode row：

```text
num_sequences = B * (N+1)
q_max_seq_len = 1
batch_forward_type = DECODE
```

ATB speculative kernel 路径则按每条原序列一个长度为 N+1 的分组 query：

```text
num_sequences = B
q_len per seq = N+1
batch_forward_type = CHUNKED_PREFILL
```

公共构造逻辑见 [speculative_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/speculative_worker_impl.cpp:214)，MTP 的模型专项版本见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:1771)。

### 12.3 SamplingParameters 也要扩 N+1 倍

Target Verify 每个位置都可能需要 temperature、top-k、top-p、repetition penalty、logprobs 等参数。`update_sampling_params()` 将原 batch 的采样张量按 N+1 重复，使每个验证 row 继承所属请求的采样配置。

这说明 Target Verify 不是脱离正常 Sampler 的特殊旁路；它仍必须遵守用户请求的生成参数。

---

## 13. 为什么 Qwen3.5 和 MiMo 的 Verify 要走 Chunked Prefill

普通 batch-decode kernel 可能采用“先读所有 row 的 KV，再统一写新 KV”的执行顺序。Target Verify 内部却有因果依赖：

```text
位置 p+1 必须看到本次 forward 中位置 p 刚写入的 KV
位置 p+2 必须看到 p 和 p+1
```

若 kernel 先完成所有读，再做所有写，后一个验证 token 会读到尚未写好的 cache，形成 read-before-write race。

xLLM 对 Qwen3.5 和 MiMo 明确使用 Chunked Prefill 型 Verify，源码注释直接解释了这一问题，见 [mtp_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:627)。该路径会：

- `batch_forward_type=CHUNKED_PREFILL`；
- `is_spec_verify=true`；
- q_len 为 N+1，构造累计 q 长度；
- 带上上一轮 `num_accepted_tokens`；
- Qwen3.5 NPU 路径额外构造 expanded graph input。

这里的 Chunked Prefill 不是 Scheduler 在处理 Prompt，而是复用其“一个 query chunk 内严格因果可见”的 Attention 语义来完成 speculative Verify。

这也是第四篇和本文的关键交叉点：

```text
Scheduler Chunked Prefill: 长 Prompt 跨 step 切块
Spec Verify Chunked Prefill: 一次 Verify 内让 N+1 个位置因果可见
```

名字相同，使用目的不同。

---

## 14. RejectionSampler：正确性最关键的一层

### 14.1 xLLM 的三条分支

[RejectionSampler::forward()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/sampling/rejection_sampler.cpp:65) 根据 batch 采样模式选择：

```text
all_greedy_sample -> token ID 相等比较
all_random_sample -> 概率拒绝采样
mixed             -> 两条都算，再按每条请求 do_sample 选择
```

若请求需要 logprobs，还会对 Target Verify 的完整 logits 做 `log_softmax`，抽取最终提交 token 的 logprob 和 top logprobs。

### 14.2 当前随机路径的实现

随机分支会：

1. 对 target 的前 N 行 logits 做 softmax 得到 `p`；
2. 取 draft 候选处的 `q(d_i)` 和 target 的 `p(d_i)`；
3. 用 `p(d_i)/q(d_i)` 与均匀随机数比较；
4. 拒绝时从 recovery distribution 采补偿 token；
5. 拼上 bonus，并在第一个拒绝位置后填 `-1`。

实现见 [rejection_sampler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/sampling/rejection_sampler.cpp:197)。

### 14.3 一个必须正视的源码边界：草稿概率被压缩了

标准精确拒绝采样在拒绝后需要完整 `q(x)` 来计算：

```text
normalize(max(p(x) - q(x), 0))
```

但 MTP 和 DFlash 当前会在每个 draft step 后把 `[B,V]` 草稿概率压缩成仅包含已选 token 概率的 `[B]`，见 [process_draft_sample_output()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_worker_impl.cpp:1640)。这样可显著减少 HBM 保存量：

```text
完整缓存: B * N * vocab
压缩缓存: B * N
```

随后 [build_validate_tensors()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/spec_input_builder.cpp:527) 有两种形式：

- `enable_opt_validate_probs=true`：直接传 `[B,N]` selected-only probability；
- `false`：scatter 成 `[B,N,V]`，但除候选 token 外仍全是 0。

第二种只是恢复了 dense **形状**，没有恢复 draft 在其他词表项上的真实概率质量。单元测试也验证 dense tensor 的每行总和等于被选 token 的单点概率，而不是 1。

数学后果是：selected-only 分支拒绝后会把目标分布中候选 token 置零再归一化；sparse-dense 分支则计算 `p - sparse(q)`。两者通常都不等于使用完整 q 的标准 `(p-q)_+`。

因此必须做出严格区分：

- **贪心验证**：当前 token-ID 比较逻辑与目标 greedy 结果一致。
- **标准随机投机采样理论**：保留完整 q 时可严格保持目标分布。
- **xLLM 当前 MTP/DFlash 概率压缩路径**：只保留 `q(d_i)`，一般不具备完整 q 拒绝补偿的严格分布等价保证。

这不是开关名字能消除的问题。`enable_opt_validate_probs=false` 只把 selected-only 值 scatter 到 dense shape，并不会复原已经丢弃的其他概率。若业务要求随机采样分布严格等价，应把这一点视为实现风险，优先做统计分布测试或保留完整 draft distribution 的实现改造；不能仅看到类名 `RejectionSampler` 就默认已满足教科书证明的全部前提。

### 14.4 Fused kernel 的约束

Fused rejection kernel 只支持 dense draft probs，并且只支持把拒绝后位置 mask 掉，不支持恢复原值。源码仅在“不需要 logprobs且要求 mask rejected tokens”时启用 fused path。

若 `enable_opt_validate_probs=true` 直接传 `[B,N]`，fused kernel 的 dense shape 检查也决定它不能走该路径。

---

## 15. 接受、拒绝与“回滚”到底怎样发生

### 15.1 输出不是 accepted_count，而是 `-1` 截断行

若 N=3，一条请求可能返回：

```text
全接受:     [t1, t2, t3, bonus]
第二个拒绝: [t1, fix2, -1, -1]
首个拒绝:   [fix1, -1, -1, -1]
```

`-1` 是内部终止哨兵。[forward_output_to_proto()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/params_utils.cpp:84) 遍历二维 token 行，遇到 `-1` 立即停止序列化，所以用户只会收到连续有效前缀。

### 15.2 EmbeddingCache 保存下一轮真实尾部

[write_target_context()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/kv_cache/embedding_cache.cpp:119) 从左扫描到第一个负数，得到 `accepted_len`，然后缓存：

```text
last accepted token / hidden
previous accepted token / hidden
position_offset = accepted_len - 1
all_draft_accepted = (accepted_len == N+1)
```

下一轮 Scheduler 可能仍按普通 step 留下旧位置或 overlap placeholder，`update_decode_step_input()` 会用这个 offset 修正真实 current token、position 和 KV length。

### 15.3 Sequence 如何一次提交多个 token

关闭 schedule overlap 时，每个有效 token 依次调用 `Sequence::append_token()`，同步增加 `num_tokens` 并推进逻辑 KV token 数。

开启 overlap 时，Scheduler 可能已经放入 fake token；[Sequence::update_last_step_token()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.cpp:390) 会用真实结果覆盖假 token。`token_offset>0` 时还会扩展序列长度、推进 KV 状态并移动临时位置。

### 15.4 为什么源码里很少看到“清空所有 rejected KV”

Paged KV 是按逻辑长度和 slot mapping 解释的。拒绝后：

- Sequence 只提交 accepted prefix；
- 下一轮位置由 EmbeddingCache 的 accepted offset 修正；
- cache slot 根据真实位置重新构造；
- 被拒绝位置位于逻辑尾部之外，后续可覆盖。

所以“回滚”主要是**逻辑长度、尾 token、hidden 和下一次 slot mapping 的修正**，不一定需要把整片物理显存归零。真正危险的是下一轮仍使用 speculative 最大位置，或者让 repair row 覆盖已接受 KV；异步 builder 对此有专门保护。

---

## 16. Schedule Overlap：把下一轮 Draft 提前发射

### 16.1 不重叠的关键路径

```text
Draft x N
 -> Target Verify
 -> RejectionSampler
 -> D2H accepted tokens
 -> CPU 更新 EmbeddingCache/Sequence
 -> 构造下一轮 Draft
```

目标 Verify 结束到下一轮 Draft 发射之间存在 host 同步空洞。

### 16.2 xLLM 的设备侧 accepted state

`mtp_async::AcceptedState` 从设备上的 accepted token/embedding 直接计算：

- accepted length；
- 是否全接受；
- last/previous token；
- last/previous hidden；
- 下一轮 base position 和 KV length。

定义见 [mtp_async_state.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_async_state.h:33)。

[prepare_next_draft_from_accepted_state()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/mtp_async_input_builder.cpp:80) 可在设备侧填好下一轮 Draft 的 token、embedding、position 和 cache slot；NPU 还会尝试专用 `mtp_prepare_next_draft` kernel。

### 16.3 repair row 为什么存在

为保持 Graph 输入 shape 稳定，下一轮 Draft Extend 可能固定使用 `[repair, current]` 两行。若上一轮全接受，repair row 用来补齐/修复必要的 draft KV；若发生拒绝，它不应覆盖有效位置。

源码把拒绝场景的 repair cache position 重定向到未来 scratch 位置，使其只承担形状填充作用而不污染有效 draft KV。

### 16.4 真正的 overlap 边界

`run_validate()` 在目标验证和拒绝采样后，可以在返回 Scheduler 之前直接 `enqueue_next_first_draft()`。于是：

```text
设备: Target Verify -> Reject sample -> 下一轮 Draft 0
主机:                          同时处理 accepted token / Scheduler
```

但 batch 内容变化、请求切换或 pending context 不匹配时，预发射结果必须 drain 并退回 host materialization 慢路径。Overlap 是稳态优化，不改变接受拒绝语义。

---

## 17. Scheduler 为什么也必须懂投机宽度

### 17.1 每条序列需要 N+1 token budget

普通 Decode 每条序列本轮预算 1。投机 Verify 要处理当前 token 加 N 个草稿位置：

```text
decode_step_tokens = N + 1
```

[SchedulerPolicy::schedule_decode_from_queue()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:583) 在纳入序列前检查：

```text
remaining_token_budget > N
allocate sequence to num_tokens + N
allocated_tokens += N + 1
```

注意 KV manager 额外分配的是 N 个未来位置，因为当前真实 token 的位置已经存在；模型计算预算则是 N+1 行。

### 17.2 Overlap 为什么变成 2N

`ContinuousScheduler` 在启用 schedule overlap 时设置：

```text
min_speculative_tokens_required = 2 * N
```

否则为 N，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:167)。原因是 overlap 存在在途的推测状态和下一轮预留，Scheduler 需要更保守的容量边界，避免两代 speculative positions 争用末尾空间。

### 17.3 为什么低接受率仍消耗完整调度预算

Scheduler 在 forward 前不知道会接受几个 token，只能按最大验证宽度预留。即使最终只提交 1 个，设备仍然算了 N+1 行，KV 也必须有空间容纳最坏情况。

所以性能分析不能只看输出 token/s，还要看：

```text
平均提交 token 数 / 每次 Verify 预留宽度
```

### 17.4 指标

Scheduler 记录：

- `speculative_mean_tokens_per_decode_step`：每条 Decode 序列每 step 平均提交 token 数；
- `speculative_per_token_latency_milliseconds`：把一个 step 的 TBT 按提交 token 数摊销；
- Worker 分别累计 draft、target、validation 执行时间。

实现见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:440) 和 [metrics.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/common/metrics.cpp:165)。

---

## 18. KV cache 与显存账本

### 18.1 两套 cache

MTP/Eagle3/DFlash 通常有：

```text
Target KV cache: 完整目标模型的历史状态
Draft KV cache:  草稿浅层模型的历史状态
EmbeddingCache:  每请求最后已接受的目标 hidden/token
```

草稿模型虽浅，仍可能需要按序列保存自己的 KV。`MTPWorkerImpl::allocate_kv_cache()` 会给两者分配 cache；草稿 TP1 时按 draft model args 单独构造 shape。

### 18.2 为什么并发可能下降

同一设备上，显存要容纳：

```text
target weights + draft weights
+ target KV + draft KV
+ N+1 target logits / hidden / metadata
+ graph persistent buffers / accepted-state buffers
```

`SpeculativeEngine::allocate_kv_cache()` 最终选择目标和草稿共同能承受的 block 数。若普通模式可并发 100 条，开启投机后不应默认仍能保持同样并发。

### 18.3 KV 正确性的三条硬约束

1. Target Verify 的每个 position 必须映射到正确 slot。
2. 后一个验证 token 必须看到前一个验证 token 的 KV。
3. 下一轮只能从 accepted prefix 的逻辑尾部继续。

投机解码最难排查的 bug 往往不是采样公式，而是 position、kv_seq_len、block table 和 cache slot 四者不一致。

---

## 19. Eagle3：目标中间层是草稿的老师

### 19.1 与普通 MTP 的区别

Eagle3 不只使用目标最后一层 hidden，而是从目标若干中间层捕获辅助 hidden，再融合成草稿输入。`SpeculativeConfig::requires_aux_hidden_capture()` 对 `Eagle3` 返回 true，Worker 据此让 target model 填充 `layers_to_capture`。

目标 Worker 输出时优先使用 `aux_hidden_states`，因此公共 MTP 接力协议无需为 Eagle3 另造一套输出结构。

### 19.2 模型侧融合

当前 Qwen3 Eagle3 NPU 模型的 FC 输入宽度是 `3 * target_hidden_size`，随后进入一个 Eagle3 decoder layer 和 norm，见 [qwen3_eagle3.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/npu/qwen3_eagle3.h:100)。

`Eagle3WorkerImpl` 继承 MTP Worker 主循环，但：

- target 开启 aux hidden capture；
- placeholder hidden 宽度改为 `3*hidden`；
- 加载 draft-to-target token 映射；
- 草稿采样 token 可通过 hot-token 映射转换到目标词表 token。

见 [eagle3_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/eagle3_worker_impl.cpp:28)。

因此 Eagle3 改的是“怎样产生更准的草稿”，验证、拒绝、连续前缀提交仍复用同一套机制。

---

## 20. Suffix：不用模型也能猜

### 20.1 基本思想

语言中经常出现重复片段，例如代码缩进、固定模板、JSON 字段或长上下文中的重复语句。Suffix speculative decoding 保存请求的 token 历史，用当前最近后缀在缓存树中寻找曾出现的续写：

```text
历史: ... A B C D E ... A B C ?
匹配:     A B C
草稿:           D E ...
```

它没有神经网络 draft forward，草稿成本主要是 CPU 后缀结构查询。

### 20.2 xLLM 实现

`SuffixWorkerImpl` 在 Prefill 阶段把 Prompt token 和目标首 token 写入 `SuffixDecodingCache`。Decode 阶段按 request 查询最多 N 个候选，填进通用 Verify 输入，再让目标模型验证，见 [suffix_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/suffix_worker_impl.cpp:97)。

没有匹配时，当前实现用输入 token 填充候选位作为 fallback；这些候选大概率很快被目标拒绝，但协议仍保证至少输出目标修正 token。

### 20.3 当前重要限制

Suffix 的 `validate()` 无视用户的 `sampling_params`，强制构造全 greedy `RejectionSampler`，源码注释明确写着“regardless of the user's sampling parameters”，见 [suffix_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/suffix_worker_impl.cpp:381)。

因此当前 Suffix 最适合 `temperature=0` 的贪心服务。对随机采样请求，它会改变预期采样语义，不能只把它看成透明性能开关。

### 20.4 适合的场景

- 代码补全和重复模板；
- 长文档中大量引用原文；
- JSON、SQL、日志等高重复结构；
- 没有可用 MTP/Eagle checkpoint，但希望尝试低成本草稿。

自然语言开放式创作若重复度低，Suffix 接受率可能不足以覆盖 N+1 Target Verify 的额外成本。

---

## 21. DFlash：一次并行提出整个候选块

### 21.1 与 MTP 串行草稿不同

MTP：

```text
draft_1 -> draft_2 -> ... -> draft_N
```

DFlash：

```text
[real token, MASK, MASK, ..., MASK]
       一次 draft forward
 -> [draft_1, draft_2, ..., draft_N]
```

[build_query_rows()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/dflash_worker_impl.cpp:163) 为每条序列构造宽度 N+1 的 query，第 0 行是真实 token，后 N 行是配置中的 mask token。

### 21.2 为什么是非因果块注意力

DFlash 草稿模型属于 block-diffusion 设计，块内各 draft position 可以同时看完整候选块，不使用普通逐行 causal diagonal mask。模型实现明确把一行 kv mask 重复到所有 q row，见 [qwen3_dflash.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/npu/qwen3_dflash.h:104)。

这不是把普通自回归模型的 mask 随便去掉；它要求专门训练和导出的 DFlash draft checkpoint。

### 21.3 目标上下文怎样进入 Draft

DFlash 也要求目标捕获中间层 hidden。模型侧把这些 hidden 经过：

```text
fc -> hidden_norm -> 各 draft layer 的 K/V projection
   -> k_norm + RoPE -> 写入 Draft paged KV
```

见 [DFlashQwen3ModelImpl::write_context_kv()](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/npu/qwen3_dflash.h:118)。

### 21.4 当前平台边界

源码明确说明当前 DFlash draft body 注册为 NPU ATB-only；Torch backend 会在模型创建阶段失败，见 [dflash_worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/dflash_worker_impl.cpp:315)。

DFlash 一次输出 `[B,N]` token block 和 selected probabilities，再复用 Target Verify 与 RejectionSampler。它减少了 N 次串行 Draft 的关键路径，但草稿 body 更复杂、平台约束也更强。

---

## 22. Graph、ATB 与投机解码

### 22.1 Graph 需要固定最大宽度

投机 Decode 的目标输入通常固定为 `B*(N+1)` row 或每序列 q_len=N+1，这天然适合按最大 shape 捕获 Graph；但 accepted_len 每轮可变，下一轮真实 token/position/hidden 又是动态的。

因此 Graph 需要把以下数据设计成持久、可更新输入：

- token IDs；
- positions；
- KV lengths 和 slots；
- target aux hidden states；
- accepted lengths；
- MTP top-k state 等模型专项状态。

MTP target/draft options 都开启 `enable_graph_aux_hidden_states`，使 hidden 接力能进入 Graph 的持久输出协议。

### 22.2 NPU double buffer 当前会自动关闭

xLLM 初始化配置时，只要 NPU 上 `num_speculative_tokens>0` 且 `enable_graph_double_buffer=true`，就记录 warning 并把 double buffer 关闭，见 [xllm.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/xllm.cpp:367)。

这不等于所有 Graph 都关闭；它只说明当前 speculative path 与这一种 graph double-buffer 机制不兼容。是否命中 ACL Graph、piecewise graph 或模型专项 graph，还要结合第三篇中的 executor 选择继续判断。

### 22.3 ATB speculative kernel

`enable_atb_spec_kernel` 会让 Verify 输入从“B*(N+1) 条独立 Decode row”改成“B 条、每条 q_len=N+1 的 Chunked Prefill row”，从而由 ATB 算子在一次序列级调用中处理完整验证块。

Qwen3.5/MiMo 即使没有显式打开该 flag，也会因因果读写要求走同类布局。

---

## 23. PD 分离下为什么还要传 hidden

普通 PD 分离只需让 Decode 实例获得：

```text
Prompt KV / 线性状态 + 目标首 token
```

MTP 第一轮 Draft 还需要目标 Prefill 产生的 hidden。若 P 实例算完 Prompt 后只把 token 和 KV 交给 D，D 没有这份 hidden，就无法构造正常的 MTP draft input。

因此 xLLM 在 P 侧 `Sequence` 中保存 `mtp_bootstrap_embedding`，首 token handoff 时序列化到 RPC；D 侧校验：

- embedding block slot 有效；
- token ID 有效；
- bootstrap embedding 已定义；

然后写入 Decode Sequence，第一轮 `MTPWorkerImpl::step_decode()` 再调用 `write_mtp_bootstrap_context()` 填入 EmbeddingCache。

P 侧发送见 [disagg_pd_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp:758)，D 侧接收见同文件的 [bootstrap 校验](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp:916)。

这揭示一个通用原则：

> PD 分离传输的不只是 Attention cache；任何 Decode 首步依赖、而 P 阶段才生成的模型状态，都必须成为显式 handoff 协议的一部分。

对 Eagle3/DFlash，这份上下文 hidden 的宽度和用途还可能不同，不能默认所有算法都只传一个普通 hidden-size 向量。

---

## 24. 与其他特性的组合关系

| 特性 | 当前关系 | 原因 |
|---|---|---|
| Continuous Batching | 支持，但每轮按 N+1 计预算 | 请求仍逐 step 重组 |
| Chunked Prefill | Prompt 可切块；不与 Decode 混批 | speculative Worker 只跑纯 Decode |
| Mix Batch | N>0 时关闭 | Prefill/Decode forward 宽度和 collective 不一致 |
| CP | `cp_size>1` 同样关闭 Mix；算法支持看模型/backend | CP 与 speculative metadata 组合复杂 |
| DP/EP | 纯 Decode rank 才投机，token 数乘 N+1 | 保证 collective shape 一致 |
| Prefix Cache | Target 前缀 cache 可继续使用 | Draft cache/hidden 仍需正确 bootstrap |
| PD 分离 | 有专项 bootstrap hidden 协议 | D 首步不能只依赖 KV/token |
| Schedule Overlap | 支持专项异步状态；预算变 2N | 存在下一轮预发射和假 token 修正 |
| ACL Graph | 有专项 hidden/expanded input 支持 | 动态 accepted state 要进入持久 buffer |
| Graph Double Buffer | NPU 当前自动关闭 | 与 speculative 数据生命周期不兼容 |
| Beam Search | Scheduler 有 N+1 预算分支 | 仍需针对模型和端到端输出单独验证 |

这里的“支持”只表示源码存在组合路径，不等于所有模型、设备、并行规模都已完成请求级验证。

---

## 25. Kimi K3 当前边界

本仓库 Kimi K3 已验证的主线包括 NPU Python 模型、MLA、KDA、基础 DP/TP/EP 在线服务与 Decode ACL Graph；Prefix Cache 和真正的多 chunk Prefill 仍是代码可达、缺少请求级专项等价性验证。当前没有证据表明 Kimi K3 checkpoint 已带 xLLM 可识别的 MTP/Eagle3/DFlash draft 权重，也没有完成 Kimi 请求级 speculative decode 验证。

因此必须区分：

```text
xLLM 框架有投机 Worker
!= 任意模型都能自动启用投机解码
!= 当前 Kimi K3 已具备配套草稿模型
```

要让 Kimi K3 真正支持 MTP，至少需要确认：

1. checkpoint 是否训练和导出了兼容的多 token 预测层；
2. ModelRegistry 能否识别对应 draft model type；
3. 目标输出 hidden 宽度是否符合 draft 输入契约；
4. KDA/MLA 混合模型的 draft/target cache 如何推进；
5. NPU Python backend 是否实现 speculative Verify 所需 metadata；
6. Prefill、PD bootstrap、随机/贪心采样和长上下文是否有测试。

在这些问题没有源码和实测证据前，不应通过伪造 `draft_model` 路径或只设置 N 来宣称 Kimi K3 已支持投机解码。

---

## 26. 如何选择算法

### 26.1 有配套 MTP checkpoint

优先从 MTP 开始。它与目标模型训练绑定，通常接受率比纯历史匹配稳定，也是 xLLM 当前最完整的主路径。

先用：

```text
temperature=0
N=1 或 2
关闭 schedule overlap
关闭专项概率优化
```

验证输出等价和状态正确，再逐项扩大 N、开启 overlap/Graph/并行优化。

### 26.2 有 Eagle3 checkpoint 且目标模型支持 hidden capture

Eagle3 值得在 MTP 基线后测试。它可能用更丰富的目标中间层表示提高接受率，但也增加 aux hidden 输出、显存和 Graph 接口复杂度。

### 26.3 高重复、贪心场景，没有 draft model

可评估 Suffix。它部署成本低，但应明确限制随机采样请求，且效果高度依赖文本重复度。

### 26.4 有专用 DFlash checkpoint 和 NPU ATB 环境

DFlash 的并行 block draft 能缩短串行草稿关键路径，但不是通用 checkpoint 可开启的开关。需要把模型导出、mask token、target layer capture 和 backend 一起验证。

---

## 27. 性能调优方法

### 27.1 先记录普通 Decode 基线

至少记录：

- TTFT；
- TPOT / inter-token latency；
- output tokens/s；
- 并发请求数；
- KV block 容量和显存；
- greedy 与随机采样分别的输出。

### 27.2 扫 N，而不是只测一个值

建议 `N=1,2,3,4` 逐点测试：

| N | 平均提交 token/step | Draft 时间 | Target Verify 时间 | 摊销 TPOT | 可并发数 |
|---|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |

关键比率：

```text
acceptance efficiency = (mean committed tokens - 1) / N
```

减 1 是因为每轮至少有一个目标 token。这个指标越接近 1，草稿位利用越充分。

### 27.3 分拆三段耗时

xLLM 已有 draft、target、validation counter。若总性能没提升，先判断：

```text
Draft 太慢？
Target(N+1) 扩张太大？
Rejection/概率张量太重？
还是 accepted tokens 太少？
```

不要只靠总 token/s 猜原因。

### 27.4 看并发下的收益

单请求 Decode 常更 memory-bound，投机收益可能明显；大 batch 已经把目标模型算力吃满时，Target Verify 多行成本会上升，草稿又占资源，收益可能缩小甚至反转。

投机解码是否有效必须在目标服务并发和 Prompt/输出长度分布下测，不能只用单条短请求下结论。

---

## 28. 正确性测试应该怎样做

### 28.1 贪心端到端等价

固定模型、Prompt、seed，分别运行：

```text
baseline: N=0, temperature=0
spec:     N>0, temperature=0
```

完整 token IDs 应一致，而不只是最终文本看起来相似。覆盖：

- 第一个候选就拒绝；
- 中间拒绝；
- 全部接受并输出 bonus；
- EOS 出现在 accepted prefix 中；
- 跨 KV block 边界；
- batch 中各请求 accepted_len 不同。

### 28.2 随机采样做统计分布测试

不能要求相同 seed 下逐 token 完全一致，因为投机路径消耗随机数的方式不同。应在大量重复样本上比较 token 分布、KL/TV distance 和关键序列概率。

鉴于第 14 节的 selected-only q 边界，当前 MTP/DFlash 随机路径尤其需要这类测试，而不是只跑一次 `temperature=0.8` 看文本是否通顺。

### 28.3 状态测试

应直接断言：

```text
accepted_len
last / previous token
position_offset
next kv_seq_len
next cache slots
all_draft_accepted
```

xLLM 已有：

- `rejection_sampler_test.cpp`：贪心、随机、mask、recovery、fused；
- `spec_input_builder_test.cpp`：Verify 行、概率压缩和 sparse dense；
- `mtp_async_state_test.cpp`：设备侧 accepted state；
- `sequence_mtp_bootstrap_test.cpp`：PD bootstrap 生命周期；
- Scheduler speculative token budget 和 PD handoff 相关测试。

单元测试覆盖组件契约，但完整模型、真实 kernel、Graph、PD 和在线请求仍需端到端测试。

---

## 29. 常见故障与排查顺序

### 29.1 启动即报 draft model path required

检查：

```text
N 是否 > 0
算法是否真的是 Suffix
draft_model 是否被 JSON 覆盖为空
```

除 Suffix 外，Master 明确要求非空 draft path。

### 29.2 Target 与 Draft 最大长度不同

Engine 会取两者较小值。若请求在过去能跑、启用投机后被截断或拒绝，检查 draft `max_position_embeddings`。

### 29.3 position/kv_len mismatch

优先打印每条序列：

```text
current token
start_position
kv_seq_len
accepted_len / position_offset
block table capacity
N+1 validate positions
```

正常 Decode 起点应满足 `start_position + 1 == kv_len`。Qwen3.5/MiMo 有限定范围内的专项归一化，但不能用它掩盖任意长度错位。

### 29.4 开启后输出 token 重复或跳位

检查 EmbeddingCache 的 request_id 是否与 embedding slot 复用冲突、`-1` 是否正确截断、overlap fake token 是否被 `update_last_step_token()` 覆盖，以及 preempt 后旧 pending context 是否被清理。

### 29.5 吞吐下降

按顺序检查：

1. `speculative_mean_tokens_per_decode_step` 是否明显大于 1；
2. draft latency 是否过高；
3. N 是否过大；
4. target Verify 是否走了低效 backend；
5. 开启后 KV block 数和 batch size 是否下降；
6. 大并发下是否已经 compute-bound；
7. overlap/Graph 是否实际生效。

### 29.6 随机采样质量或分布变化

先不要把问题归因于模型。核对是否走了 selected-only draft probability、Suffix 是否强制 greedy、fused kernel 是否改变了可用路径，并做 baseline/spec 多样本统计比较。

---

## 30. 一次 N=3 的完整链路

下面把在线请求从 Scheduler 到输出串成一条线。

### Step 1：Scheduler 选中请求

Sequence 已处于 Decode，Scheduler 检查至少有 4 个 token budget，并为未来 3 个 speculative position 分配 KV 容量。

### Step 2：Worker 判断能否投机

batch 必须是纯 Decode，DP 各 rank 也必须都是 Decode；否则走 Prefill/Empty 对齐路径。

### Step 3：读取上轮 Target 尾状态

EmbeddingCache 给出最后已接受 token、target hidden、position offset，以及必要时的 previous token/hidden。

### Step 4：Draft 连续提出三个 token

```text
target tail -> MTP draft -> d1
d1 hidden   -> MTP draft -> d2
d2 hidden   -> MTP draft -> d3
```

每步保存 token 和候选 token 的草稿概率；非 greedy 时概率被压缩为 selected-only。

### Step 5：构造 Target Verify

```text
tokens    = [current, d1, d2, d3]
positions = [p, p+1, p+2, p+3]
```

采样参数扩成 4 份，DP token counts 也乘 4。

### Step 6：目标模型一次 forward

Target 返回四个位置的 logits/next tokens/hidden。前三行用于验证 d1-d3，最后一行提供 bonus。

### Step 7：RejectionSampler

假设 d1、d2 接受，d3 拒绝，输出：

```text
[d1, d2, repair_from_target, -1]
```

### Step 8：保存真实尾状态

EmbeddingCache 写入第三个有效输出 token 和对应 target hidden：

```text
accepted_len = 3
position_offset = 2
all_draft_accepted = false
```

### Step 9：提交 Sequence

RPC/本地输出在 `-1` 处截断，只把三个有效 token 交给 Sequence 和流式响应。

### Step 10：下一轮

下一轮 position 和 KV length 从 accepted offset 继续；之前计算过但未提交的 speculative 尾部不进入逻辑上下文，cache slot 可被覆盖。

---

## 31. 源码阅读检查题

如果能独立回答以下问题，说明已经真正掌握 xLLM 投机解码主线：

1. 为什么 N 个草稿 token 对应 N+1 个 Target Verify row？
2. 为什么第一个候选被拒绝时仍至少输出一个 token？
3. 为什么后续恰好匹配的候选也不能越过第一个拒绝点继续接受？
4. MTP Prefill 为什么还要跑一次 draft model？
5. `sample_output.embeddings` 在 speculative path 中为什么可能实际是 hidden states？
6. Target 和 Draft 为什么各有 KV cache？
7. Scheduler 为什么分配 `num_tokens+N`，但 token budget 记 `N+1`？
8. Qwen3.5/MiMo Verify 为什么伪装成 Chunked Prefill？
9. EmbeddingCache 的 `position_offset` 怎样修复下一轮输入？
10. schedule overlap 为什么让最小 speculative reserve 变成 `2N`？
11. `enable_opt_validate_probs=false` 为什么不能恢复完整 draft 分布？
12. Suffix 为什么不需要 draft path，又为什么当前只适合 greedy？
13. DFlash 为什么一次出整个块，而 MTP 通常要跑 N 次 draft？
14. PD handoff 为什么除了 KV 和首 token 还要传 MTP hidden？
15. 为什么 Kimi K3 不能因为框架有 MTPWorker 就直接宣称支持 MTP？

---

## 32. 最终总结

投机解码最简洁的理解是：

```text
便宜模型负责提出候选
昂贵模型负责一次并行裁决
拒绝采样负责保持输出语义
EmbeddingCache 负责保存真实尾部
Scheduler 负责为最坏宽度预留资源
Sequence 只提交连续有效前缀
```

xLLM 的 MTP 完整主链是：

```text
Target Prefill 产生首 token + hidden
 -> Draft Prefill 建立草稿 KV
 -> EmbeddingCache 保存 target tail
 -> N 次浅层 Draft 逐 token 提案
 -> 构造 B*(N+1) 或 B 条宽 Verify 输入
 -> Target 一次并行计算 N+1 个位置
 -> Greedy compare / RejectionSampler
 -> `-1` 截断 accepted prefix
 -> 更新 hidden、position、KV logical tail
 -> 一次向用户提交 1..N+1 个 token
```

真正决定系统能否正确且更快的，不是“有没有 draft model”这一件事，而是四项同时成立：

1. 草稿模型足够便宜且接受率足够高；
2. Target Verify 的因果 Attention 和 cache slot 完全正确；
3. 拒绝采样拥有其数学正确性所需的信息；
4. accepted prefix 能跨 Scheduler、KV、Graph、PD 和输出层保持同一个真实进度。

最后保留本文最重要的工程判断：

> 投机解码不是一个孤立 kernel，而是一套跨模型、采样、缓存、调度和服务输出的事务协议。候选可以大胆计算，只有目标确认的连续前缀才能提交；所有下一轮状态都必须从这个提交点继续。
