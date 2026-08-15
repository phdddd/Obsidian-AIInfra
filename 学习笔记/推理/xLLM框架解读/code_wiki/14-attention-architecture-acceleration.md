# xLLM Attention 架构加速：GQA、MLA、线性状态与稀疏 Indexer

## 1. 与运行时优化的边界

Paged KV 只改变存放方式；本篇方法改变“模型本来需要保存和计算什么”。GQA/MLA/线性注意力通常由 checkpoint 架构定义，不是服务启动后可任意替换，否则就换了模型。

## 2. GQA/MQA：减少 KV heads

MHA 中 query 与 KV head 数相同；GQA 让多个 query heads 共享一组 KV，MQA 则只保留一组 KV。

```text
KV_bytes ∝ num_kv_heads
```

这直接降低 KV 容量和 Attention 读带宽。xLLM 从 model args 读取 `num_key_value_heads`，KV shape、并行 head mapping 和 backend 必须共同遵守。

精度由模型训练决定；运行时按 checkpoint 正确执行不属于近似。

## 3. MLA：保存压缩 latent

MLA 不把完整 K/V heads 逐 token 常驻，而保存低秩 latent 和必要 RoPE 分量，Attention 时通过投影/吸收技巧恢复等价计算。

源码线索：

- `model_args.enable_mla()` 与 HF loader 判定。
- `core/framework/kv_cache/kv_cache_shape.cpp` 的 MLA shape。
- `indexed_kv_cache_impl.cpp` 注明 DeepSeek MLA 没有普通 value cache。
- 各平台 `deepseek_v2_attention`、fused MLA Q/KV kernel。
- `enable_customize_mla_kernel` 控制 NPU 专用路径。

## 4. MLA 内存与计算交换

普通 KV 随 `kv_heads × head_dim` 增长；MLA 随 latent rank 和 rope dim 增长，通常显著更小。代价是额外低秩投影、权重吸收和更复杂 kernel。

Decode 往往受 KV 读带宽限制，因此少读 KV 即使多做一点计算也可能更快；Prefill 的收益结构不同。

## 5. MLA 的通信

DP/TP mapping 常针对 MLA Attention 单独定义，DP 可让不同 token/request 副本独立处理，TP 切投影或 heads。PD 传输的 KV 表示也更小，但异构 TP 必须理解 latent shard，而非套普通 K/V head 拆分。

FlashComm eligibility、custom MLA kernel 和 CP 组合要看模型分支，不应从单个开关推断。

## 6. 线性注意力：从 KV 列表到递推 state

线性注意力把历史压缩进固定/受控大小状态：

```text
state_t = update(state_{t-1}, k_t, v_t, gates...)
output_t = query(state_t, q_t)
```

Decode 不再扫描全部历史 KV，复杂度与缓存增长显著下降。xLLM 支持 Gated Delta Net/KDA/Mamba 类 state，并用 `mamba_ssm_dtype`、linear layer mask、state slots 管理。

## 7. 混合层模型

Qwen3-Next、Kimi K3 等可把 full attention 与 linear attention 交错。模型既需要 KV blocks，也需要 linear state slots。源码优先使用显式 `layer_types` 构造 per-layer mask，兼容非固定 interval 的层分布；只用 `full_attention_interval` 会误映射 K3 的任意 `kda_layers`。

容量估算、worker cache 创建和 decoder dispatch 必须使用同一层类型来源。

## 8. 线性状态的 Prefix/Chunk/Spec 语义

- Prefix Cache 必须保存可恢复 state，而非仅 KV block。
- Chunked Prefill 的 chunk 边界要连续传递 state。
- Speculative 分支被拒绝时要回滚或隔离 draft state。
- PD 要传 state 或在 D 端重建。

这些状态机错误会造成后续所有 token 偏离，而不只是一个 block 错位。

## 9. Sparse Attention 与 Lightning Indexer

DeepSeek 类 sparse/indexer 路径先用低成本索引器为每个 query 选出 top-k 历史位置，再只对候选做主 Attention。源码包含 index heads、index head dim、index cache、量化 indexer kernel 和模型专用 sparse attention。

```text
query -> index scores -> top-k positions -> gather selected KV -> attention
```

它把全上下文扫描变成候选检索 + 稀疏 Attention。

## 10. 稀疏方法的精度分类

若稀疏模式是模型训练时定义的一部分，正确 runtime 是模型语义；若运行时额外提高 sparsity/裁剪候选，则属于近似，top-k recall 会影响输出。

Indexer INT8 还可能改变候选排序，需同时评估索引 recall 与最终生成。

## 11. 内存账本

- GQA：按 KV head 比例下降。
- MLA：存 latent/rope cache，通常远低于完整 K/V。
- Linear：保存 recurrent state；若混合层仍需部分 KV。
- Sparse indexer：额外 index cache 与 scale，但主 Attention 读取更少 KV。

不能只使用统一 `KV_bytes/token` 公式估算混合模型；应按 layer type 求和。

## 12. Overlap 与通信

Indexer 计算可和部分主 Attention 准备流水；候选确定后 gather 是硬依赖。MLA 的低秩投影可融合，线性层 state update 是严格时序依赖，跨 token overlap 受限。

并行下稀疏候选可能跨 KV shards，需要交换 index score/top-k 或拉取远端 KV。省下的 dense Attention 必须覆盖候选通信。

## 13. 验证矩阵

- cache shape 与模型 config 一致。
- 每 layer type 的 dispatch 与容量估算一致。
- Prefill、chunked、decode、prefix restore、spec rollback。
- 不同 TP/DP/PD 组合。
- 长上下文边界和 state dtype。
- sparse top-k recall、量化前后候选重合率。
- fused/custom kernel 与 reference 输出。

## 14. 结论

Attention 架构加速的收益通常比局部 kernel 更根本：它直接改变每 token 必须保存或读取的信息量。但也因此最不能靠运行时猜测，必须从 checkpoint 的模型结构贯穿到 cache shape、调度状态和分布式传输。

