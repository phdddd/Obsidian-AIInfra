# xLLM 量化推理全解：权重、激活、KV 与 Indexer Cache

## 1. 量化不是一个开关

xLLM 至少有四个不同量化对象：

- 权重：降低模型常驻内存和权重读取带宽。
- 激活：让 W8A8/FP8 GEMM 使用低精度输入。
- KV Cache：降低每 token 缓存字节数。
- Lightning Indexer Cache：降低稀疏索引器状态。

它们的配置来源、kernel、scale 粒度、精度风险和兼容矩阵都不同。

## 2. 当前源码入口

- `core/framework/quant_args.h`：quant method、bits、group size、dynamic/static、ignored modules、per-weight 描述。
- `core/framework/hf_model_loader.cpp` 与 model loader：解析 checkpoint 量化元数据。
- `core/layers/common/linear.cpp`：W8A8、FP8 权重加载和 forward 分派。
- 平台 quantize/quant_matmul/scaled_matmul kernels。
- `core/framework/kv_cache/quantized_kv_cache_impl.*`：KV 与 K/V scale tensor。
- `core/runtime/worker_impl.cpp`：KV/indexer quant 的能力校验。

## 3. 支持格式不是全局枚举

`QuantArgs` 识别 FP8、SmoothQuant、Ascend INT4/INT8，以及 per-weight `w8a8`、`w8a8_dynamic` 描述；具体模型与平台只实现其子集。`ignored_modules` 允许路由 gate、LM head 或敏感投影保持高精度。

正确判断方式是：checkpoint 元数据 -> loader 解析 -> 目标 layer 分支 -> 平台 kernel。只看到 `quant_method` 字符串不等于所有层都量化。

## 4. Weight-only 与 W8A8

Weight-only 主要减少权重 HBM 流量，激活仍为 BF16/FP16，矩阵乘中解码/反量化权重。

W8A8 同时量化权重和激活：

```text
x_q = clamp(round(x / s_x))
w_q = clamp(round(w / s_w))
acc = GEMM_int8(x_q, w_q)       // usually wider accumulator
y   = acc * s_x * s_w
```

static activation scale 来自校准，运行快但分布漂移风险高；dynamic scale 每批/每 token 计算，适应性更好但增加 reduction 和 quantize kernel。

## 5. SmoothQuant 与混合精度

SmoothQuant 将 activation outlier 的尺度迁移到 weight，使激活更适合 INT8。MoE 可出现 routed expert W4A8、其他层 W8A8/BF16 的混合。

融合 QKV/gate-up 权重前，各 partition 的 quant type、scale 语义必须一致。源码在 merged prefix 查询中拒绝“部分解析或方法不一致”，这是防止静默错算的关键校验。

## 6. FP8 路径

FP8 常用 E4M3，动态范围与精度和 INT8 不同。`linear.cpp` 支持输入量化 + scaled matmul，也能消费前一 fused RMSNorm 直接产生的 FP8 tensor，省掉一次单独量化和 HBM 往返。

融合多个 FP8 partition 时，源码可能先恢复到高精度再以统一 global scale 重新量化，保证融合 tensor 使用一致 scale。这个加载期转换也需纳入 checkpoint parity 测试。

## 7. KV Cache INT8

普通 KV 每 token 字节近似：

```text
2 × layers × kv_heads × head_dim × bytes(dtype)
```

BF16 到 INT8 理论 KV 本体减半，但要加 K/V scale tensors。`QuantizedKVCacheImpl::swap_blocks` 在搬 block 时同时复制 scale，说明 scale 是缓存语义的一部分。

当前源码明确限制：KV INT8 只支持 MLU backend，并且不支持 VLM 和 PD disaggregation。其他后端会 `LOG(FATAL)`，不能靠传参强开。

## 8. Indexer Cache INT8

带 Lightning Indexer/稀疏注意力的模型还有 index cache 与 scale shape。worker 会检查模型推导的 cache shape 是否与 `indexer_cache_dtype == int8` 一致。

它降低索引状态内存，但可能改变 top-k 候选，误差会被离散选择放大，因此要单独评估 recall/top-k 重合度，而不只是最终 logits 均方误差。

## 9. 通信影响

权重通常不跨 rank 每步传输，因此量化主要降低 HBM 而非 collective payload。激活若保持低精度跨 collective，通信可下降；若 collective 前已反量化则不会。

MoE dispatch 是否传 BF16 还是低精度 token 取决于融合协议。PD KV 若支持量化传输可降低网络字节，但当前 KV INT8 与 PD 显式不兼容。

## 10. Overlap

动态 activation quantization 是额外前置计算，可与前一独立路径融合或流水。fused RMSNorm+FP8、dequant+SwiGLU+requant 的价值主要是少 launch、少 materialization，而非改变 GEMM FLOPs。

反量化若成为独立 kernel，可能抵消低精度 GEMM 收益；必须看端到端 layer timeline。

## 11. 内存账本

收益：低精度权重、KV/Indexer 本体。

成本：scale、zero-point、dequant scale、packed layout、临时量化 tensor、校准元数据、kernel workspace。动态量化还会短时同时持有高精度 input 和低精度 copy。

因此 8 bit 不保证峰值恰好减半，但权重/KV 稳态常接近该数量级。

## 12. 精度风险排序

通常可按以下顺序审视，而非当作绝对规律：

1. 权重 FP8/INT8，敏感层保高精度。
2. W8A8 dynamic。
3. W8A8 static，依赖校准分布。
4. INT4/更低权重。
5. KV INT8，误差会随长上下文参与每步 Attention。
6. Indexer quant，可能改变离散候选集。

实际结果由模型训练方式和 checkpoint 决定。

## 13. 质量评估

- perplexity/任务集准确率。
- greedy token 一致率与首个分叉位置。
- 长上下文不同长度 bucket。
- tool call/JSON 格式成功率。
- MoE router top-k 一致率。
- Indexer candidate recall。
- 多轮 decode 的误差累积。

只比较几条聊天样例不足以验收量化。

## 14. 性能评估

- 权重 HBM 与总峰值。
- KV bytes/token 与最大并发 token。
- quant/dequant kernel 时间。
- 低精度 GEMM 是否真实命中专用 kernel。
- Prefill/Decode 分开测，因为前者算力密集、后者权重带宽敏感。
- 小 batch 与大 batch 交叉点。

## 15. 结论

量化的工程本质是重新分配误差预算，以换取 HBM 容量、内存带宽和低精度算力。xLLM 的支持是 model/backend/layer/dtype 四维矩阵，部署结论必须来自真实运行分支和质量测试。

