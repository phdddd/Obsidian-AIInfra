# xLLM DiT 推理加速：Cache、Sequence Parallel、稀疏 Attention 与 VAE Tiling

## 1. DiT 与 LLM 的瓶颈不同

DiT 在多个 diffusion step 上反复执行大 Transformer，输入 token 来自图像/视频 latent；没有自回归 KV 逐 token decode。它的加速重点是跨 step 复用、空间/序列并行、稀疏 Attention 和 VAE 峰值内存。

当前统一配置在 `core/framework/config/dit_config.*`，模型支持仍需按 pipeline 检查。

## 2. DiT Cache 统一接口

`core/framework/dit_cache` 提供 `on_before_step/on_after_step` 与 `on_before_block/on_after_block` hook。`DiTWorkerImpl` 根据 `dit_cache_policy` 构造：

- `None`
- `FBCache`
- `TaylorSeer`
- `FBCacheTaylorSeer`
- `ResidualCache`

这些是近似跳算方法，会影响生成质量，不能按 LLM Prefix Cache 的“精确复用”理解。

## 3. FBCache

FBCache 在第一个 Transformer block 后计算当前 residual，与上一步缓存 residual 比较。相似度满足阈值时，跳过后续重计算，将上一步完整 hidden residual 加到本步输入。

源码关键参数：warmup steps、`residual_diff_threshold`。阈值越宽松，命中越多、近似越强。

内存需保存首 block residual 和最终 hidden/encoder residual；命中后省下大部分 block 计算与其中通信。

## 4. TaylorSeer

TaylorSeer 保存 hidden 的若干阶时间导数，用 Taylor 展开预测被跳过 step：

```text
Y(t+Δ) ≈ Σ_i d^iY(t) * Δ^i / i!
```

`n_derivatives` 控制阶数，`skip_interval_steps` 决定多久做一次真实 forward 更新导数。源码按真实 step 间隔计算差分，并在 warmup 内禁用缓存。

高阶保存更多与 hidden 同形 tensor，内存随 derivative 数增长；step 间变化不平滑时预测误差增大。

当前源码还有一个需要部署侧规避的边界：`TaylorSeer::init` 只检查 `skip_interval_steps >= 0`，但 `on_before_step` 随后会对它取模；配置为 0 会触发除零。修复校验前应保证该值严格大于 0，默认值 3 不受影响。

## 5. FBCacheTaylorSeer

组合策略先用 residual 相似性判断能否跳算，再用 Taylor 近似输出。它试图兼顾自适应命中与跨 step 预测，但同时承担两类缓存内存和参数耦合。

验收应单独比较 FBCache、TaylorSeer、组合，不应假定组合必然最优。

## 6. ResidualCache

ResidualCache 只在中间 step 区间和中间 block 区间工作。真实更新轮保存入口 hidden，并在目标结束 block 保存跨 block residual；跳算轮直接跨过中间 blocks 后加回 residual。

`start/end steps` 保留扩散早期与末期关键步骤，`start/end blocks` 保留网络两端，`skip_interval` 决定更新频率。

它省下的是中间 block 子图，而非整个 DiT step。

## 7. DiT Cache 精度与内存

这是明确的质量交换：缓存 residual/导数不是当前 step 的真实 Transformer 输出。需要用任务对应指标：图像 FID/CLIP/人工偏好，视频时序一致性、运动质量、细节和 prompt adherence。

内存成本可达若干完整 hidden/encoder tensor；高分辨率/长视频时不可忽略。跳算减少激活临时峰值和 collective 次数，但缓存本身跨 step 常驻。

## 8. Sequence Parallel

DiT token 数随图像面积/视频帧数很大，SP 沿 sequence 维分片 Attention/MLP 激活。`dit_parallel_mixin`、parallel linear、sequence pad manager 和 DiT collective communicator 组织布局与通信。

SP 降每卡激活，代价是 all-to-all/all-gather/reduce-scatter；非整除 token 需要 padding。`dit_sp_communication_overlap` 尝试将这些 collective 与独立计算重叠。

## 9. SP 通信 Overlap

与 LLM 多流类似，只有后续计算不依赖 collective 完整输出时才能隐藏。DiT 大 token/GEMM 提供较长窗口，但视频 tensor 大，通信也可能饱和链路。

验证必须看 timeline 中 collective 与 compute 的并发，而非仅看开关。padding 比例、跨节点拓扑与 Graph 会改变交叉点。

## 10. Sparse Attention / RainFusion

WAN pipeline 支持两类版本：

- `rain_fusion`：frame pairing + `aclnnRainFusionAttention`。
- `sparse_attention`：block decomposition + `aclnnBlockSparseAttention`。

配置包括 sparsity、pool size、开始 step、mask refresh interval。稀疏度 0.5 表示丢弃约 50% block；越晚开始通常更保守。

## 11. 稀疏 mask 的成本

生成 block mask 本身需要 pooling/score 和 metadata。`mask_refresh_steps > 1` 可复用 cached int8 mask，减少预处理但使选择更陈旧。

收益条件：省下的 Attention 计算/读写 > mask 构建 + 不规则 kernel + padding。短序列或低稀疏度可能无收益。

它是算法近似，应观察细节、运动和跨帧一致性，不只看 latency。

## 12. Laser Attention

`dit_laser_attention_enabled` 是另一模型/平台专用 Attention 加速路径，并与 WAN sparse attention 在文档中存在互斥。真实支持要从 pipeline 分支与 kernel availability 判断，不能同时强开两个假设不同稀疏布局的实现。

## 13. VAE Tiling

VAE decode/encode 对高分辨率图像产生大 feature map。Tiling 将空间区域分块处理并在边界融合：

```text
peak_activation ≈ tile_area / full_area × original_peak + overlap_buffers
```

它主要降峰值内存，可能增加重复边界计算、kernel 次数和总延迟。当前配置注释与文档显示支持范围有限，例如 Qwen Image Edit Plus 路径显式读取开关。

质量风险是 tile seam/边界不连续，需检查 overlap 和融合窗口。

## 14. CFG/TP/SP/VAE 并行

DiT 还有 CFG parallel、Transformer TP、SP、VAE parallel 等维度。CFG 的正/负条件分支天然可并行，但会复制部分权重/激活；TP 切权重，SP 切 token，VAE 切后处理工作。

选型应根据 transformer 权重、latent token、VAE 峰值和节点拓扑，而不是把所有 degree 同时增大。

## 15. 通信与内存总表

| 方法 | 通信变化 | 常驻/峰值内存 | 精度 |
|---|---|---|---|
| DiT Cache | 命中时减少被跳层 collective | 增加 residual/导数缓存，减少临时激活 | 近似 |
| SP | 增加布局 collective | 降每卡序列激活，增通信 buffer | 数学等价、浮点漂移 |
| Sparse Attention | 可能减少 Attention 读写，增加 mask 元数据 | mask cache/workspace | 近似 |
| VAE Tiling | 通常无设备间通信 | 降峰值，增 tile overlap buffer | 边界可能影响质量 |
| CFG parallel | 分支间少依赖，最终需组合 | 可能复制分支资源 | 数学等价 |

## 16. 调参顺序

1. `None` cache + dense attention 建立质量/性能基线。
2. 先选 TP/SP 拓扑，使模型和激活可运行。
3. 单独打开一种 cache policy，扫 interval/threshold。
4. 单独打开 sparse attention，扫 start step/sparsity/refresh。
5. 高分辨率 OOM 再启 VAE tiling。
6. 最后做组合，并重新评估质量，近似误差可能叠加。

## 17. 观测与验收

- 每种 cache 的 hit/skipped steps/skipped blocks。
- 每 step latency 与通信 bytes。
- residual diff、Taylor 预测误差。
- sparse mask density/build time/refresh count。
- VAE tile 数、峰值内存、边界质量。
- 固定 seed 的基线输出和业务质量集。
- 不同分辨率、帧数、sampling steps 的曲线。

## 18. 结论

DiT 加速的主轴是“跨 diffusion step 少算”和“沿大 token 维并行”。它比 LLM runtime 优化更直接触碰生成质量，因此每个速度数字都必须带上分辨率、step、cache 参数、稀疏度和质量结果。
