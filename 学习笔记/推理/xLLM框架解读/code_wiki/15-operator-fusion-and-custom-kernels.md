# xLLM 算子融合与专用 Kernel：从少 Launch 到少 HBM 往返

## 1. 融合为什么能加速

两算子串行时，中间 tensor 常经历：写 HBM -> 下一 kernel 再读 HBM，并付两次 launch。融合后中间值可留在寄存器/片上缓存：

```text
T_saved ≈ launch_overhead + intermediate_HBM_write_read
```

它对 Decode 小 batch 的 launch-bound 路径和带宽受限 elementwise 链尤其重要。

## 2. 当前配置入口

`core/framework/config/kernel_config.h` 包括 NPU 侧：

- `enable_customize_mla_kernel`
- `npu_kernel_backend`：AUTO/ATB/TORCH 等
- `enable_intralayer_addnorm`
- `enable_interlayer_addnorm`
- `enable_fused_mc2`
- `enable_split_rmsnorm_rope`
- `enable_aclnn_matmul`、`enable_aclnn_swiglu`
- `enable_flashcomm1`
- `enable_mmrs_fusion`、`mmrs_comm_mode`

此外各 CUDA/MLU/MUSA/DCU backend 有自己的 FlashAttention、quant、MoE、RoPE 等 kernel 选择。

## 3. AddNorm 融合

Transformer 层频繁出现 residual add + RMSNorm。融合可一次读 hidden/residual，写 normalized output 和更新 residual，避免独立 add tensor。

intralayer 与 interlayer 的区别在融合边界是否跨子层/decoder layer。跨层融合对 buffer ownership 和 Graph 更敏感，因为输出同时是下一层输入。

## 4. RMSNorm + RoPE/Quant

`split_rmsnorm_rope` 针对 Q/K 投影后的规范化与旋转位置编码布局，可能减少 split/reshape/materialization。FP8 路径还有 fused RMSNorm + static FP8 quant，让下一个 linear 直接消费 FP8。

融合必须保持 epsilon、RoPE layout/interleaving、position id 和 scale 完全一致。

## 5. Matmul + Collective

MC2/MMRS 将矩阵乘和 all-reduce/reduce-scatter 耦合，使 partial output 不必完整落 HBM，并可分 tile 进入通信。FlashComm 专题中的 MMRS 是典型例子。

收益高度依赖消息大小、TP 拓扑、dtype、对齐和通信 backend，fallback 路径不可省。

## 6. Attention 专用 kernel

平台 backend 将 QK、mask、softmax、PV、Paged KV gather 等融合。Prefill 和 Decode shape 不同，通常使用不同 kernel；MLA、sparse、quantized KV 又有专用变体。

`enable_customize_mla_kernel` 只是在满足模型/平台/shape 时选择自定义路径，batch builder 也可能因此物化不同 metadata。它不是局部函数替换。

## 7. MoE 专用 kernel

Fused top-k、dispatch/combine、grouped GEMM、SwiGLU、dequant/requant 可减少 token permutation 和中间 tensor。由于 expert token 数动态，workspace 与容量上界必须保守规划。

某些通信融合在 ACL Graph 下明确禁用或走 fallback，组合测试不可缺。

## 8. ATB、Torch 与设备 Graph 的层次

- Torch/ATen 表达 tensor op，便于组合和 fallback。
- ATB 组装 NPU 高性能算子图并管理特定并行域。
- 自定义 aclnn/CUDA 等 kernel 实现具体融合。
- 设备 Graph 捕获一整段 launch 序列，减少 host 开销。

融合减少图内节点和内存流量；Graph 减少重复提交。两者互补，但自定义 op 必须可捕获、地址稳定、无非法 host callback。

详细组图链路见 [03](03-graph-execution-complete-chain.md)。

## 9. 通信影响

纯 elementwise fusion 不改变跨卡 bytes。Matmul-collective、MoE dispatch fusion 可减少中间驻留和启动，并可能更早发送 tile，但 collective 数学 payload 常近似不变。

不要把 timeline 中 kernel 名减少误读为网络字节减少。

## 10. Overlap

融合可能增加单 kernel 持有的计算资源，反而压缩通信并发；也可能通过 tile pipeline 增强 overlap。是否更好取决于 device scheduler 和通信引擎。

Graph capture 下 stream/event 行为还可能与 eager 不同，必须实际 profile。

## 11. 内存影响

通常减少中间激活和 allocator 压力，但 fused kernel 可能申请更大 workspace、固定最大 shape buffer 或 padding。峰值需比较完整 forward，不看单个 tensor。

## 12. 精度影响

融合目标是数学等价，实际累加顺序、近似激活函数、softmax precision 和量化边界可能变化。每个 fused path 都应与 reference path 做：

- 多 shape/dtype logits 容差。
- NaN/Inf、极值与空 batch。
- 长上下文 mask/position。
- greedy 生成一致率。
- 量化 scale 与 ignored modules。

## 13. 回退设计

成熟 kernel 选择必须是 capability predicate：backend、dtype、shape、alignment、model、Graph mode、并行度都满足才进入；否则回 reference。

性能优化最危险的不是回退慢，而是“不支持的 shape 仍进入快路径并静默错算”。

## 14. 性能分析方法

1. 先确认真实命中 kernel，而非只看开关。
2. 比较 launch 数、HBM bytes、workspace。
3. Prefill/Decode、短/长序列分别测。
4. 检查与 Graph、TP、quant、chunked 的组合。
5. 用端到端 TTFT/TPOT 证明收益，不只看 kernel latency。

## 15. 结论

算子融合优化的是数据生命周期：尽量在片上完成相邻计算，少落一次 HBM、少发一次 launch。它必须建立在完整 fallback 与数值对照上，不能用平台特化换掉系统正确性。

