# Small-1G · FLOPs 审计（真实前向，2×MACs）

方法：`flops_audit.py` 在真实前向上挂钩 `nn.Linear` 与 `nn.MultiheadAttention`。
Linear 计 `2·in·out·B·T`；MHA 计 qkv 投影 + QKᵀ + AV + 输出投影。
**校验**：同一计数器对原 SmallPreset 得到 **4.954039 GFLOPs**，与 Stage 16 preflight
实测值逐位一致，说明口径正确。

| 模型 | 参数量 | 总 FLOPs | Token MLP 占比 | 10 TFLOPS 时延 |
|---|---:|---:|---:|---:|
| Small-5G（原） | 61,121,028 | **4.954039 G** | 4.9325 G（99.6%） | 0.4954 ms |
| **Small-1G（新）** | **20,740,996** | **0.997762 G** | 0.9762 G（97.8%） | **0.09978 ms** |

与预期一致（20.7 M / 0.998 G / 0.100 ms）。计算量几乎全部来自 token MLP
`2048→2432→2048 × 49 tokens`，其余模块合计约 0.022 G。

逐模块明细见 `flops_audit.json`。
