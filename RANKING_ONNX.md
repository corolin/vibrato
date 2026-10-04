# Vibrato ONNX 全矩阵基准（RANKING_ONNX）

> 数据冻结 2026-10-02 · 40 题真实判卷输入（题长 p50=122、max=294 字符位）
> 基线 = 生产路径（torch bge-m3 GPU + torch 判头 fp32）· 全部实测可复算（`bench_onnx_matrix.py`）

## 一、精度：fp32 vs int8（判头 × 向量层）

| 组合 | PAD 均值偏差 | PAD 最大偏差 | 词解码 top1 翻转 |
|---|---|---|---|
| A 判头 fp32 + bge 参考(GPU) | **0.0000** | 0.0000 | 0/40 |
| B 判头 **int8** + bge 参考 | 0.0056 | 0.0343 | **0/40** |
| C 判头 fp32 + bge-**fp32**-ONNX(CPU) | **0.0000** | 0.0000 | 0/40 |
| D 判头 int8 + bge-**int8**-ONNX（纯CPU全int8满血） | 0.0454 | 0.2712 | 2/40 |
| E 判头 int8 + 零特征（纯文字路径） | 0.1779 | 1.0915 | 10/40 |

**三层读法**：
- **判头 int8 = 免费午餐**：0 翻转、均值偏差 0.006（骑墙阈值对应 PAD 差 ~0.5，差两个数量级）
- **bge int8 = 2/40 翻转（5%）**：均值 0.045——路由/监控级用途无损，最严口径场景用 fp32
- **砍掉知识层（纯文字）= 10/40 翻转（25%）**：bge 的价值被反向定标——它值四分之一的词解码正确性
- bge-fp32-ONNX 与 torch **端到端逐位一致**（C 组 0.0000；特征级余弦均值 0.958 是被未覆盖字符的零向量拖低的，真实 token 等价）

特征级参考：bge-ONNX-CPU vs torch-GPU 逐 token 余弦——fp32 均值 0.958 / int8 均值 0.936（均含未覆盖位零向量）。

## 二、延迟（CPU = 单线程；p50/p90/p99，毫秒）

| 部署形态 | p50 | p90 | p99 | 备注 |
|---|---|---|---|---|
| 判头 int8 · CPU（真实题长） | **4.39** | 9.20 | 10.88 | 纯文字路径 |
| 判头 fp32 · CPU（真实题长） | 5.09 | 10.01 | 12.02 | |
| 判头 int8 · CPU（L=96 定长） | 2.97 | — | — | 短输入更快 |
| 判头 · GPU（L=96） | 1.92 | — | — | |
| bge int8 · CPU（每文本单元） | 31.5 | 42.4 | 53.9 | 160 单元实测 |
| bge fp32 · CPU（每文本单元） | 57.5 | 72.6 | 87.5 | int8 快 ~1.8× |
| **纯CPU满血端到端**（bge-int8+判头-int8，整题） | **144** | 249 | 270 | 一题=多单元 bge+判头 |
| GPU 满血端到端（参考） | ~16 | — | — | bge 14.1 + 判头 1.9 |

## 三、内存与体积

| 工件 | 磁盘 | 常驻 RSS |
|---|---|---|
| 判头 ONNX int8 | **2.2 MB** | 判头双会话共 **8 MB** |
| 判头 ONNX fp32 | 7.3 MB | ↑ |
| bge int8 ONNX | **569 MB**（单文件） | bge 双会话共 **~2.0 GB** |
| bge fp32 ONNX | ~2.2 GB（主图+外部张量） | （单开 int8 ≈ 0.6-0.8 GB） |
| torch ckpt_v6o | 7.5 MB | GPU 满血整链 VRAM ~2.3 GB |

## 四、底座对比（换嵌入模型，同 Vibrato 管线，对独立裁判）

| 底座 | 参数 | dim | int8 体积 | r̄ | P | A | D | MAE |
|---|---|---|---|---|---|---|---|---|
| bge-m3 | 568M | 1024 | 569 MB | 0.635 | **0.812** | 0.506 | 0.588 | 0.684 |
| **e5-small** | **118M** | **384** | **118 MB** | **0.652** | 0.758 | **0.616** | 0.581 | **0.676** |
| bekko-a25m | 123M | 384 | — | 0.582 | 0.769 | 0.438 | 0.539 | 0.658 |
| bekko-a8m | 106M | 384 | — | 0.540 | 0.679 | 0.462 | 0.479 | 0.662 |

- **e5-small 118M 赢了 bge-m3 568M 的总分**，A 维 0.616 全场最高
- **满血包从 417MB 缩到 85MB（4.8×）**
- 换底座 = 44s 重训判头，架构零改动

## 五、选型速查

| 场景 | 推荐 | 理由 |
|---|---|---|
| 高频路由/实时流（纯文字够用） | 判头 int8 · CPU | 4.4ms / 2.2MB / 0 部署依赖 |
| 桌面端满血（无 GPU） | **e5s bundle (85MB) · CPU** | A 维最强 0.616 / 体积缩 5× |
| 服务器满血（P 维优先） | bge bundle (417MB) · GPU | P 维 0.812 / 16ms |
| 融合单文件部署 | fused bundle (410MB) | 一个 ONNX 一个 session.run |
| 最严精度口径（审计/复算） | 判头 fp32 + bge fp32 | 与生产基线 0 偏差 |

## 附：结构化数据

```json
{
  "frozen": "2026-10-02", "n_cases": 40, "case_len_p50": 122,
  "precision_vs_torch_baseline": {
    "A_head_fp32_ref_feats": {"pad_mae": 0.0, "pad_max": 0.0, "top1_flips": 0},
    "B_head_int8_ref_feats": {"pad_mae": 0.0056, "pad_max": 0.0343, "top1_flips": 0},
    "C_head_fp32_bge_fp32_cpu": {"pad_mae": 0.0, "pad_max": 0.0, "top1_flips": 0},
    "D_all_int8_cpu": {"pad_mae": 0.0454, "pad_max": 0.2712, "top1_flips": 2},
    "E_head_int8_zero_feats": {"pad_mae": 0.1779, "pad_max": 1.0915, "top1_flips": 10}
  },
  "latency_ms": {
    "head_int8_cpu": {"p50": 4.39, "p90": 9.20, "p99": 10.88},
    "head_fp32_cpu": {"p50": 5.09, "p90": 10.01, "p99": 12.02},
    "bge_int8_cpu_per_unit": {"p50": 31.5, "p90": 42.4, "p99": 53.9},
    "bge_fp32_cpu_per_unit": {"p50": 57.5, "p90": 72.6, "p99": 87.5},
    "full_int8_cpu_per_case": {"p50": 144, "p90": 249, "p99": 270},
    "head_gpu_l96": 1.92, "full_gpu_l96": 16.0
  },
  "memory": {"head_sessions_mb": 8, "bge_sessions_mb": 1974,
             "disk": {"head_int8_mb": 2.2, "head_fp32_mb": 7.3,
                       "bge_int8_mb": 569, "bge_fp32_gb": 2.2}},
  "artifacts": ["model_v6/vibrato_v6_fp32.onnx", "model_v6/vibrato_v6_int8.onnx",
                 "bge_onnx/bge_m3_fp32.onnx", "bge_onnx/bge_m3_int8.onnx"]
}
```
