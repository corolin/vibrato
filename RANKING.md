# Vibrato 挑战赛 · 完整榜单（落地页数据源）

> 数据冻结 2026-10-02 · 40 题盲答卷 · 全部实测可复算（`judge.py` / `vs_ref.py --ref <tag>`）
> 用法：本文档为唯一数据源，落地页所有数字以此为准，勿手抄改动。

---

## 榜 A · 连续感知（主榜，对独立裁判组）

裁判组构成：两个零上下文盲评代理（同一提示词、隔离会话），pad 取双均值。
裁判自一致：r(P/A/D)=0.98/0.98/0.97 · family 0.82 · noul 0.94。

| # | 选手 | r̄ | P 愉悦 | A 唤醒 | D 支配 | 延迟/题 | token/40题 |
|---|---|---|---|---|---|---|---|
| 1 | GLM-5.3 | **0.902** | 0.942 | 0.835 | 0.928 | 4.7s | 31.2k |
| 2 | gemini-3.8-flash | 0.866 | 0.951 | 0.763 | 0.884 | 8.8s | 60.3k |
| 3 | claude-sonnet-5.5 | 0.857 | 0.920 | 0.737 | 0.915 | 4.7s | 51.1k |
| 4 | grok-4.1-fast | 0.841 | 0.935 | 0.740 | 0.848 | 3.4s | 26.7k |
| 5 | Qwen3.8-27B | 0.807 | 0.902 | 0.734 | 0.785 | 12.8s | 35.6k |
| 6 | gpt-5-mini | 0.762 | 0.902 | 0.677 | 0.706 | 13.2s | 79.4k |
| 7 | deepseek-v4-flash | 0.716 | 0.846 | 0.527 | 0.774 | 3.7s | 32.7k |
| 8 | **Vibrato v0.1.1** | 0.635 | 0.812 | 0.506 | 0.588 | **0.17s** | **0** |
| 9 | Hunyuan-A13B | 0.619 | 0.861 | 0.284 | 0.712 | 6.7s | 33.6k |
| 10 | ref4b（本地4B教师） | 0.597 | 0.733 | 0.499 | 0.559 | 6.4s | 0（本地） |
| 11 | gpt-4o-mini | 0.589 | 0.757 | 0.284 | 0.725 | 3.9s | 31.8k |
| 12 | GLM-4-32B | 0.471 | 0.785 | 0.007 | 0.622 | 5.1s | 29.1k |
| 13 | Qwen2.5-7B | 0.438 | 0.541 | 0.260 | 0.512 | 5.0s | 30.8k |
| 14 | Xing4.0-29B | 0.427 | 0.687 | 0.103 | 0.489 | 5.8s | 30.4k |

## 榜 B · 行为一致（对 Vibrato；相似度 ≠ 正确率）

| 选手 | top1 | top2 | noul | 备注 |
|---|---|---|---|---|
| Qwen2.5-7B | **0.72** | **0.90** | 0.82 | 榜A仅0.438——风格同盟实证 |
| claude-sonnet-5.5 | 0.70 | 0.80 | 0.75 | 双榜前四，跨方言通用 |
| Xing4.0-29B | 0.70 | 0.85 | 0.76 | vib 老伙伴 |
| deepseek-v4-flash | 0.68 | 0.78 | 0.79 | |
| grok-4.1-fast | 0.62 | 0.70 | 0.76 | 双榜前四，跨方言通用 |
| gemini-3.8-flash | 0.60 | 0.72 | 0.82 | |
| Qwen3.8-27B | 0.57 | 0.72 | 0.81 | |
| glm-5.3 | 0.55 | 0.68 | 0.81 | |
| gpt-5-mini | 0.55 | 0.70 | 0.80 | |
| gpt-4o-mini | 0.53 | 0.70 | 0.83 | |
| Hunyuan-A13B | 0.42 | 0.47 | **0.83** | |
| GLM-4-32B | 0.35 | 0.47 | 0.77 | |

## 升级曲线（conf 头 → 转 LLM）

| 转交比例 | API调用/40题 | →sonnet-5.5 r̄ | →glm-5.3 r̄ |
|---|---|---|---|
| 0% | 0 | 0.635 | 0.635 |
| 20% | 8 | 0.652 | 0.667 |
| 50% | 20 | 0.713 | **0.753** |

## 成本视角（Vibrato 是唯一帕累托点）

| | Vibrato | 榜首 GLM-5.3 | 海外最快 grok-4.1 |
|---|---|---|---|
| r̄ | 0.635 | 0.902 | 0.841 |
| 延迟/题 | **0.17s** | 4.7s（28×） | 3.4s（20×） |
| token | **0** | 31.2k | 26.7k |
| 参数 | **1.86M** | 万亿级 | 万亿级 |
| 隐私 | **不出本机** | 出机器 | 出机器 |

---

## 落地页叙事锚点（建议文案角度）

1. **帕累托独一点**：中游精度（0.635，压过 4 个 27B+ 级模型）+ 断层级成本（0 token / 0.17s / 1.86M / 本地）——"不是最准的，是每单位成本最准的，且差距是数量级"
2. **A 维代际断层**：新一代 LLM 0.68–0.84 vs 上一代崩坏（GLM-4-32B 仅 0.007）；Vibrato 0.506 站旧阵营顶——可视化建议：A 维单独一张散点图，断层线一眼可见
3. **阵营即方言**：Qwen2.5-7B 对 vib 一致 0.90（榜B第一）但榜A 倒数第三——"行为一致 ≠ 判断正确"的教学案例
4. **noul 方言免疫**：路由五旗全员 0.73–0.85 挤成一团——最稳的输出维度
5. **升级机制**：conf 最低 50% 转交 glm-5.3 → r̄ 0.635→0.753，代价仅半数 API 调用

## 诚实红线（落地页必须保留的声明，防过度宣传）

- **榜 A 裁判是 GLM 系盲评代理**（与 glm-5.3 存在方言亲缘）；榜 B 是行为相似度不是正确率
- **n=40**：相邻名次差距 <0.05 属噪声，Vibrato/Hunyuan/ref4b（0.635/0.619/0.597）应表述为"同一梯队"
- **主场声明**：40 题为 Vibrato 自家 battery 分布，LLM 零样本作答
- Vibrato 对 ref4b 的高一致（0.68/0.80）含师生血统，不作优势宣传
- 不使用"击败 LLM"表述；准确口径见上文帕累托句式

## 机型对照（tag → 全名）

| tag | 全名 | 渠道 |
|---|---|---|
| glm53 | z-ai/glm-5.3 | cherryin |
| gem3flash | google/gemini-3.8-flash | cherryin |
| sonnet55 | anthropic/claude-sonnet-5.5 | cherryin |
| grok41 | x-ai/grok-4.1-fast-non-reasoning | cherryin |
| q27 | Qwen/Qwen3.8-27B | SiliconFlow |
| gpt5m | openai/gpt-5-mini | cherryin |
| dkv4 | deepseek/deepseek-v4-flash | cherryin |
| hunyuan13 | tencent/Hunyuan-A13B-Instruct | SiliconFlow |
| ref4b | qwen3.5:4b-q8_0（本地 Ollama） | 本地 |
| gpt4om | openai/gpt-4o-mini | cherryin |
| glm32 | THUDM/GLM-4-32B-0414 | SiliconFlow |
| q7 | Qwen/Qwen2.5-7B-Instruct（用户日用款） | SiliconFlow |
| xing29 | XingChenAGI/Xing4.0-29B | SiliconFlow |

## 附：结构化数据（可直接消费）

```json
{
  "frozen": "2026-10-02", "n_cases": 40,
  "referee": {"id": "zcode", "method": "dual blind raters, mean PAD",
              "self_agreement": {"pad_r": [0.98, 0.98, 0.97], "family": 0.82, "noul": 0.94}},
  "board_a": [
    {"rank": 1, "id": "glm53", "r_mean": 0.902, "r_pad": [0.942, 0.835, 0.928], "latency_s": 4.7, "tokens": 31231},
    {"rank": 2, "id": "gem3flash", "r_mean": 0.866, "r_pad": [0.951, 0.763, 0.884], "latency_s": 8.8, "tokens": 60315},
    {"rank": 3, "id": "sonnet55", "r_mean": 0.857, "r_pad": [0.920, 0.737, 0.915], "latency_s": 4.7, "tokens": 51132},
    {"rank": 4, "id": "grok41", "r_mean": 0.841, "r_pad": [0.935, 0.740, 0.848], "latency_s": 3.4, "tokens": 26718},
    {"rank": 5, "id": "q27", "r_mean": 0.807, "r_pad": [0.902, 0.734, 0.785], "latency_s": 12.8, "tokens": 35612},
    {"rank": 6, "id": "gpt5m", "r_mean": 0.762, "r_pad": [0.902, 0.677, 0.706], "latency_s": 13.2, "tokens": 79445},
    {"rank": 7, "id": "dkv4", "r_mean": 0.716, "r_pad": [0.846, 0.527, 0.774], "latency_s": 3.7, "tokens": 32745},
    {"rank": 8, "id": "vibrato", "r_mean": 0.635, "r_pad": [0.812, 0.506, 0.588], "latency_s": 0.17, "tokens": 0, "params": "1.86M"},
    {"rank": 9, "id": "hunyuan13", "r_mean": 0.619, "r_pad": [0.861, 0.284, 0.712], "latency_s": 6.7, "tokens": 33619},
    {"rank": 10, "id": "ref4b", "r_mean": 0.597, "r_pad": [0.733, 0.499, 0.559], "latency_s": 6.4, "tokens": 0},
    {"rank": 11, "id": "gpt4om", "r_mean": 0.589, "r_pad": [0.757, 0.284, 0.725], "latency_s": 3.9, "tokens": 31833},
    {"rank": 12, "id": "glm32", "r_mean": 0.471, "r_pad": [0.785, 0.007, 0.622], "latency_s": 5.1, "tokens": 29069},
    {"rank": 13, "id": "q7", "r_mean": 0.438, "r_pad": [0.541, 0.260, 0.512], "latency_s": 5.0, "tokens": 30809},
    {"rank": 14, "id": "xing29", "r_mean": 0.427, "r_pad": [0.687, 0.103, 0.489], "latency_s": 5.8, "tokens": 30374}
  ],
  "board_b_top1_vs_vib": {"q7": 0.72, "sonnet55": 0.70, "xing29": 0.70, "dkv4": 0.68,
    "grok41": 0.62, "gem3flash": 0.60, "q27": 0.57, "glm53": 0.55, "gpt5m": 0.55,
    "gpt4om": 0.53, "hunyuan13": 0.42, "glm32": 0.35},
  "escalation": {"base": 0.635,
    "to_sonnet55": {"20pct": 0.652, "50pct": 0.713},
    "to_glm53": {"20pct": 0.667, "50pct": 0.753}}
}
```
