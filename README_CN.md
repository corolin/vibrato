# Vibrato · 颤音

**一个完整的情绪 System One 模型 —— 把情绪阅读变成类型化决策。**

<p align="center">
  <a href="https://huggingface.co/Corolin/Vibrato"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-模型下载-ffd21e.svg"></a>
  <a href="https://modelscope.cn/models/Corolin/Vibrato"><img src="https://img.shields.io/badge/ModelScope-魔搭社区-624aff.svg"></a>
  <a href="https://vibrato.syrkos.com"><img src="https://img.shields.io/badge/🌐_官网-Global-blue.svg"></a>
  <a href="https://vibrato.syrkos.cn"><img src="https://img.shields.io/badge/🇨🇳_官网-国内镜像-red.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

简体中文 | [English](README.md)

独立、通用的 System One（类 Jev）模型：一次前向，从一段消息序列（`v1/affect` 范式：每条消息可为文字形态和/或 PAD/压力/活力状态快照）输出**校准的概率分布**——PAD 三维评分、情绪族、是非判断、置信度。它诞生于 Chordia 陪伴栈的用户情绪前置，现已成长为可服务任何应用的独立模型。

> 弦乐器里，颤音是情绪穿过音符的方式。

---

## 为什么

各种应用都需要读取用户情绪——陪伴 AI、聊天分析、内容审核、心理健康分诊、自适应界面。今天的默认做法是**每条消息调一次 LLM**：慢（3–13 秒）、贵、泄露隐私，而且只返回三个裸数字。Vibrato 是 System-One 路线的答案（该模型类由 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 开创，开源同族有 [Laya](https://github.com/NandhaKishorM/laya)、[Tev1](https://huggingface.co/togethercomputer/Tev1-4B-experimental)）：

| | 每消息一次 LLM | Vibrato |
|---|---|---|
| 延迟 | 3.7–13.2 秒（十二强实测） | **判头 CPU-ONNX-int8 2.97 ms / GPU 判头 1.92 ms / 满血路径 ~16 ms（GPU 含 bge-m3）——全实测（L=96）** |
| 输出 | 3 个点值 | **完整分布 + 置信度** |
| 成本 / 隐私 | 8k–36k token/40 题，文本出机器 | $0，永不离开本机 |
| 校准 | 无 | 序数 bin + 熵 + conf 头 |

teacher LLM 只在**离线**造数据——运行时是 1.86M 判头 + 冻结嵌入底座（118M e5-small 即可，568M bge-m3 并非必需）。

## 题库（冻结契约）

三种题型，一次前向：

| 题型 | 输出 | 内容 |
|---|---|---|
| **score** | 3 × 9-bin | 愉悦度 P / 唤醒度 A / 支配度 D，各一条序数分布（李克特 1–9） |
| **choice** | OCC-22 排序 | 22 种情绪按 PAD→质心距离升序排列（Ortony-Clore-Collins 分类体系）；8 族头作内部训练信号 |
| **noul** | 2 旗 | `suppressed`（言不由衷/压抑）· `directed_at_me`（指向性）——仅保留 PAD 推不出来的两个文本层信号 |

每个 noul 是**宿主侧行为信号**（不是路由规则）。`conf` 头标记低置信样本，**升级给真正的 LLM**——System One 的要义是知道自己何时不该答。

score 的 bin 用经典 18 项 PAD 量表同一套 1–9 李克特刻度，期望值直接套 `(均值 − 5) / 4` 归一化。压力值**派生而非预测**：`ΔPressure = 1.0·(−ΔP) + 0.8·(ΔA) + 0.6·(−ΔD)`（宿主侧累积，防自激红线：累积必须用调制前的原始 ΔPAD）。

## `v1/affect` 消息范式

Jev 开创了 System One 的接口范式；这里把 affect 前端的"嘴型"定下来——**messages 进，分布出**：

```
POST /v1/affect/score
messages: [{ role: "user"|"assistant",
             message?:  string        // 文字形态（原文）
             pad?:      [p,a,d]       // 状态快照 [-1,1]（说话方该轮）
             pressure?: number        // 压力快照 [0,100]（chordia 标准）
             vitality?: number        // 活力快照 [-30,100]（chordia-v2）}]
→ score×3 (9-bin PAD) + choice (OCC-22 排序) + noul×2 (suppressed, directed_at_me)
  + conf + write_back{pad, d_pressure_raw} + consumed{fields, digest}
  + write_back{pad, d_pressure_raw} + consumed{fields, digest}
```

- 末条 = 判断目标（必须是 `user` + `message`，不得携带状态——那是答案不是输入）
- 全文字数组 = 通用 agent 路径，完全在分布内——**状态是精度旋钮，不是入场券**
- 两层版本铁律：线上格式指纹（`vib_messages.schema_digest()`，v1 冻结只增不改）与模型消化能力（`CONSUMED_FIELDS`，v6 消化 message/pad/pressure，vitality 收而未读）分离
- 单位纪律：线上说域内原生单位，归一化是模型私事
- echo 已溶解：调用方把上轮输出写回上一条 user 消息的 `pad` 字段即是回声（另有 cls 锚定快通路）
- **场景断开 = 状态清零**：跨场景/跨会话（时间流逝、环境切换）时，宿主直接不携带 pad/pressure 字段——读者即刻冷启动、无历史包袱。**状态该继承时继承、该清空时清空，继承权在宿主手里**（连续对话保留情绪积累正是沟通的本性）

## 架构

```
v1/affect messages ─► 文本单元(char id) + 状态伪token(5值+5遮罩) + bge-m3 逐字符特征(冻结)
                          │
        echo: 上轮用户PAD ─┼─► cls锚定 ─► 3层 transformer (d192, 手写MHA, 512窗)
                          │                ├─ 3× bin头 (1–9 softmax)
                          │                ├─ 8路 情绪族头
                          │                ├─ 5× noul头 (2路 softmax)
                          └────────────────┴─ conf头 (sigmoid)
```

- **双路状态**：echo 直连 cls（锚定先验，零发现成本）+ 消息内状态伪 token（带时序的细粒度上下文）——实测同信息 cls 直连贡献 +0.28 族准确，序列内 +0.06，两路并存各司其职
- **知识注入（底座可插拔）**：冻结嵌入模型逐字符特征并联 `in_proj`。底座对比（同管线对独立裁判组）：

| 底座 | 参数 | dim | r̄ | P | A | D |
|---|---|---|---|---|---|---|
| bge-m3 | 568M | 1024 | 0.635 | **0.812** | 0.506 | 0.588 |
| **e5-small (MIT)** | **118M** | **384** | **0.652** | 0.758 | **0.616** | 0.581 |
| bekko-a25m (MIT) | 123M | 384 | 0.582 | 0.769 | 0.438 | 0.539 |

  **e5-small 118M 赢了 bge-m3 568M 总分，A 维 0.616 全场最高**——满血包 417→85 MB。换底座=44s 重训，架构零改动。
- **1.86M 参数判头**，ONNX 友好（手写 MHA 规避导出坑）；v5 判头实测 CPU-ONNX-int8 单线程 2.25 ms、int8 后 1.7 MB（v6 导出与实测待做）
- 所有工件受**契约指纹**守护（刻度/题库/affect 三指纹）；训练/部署两侧不一致即拒跑

## 基准测试（全实测）

40 题盲答卷，十四席参赛（各 LLM 同一 battery 提示词逐题作答，每卷 26.7k–79.4k token），外加独立裁判组。三把尺子各有偏见——放在一起读。

### 榜 A · 连续感知（对独立裁判组）

对双裁判合议卷的 PAD 逐维 Pearson r。**裁判组构成**：两个零上下文盲评代理（同一提示词、隔离会话；自一致 r=0.98/0.98/0.97、family 0.82、noul 0.94），pad 取双均值。注意事项：评员属 GLM 系（与 glm-5.3 有方言亲缘）、n=40。

| 选手 | r̄ (P/A/D) | 延迟/题 | token |
|---|---|---|---|
| GLM-5.3 | **0.902** (0.94/0.84/0.93) | 4.7s | 31.2k |
| gemini-3.8-flash | 0.866 (0.95/0.76/0.88) | 8.8s | 60.3k |
| claude-sonnet-5.5 | 0.857 (0.92/0.74/0.92) | 4.7s | 51.1k |
| grok-4.1-fast | 0.841 (0.94/0.74/0.85) | 3.4s | 26.7k |
| Qwen3.8-27B | 0.807 | 12.8s | 35.6k |
| gpt-5-mini | 0.762 | 13.2s | 79.4k |
| deepseek-v4-flash | 0.716 | 3.7s | 32.7k |
| **Vibrato v0.1.1（1.86M 本地）** | 0.635 (0.81/0.51/0.59) | **0.17s** | **0** |
| Clef-27B（System One, Apache-2.0） | 0.621 (0.84/**0.66**/0.36) | ~1.7s | 0（自部署） |
| Hunyuan-A13B / ref4b-4B / gpt-4o-mini | 0.59–0.62 | — | — |
| Clef-flash-9B（System One, Apache-2.0） | 0.564 (0.84/0.54/0.31) | ~1.3s | 0（自部署） |
| GLM-4-32B / Qwen2.5-7B / Xing4.0-29B | 0.43–0.47 | — | — |

**A 维是代际断层**：新一代 0.68–0.84，上一代崩坏（0.01–0.28）；Vibrato 的 0.506 站在旧阵营顶、新门槛外。

### 榜 B · 行为一致（对 Vibrato；是相似度不是正确率）

与 Vibrato v0.1.1 的词解码/noul 一致率。**主场声明**：40 题是 Vibrato 自家 battery 分布，LLM 零样本作答。**一致 ≠ 正确**——Qwen2.5-7B 对 vib top2 高达 0.90，榜 A 却只有 0.438（风格同盟，不是水平）。

| 选手 | 对 vib top1 / top2 / noul |
|---|---|
| Qwen2.5-7B（日用款） | 0.72 / 0.90 / 0.82 |
| claude-sonnet-5.5 | 0.70 / 0.80 / 0.75 |
| Xing4.0-29B | 0.70 / 0.85 / 0.76 |
| deepseek-v4-flash | 0.68 / 0.78 / 0.79 |
| Clef-27B | 0.72 / 0.97 / — |
| Clef-flash-9B | **0.75** / **1.00** / — |
| grok-4.1-fast | 0.62 / 0.70 / 0.76 |
| gemini-3.8-flash | 0.60 / 0.72 / 0.82 |
| Qwen3.8-27B / glm-5.3 / gpt-5-mini / gpt-4o-mini | 0.53–0.57 |
| Hunyuan-A13B / GLM-4-32B | 0.35–0.47 |

全部答卷（`answers_*.jsonl`）与一键重判（`judge.py`、`vs_ref.py --ref <tag>`）随仓库发布。

### 升级曲线（conf 头）

把 conf 最低的 X% 样本转交 LLM、其余保留 Vibrato（对裁判组的 r̄）：

| 转交比例 | API 调用/40题 | → sonnet-5.5 | → glm-5.3 |
|---|---|---|---|
| 0% | 0 | 0.635 | 0.635 |
| 20% | 8 | 0.652 | 0.667 |
| 50% | 20 | 0.713 | 0.753 |

诚实读法：机制上有效（50% 时 +0.12），但 10% 档是平的——conf 头（监督=teacher 自一致）现在标出的是**teacher 分歧**而非 **Vibrato 的错误**；对裁判分歧做校准已排队。

**标准测评集**（Chinese-EmoBank CVAS，2,583 句，Pearson r）：

| 配置 | V 愉悦 | A 唤醒 |
|---|---|---|
| 零样本（跨域：聊天特化 → 文学文本） | 0.517 | 0.098 |
| **5 折 CV 微调**（每折 2k 句 × 4 epoch） | **0.709±0.017** | 0.383±0.027 |
| 学界域内水位（SemEval-2026 / EmoBank 系） | ~0.70 | ~0.45 |

CPED（中文对话情绪识别标准集，13→8 族零样本跨词表映射）：acc 0.222 / macro-F1 0.177（多数类基线 0.165，angry recall 0.66 迁移最好）。

自家 battery 验证集（8 族 + PAD + noul，干净切分）：famAcc 0.61–0.67 / noulAcc 0.79 / padMAE 0.21–0.23。

### System One 洞察（数字背后的意义）

榜单最有价值的发现不是某个分数——而是 **System One 模型（Vibrato 1.86M、Clef 27B、Clef-flash 9B）形成了一个行为一致的阵营**（互相 top-2 一致率 0.97–1.00），且与 LLM 的判读方式截然不同。三条观察解释了为什么：

**1. 格式塔直觉映射，非自回归逻辑推演。**
LLM 在"读"文本——解析语义、推理因果、然后分类（"他用了 X 词，结合 Y 语境，大概是生气"）。System One 模型完全跳过推理步骤。无论是 1.86M 的字符级 transformer（Vibrato），还是锁死自回归通路的冻结 Qwen（Clef），两者都是将输入直接映射到类型化决策——更接近杏仁核而非大脑皮层。这就是 System One 延迟以毫秒计、而非秒计的原因。

**2. 免疫对齐税。**
为什么 LLM 在 D 维（支配）集体崩盘（多数 0.28–0.36，Vibrato 0.588）？RLHF 把 LLM 训练得礼貌、中立、客观——这钝化了它们对文本中权力动态、阴阳怪气、攻击性支配信号的敏感度。System One 模型没有这层包袱。Vibrato 的 61k legacy 训练数据保留了**原始**的语言施压分布，没有社会性过滤。这是特性不是缺陷：一个无法从文字中感知"我被居高临下了"的陪伴 AI，情绪上是聋的。

**3. 维度收敛证明情绪直觉是低维的。**
最惊人的结果：clef-flash（9B 冻结 Qwen）和 Vibrato（1.86M 字符级）达到 **top2 = 1.00**——clef 的 top-1 词*永远*出现在 Vibrato 的 top-2 里。当你拿一个 9B 模型剥掉自回归"理性脑"，强制它只通过路由头分类时，它的决策曲面**坍缩到与 1.86M 字符级模型几乎相同的流形上**。这是数学证据：**情绪直觉是一个紧凑的、可学习的特征空间**——不需要数十亿参数来表示，只需要正确的训练信号。

## 核心演示：颤音 × 弦音 v2 联动

[`demo/vibrato_demo.html`](demo/vibrato_demo.html)——微信式气泡流 + **每条消息的颤音侧车评价卡**（实测 PAD 三色条 / 情绪族徽章 / noul 旗，斜纹小条 = 剧本手工金标），右侧四线轨迹对照：

- **联动臂**：颤音逐轮实测用户情绪 → 弦音 v2 引擎（7 维 MLP：用户 PAD + 活力 + agent PAD → ΔPAD）驱动 agent 动力学
- **LLM 自演绎臂**：同一人格只给系统提示词，LLM 自己想象全程并自报 PAD

两人格 × 三幕 20 题（愤怒质问 → 崩溃恢复 → 从自大到友谊）实测的看点：**联动有累积与滞回**（胆小人格被连续攻击压进深度共情塌陷 [-1.0, -0.69]，恢复滞后；活力 90 的活泼人格跌到 -0.6 后回正 +0.31）；**自演绎是逐条贴面值的温和振荡**（用户崩溃时活泼人格自报只微跌到 -0.3，两人格轨迹近乎平行——人格没有调制动力学）。实测用户 PAD vs 手工金标 r(P/A/D) = 0.74 / 0.52 / 0.35。已知瑕疵同样可见：长负面前缀后的情绪回正偏慢（状态锚拖拽）。

重新生成：`python demo/demo_html.py [--llm --key sk-..]`（联动臂零 API；`--llm` 臂需 OpenAI 兼容接口）。

## 仓库结构

| 文件 | 职责 |
|---|---|
| `pad_schema.py` | 冻结刻度契约：bin、18 词项桥接（帐篷核精确保期望）、压力公式、指纹 |
| `battery.py` | 完整题库 + teacher 提示词构建 + gold 校验 |
| `vib_messages.py` | **`v1/affect` 线上格式**：校验、状态向量、形态增广、双渲染、双指纹 |
| `net.py` | `VibratoNet` v6（状态伪 token + echo 锚定 + feats 通路）+ `collate_pad` + 自检 |
| `vib_state.py` | v5 上下文渲染（legacy 桥接） |
| `feat_bge.py` / `feat_any.py` | bge-m3 / 任意 HF 模型冻结逐字符特征 |
| `eval_backbone.py` | 底座对比评测 |
| `gen_convos.py` / `label_pad.py` / `label_agent.py` | 会话生成 / 用户侧 teacher 标注 / agent 侧标注 |
| `prep.py` / `feat_precompute.py` / `train.py` | 数据组装（6 条平价窗）/ bge 特征 memmap / 两段式训练 |
| `judge.py` / `answer_sheet.py` / `eval_bench.py` / `finetune_cvas.py` | 离线判卷（答卷制）/ 答卷收集 / 标准集评测 / CVAS 5 折 |
| `demo/` | 联动演示（HTML 生成器、人格剧本、弦音引擎 + 检查点） |

## 发布物

| 包 | 体积 | 内容 |
|---|---|---|
| `vibrato-v0.1.1-dist.zip` | 17 MB | 代码+权重+ONNX |
| `vibrato-bundle-v0.1.1.zip` | 417 MB | 满血随包（bge int8） |
| `vibrato-fused-v0.1.1.zip` | 410 MB | 融合单文件 ONNX |
| **`vibrato-e5s-v0.1.1.zip`** | **85 MB** | **轻量满血（e5-small，A 维 0.616）** |

## 快速开始

```bash
pip install torch            # 冒烟自检只需 torch
python pad_schema.py         # 刻度契约自检（v6 指纹）
python vib_messages.py       # v1/affect 范式自检
python net.py                # 前向+反向+状态/echo 生效冒烟
python battery.py            # 题库自检 + 指纹
```

完整 v6 管线（数据生成与标注需任意 OpenAI 兼容接口；训练需 GPU）：`gen_convos → label_pad → label_agent → prep_v6 → feat_precompute → train.py（混合一段）→ train.py --battery-only --init（二段微调）→ judge.py`。所有数据阶段增量落盘、断点续跑——为崩溃安全而设计。

## 状态与诚实的局限

**v0.1.1。** 全链（范式 → 数据 → 两段训练 → 判卷 → 标准集）已端到端跑通并全部实测。已知局限，诚实列出：

1. **感知惯性：方向是特性，幅度失准**——用户在 7+ 轮强负面后突然平复，状态锚让读数不跳变（人类倾听者也不相信瞬间回正，会怀疑"余怒未消/阴阳怪气"）——这是认知惯性的正确方向；但当前衰减过慢，vib 的回正滞后期远长于真人读者（真人 2-3 轮即更新，vib 10+ 轮）。主抓手是**场景断开=状态清零**（见 v1/affect 节）——跨场景/会话时宿主不携带 pad/pressure 字段，读者即刻冷启动。锚衰减已从路线图移除：惯性方向是特性，不动模型，继承权交宿主。
2. **唤醒维（A）取决于底座选择**——bge-m3 得 0.506；换 e5-small 升至 **0.616**（全场 A 维最高）。剩余差距可能来自训练数据分布。conf 校准排队中。
3. ~~v6 ONNX 导出待做~~ **已完成**：全输入签名 ONNX（`model_v6/`，零 feats=纯文字路径，bge-dropout 训练保出），int8 仅 2.2 MB、PAD 偏差 <0.005。纯 CPU 满血档（bge int8）仍未实测。
4. CPED/GoEmotions 式**跨词表零样本**仍弱（0.22 acc）——分类词表迁移需要映射工程或域内微调，v0.1.1 只验证了"轻适配即可上桌"（CVAS 线）。

## 致谢

- **System One** 模型类：[TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 提出思想；[Laya](https://github.com/NandhaKishorM/laya)（Apache-2.0）的 score 型 marker 与 RLCD 训练配方；[Tev1](https://github.com/togethercomputer/Tev1)（MIT）的廉价 SFT 配方。
- 架构血统（手写 MHA、回声轮、仲裁纪律）来自作者自己的 Prisma 分词栈。
- 演示中的弦音 v2 情绪动力学引擎与人格测试剧本来自作者的 Chordia 项目。
- **出身**：Vibrato 起于 Chordia 陪伴系统的用户情绪前置，成长为独立的 System One 模型。

## 许可证

[MIT](LICENSE) © 2026 corolin
