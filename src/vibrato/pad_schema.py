# -*- coding: utf-8 -*-
"""Vibrato(颤音) PAD 契约 — 冻结的输出刻度与换算(单文件自包含)

Vibrato 是 Chordia(弦音) 生态的情绪前置网络: 输入消息+上轮PAD回声,
输出 P/A/D 三维 × 9档序数bin分布 + 全局置信度。压力值不进网络——
按 Chordia 标准由 ΔPAD 派生(本文件提供参照实现)。

刻度对齐(勿改, 改了就是与 chordia-engine 的契约破裂):
- chordia pad_calculator.py: 18词项李克特 1-9 分; P=mean(I1..I6), A=mean(I7..I12),
  D=mean(I13..I18); 标准化 = (均值-5)/4 → [-1,1]
- 本契约: 每维直接输出 9 档 bin 分布, 期望值 E[bin] ∈ [1,9] 等价于该维 6 词项均值,
  过同一标准化公式。chordia 历史 LLM 分析器的 18 项输出可用 items_to_bins()
  无损(核平滑)转成训练软标签。

echo 状态(v5 已废): 上轮 P/A/D 回声 3 浮点已溶解进 v6 消息范式——调用方把上轮
输出写回上一条 user 消息的 pad 字段即是回声(vib_messages.py)。
契约指纹 contract_digest() 训练前后/部署两侧比对, 不一致即停; v6 起指纹含
上下文窗口与 affect 消化能力(vib_messages.CONSUMED_FIELDS)。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Sequence

from . import vib_messages

# ── 冻结刻度 ────────────────────────────────────────────────
DIMS: List[str] = ["pleasure", "arousal", "dominance"]
N_BINS = 9                          # 李克特 1..9, bin i ↔ 分值 i+1
BIN_CENTERS = list(range(1, 10))    # 1..9
ECHO_LEN = 3                        # v5 legacy: v6 起由消息范式状态伪token取代(仅桥接旧数据)

# chordia pad_calculator.py 的 18 词项分组(只取分组索引, 词项原文在
# chordia-engine/prompts/emotion_analysis.md, 不在此复制)
ITEM_GROUPING: Dict[str, List[str]] = {
    "pleasure": [f"I{i}" for i in range(1, 7)],
    "arousal": [f"I{i}" for i in range(7, 13)],
    "dominance": [f"I{i}" for i in range(13, 19)],
}
LIKERT_MIN, LIKERT_MID, LIKERT_HALF = 1.0, 5.0, 4.0


def expected_from_bins(dist: Sequence[float]) -> float:
    """bin 分布 → 李克特期望分值 ∈ [1,9]。"""
    if len(dist) != N_BINS:
        raise ValueError(f"bin 分布长度应为 {N_BINS}, 得到 {len(dist)}")
    total = float(sum(dist))
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"bin 分布未归一化: sum={total}")
    return sum(p * c for p, c in zip(dist, BIN_CENTERS))


def normalize_likert(expected: float) -> float:
    """李克特期望值 → [-1,1], 与 chordia (均值-5)/4 同式。"""
    return max(-1.0, min(1.0, (expected - LIKERT_MID) / LIKERT_HALF))


def bins_to_pad(bins: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """{dim: 9档分布} → {pleasure, arousal, dominance} 归一化 PAD。"""
    return {dim: round(normalize_likert(expected_from_bins(bins[dim])), 6) for dim in DIMS}


def likert_to_bins(value: float, half_width: float = 1.0) -> List[float]:
    """单个李克特分值 → 9档帐篷核分布(items_to_bins 的单值内核)。

    老数据桥接用: chordia 60k 的 PAD([-4,4]) 换算 L = pad + 5 即落 [1,9]。
    """
    if not (1.0 <= value <= 9.0):
        raise ValueError(f"李克特分越界: {value}")
    acc = [max(0.0, 1.0 - abs(bc - value) / half_width) for bc in BIN_CENTERS]
    total = sum(acc)
    return [v / total for v in acc] if total > 0 else [1.0 / N_BINS] * N_BINS


def items_to_bins(scores: Dict[str, float], half_width: float = 1.0) -> Dict[str, List[float]]:
    """chordia 18 词项评分 → 3×9 软分布。

    half_width=1.0(默认)是帐篷核: 整数分值落单 bin, 带小数分值在相邻两 bin
    按线性比例插值——**精确保留期望值**(两位小数分数的插值即其自然不确定性)。
    同维 6 个核平均后归一化: 词项一致时分布尖锐, 分歧时展宽, 不确定性留在
    分布形状里。half_width>1 会附加平滑但引入期望偏差(O((h-1)/4)), 谨慎用。
    """
    out: Dict[str, List[float]] = {}
    for dim, items in ITEM_GROUPING.items():
        acc = [0.0] * N_BINS
        n_used = 0
        for key in items:
            if key not in scores:
                continue
            center = float(scores[key])
            for i, bc in enumerate(BIN_CENTERS):
                d = abs(bc - center)
                if d < half_width:
                    acc[i] += 1.0 - d / half_width
            n_used += 1
        if n_used == 0:
            raise ValueError(f"{dim}: 18 词项评分中无任何 {items}")
        total = sum(acc)
        out[dim] = [v / total for v in acc] if total > 0 else [1.0 / N_BINS] * N_BINS
    return out


def pad_to_pressure_delta(d_p: float, d_a: float, d_d: float) -> float:
    """压力增量参照实现(Chordia 压力值标准 v1, 2 节):

        ΔPressure_raw = 1.0×(−ΔP) + 0.8×(ΔA) + 0.6×(−ΔD)

    累积/衰减/反馈调制归宿主(chordia)——防自激红线: 累积必须用调制前的
    原始 ΔPAD, 此函数输出即为原始值。
    """
    return 1.0 * (-d_p) + 0.8 * d_a + 0.6 * (-d_d)


# ── 训练行校验 ──────────────────────────────────────────────
def validate_gold(gold: Dict) -> List[str]:
    """校验一条训练软标签 {pad_bins: {dim: 9}, echo: [3]}。返回问题列表。"""
    problems: List[str] = []
    pad_bins = gold.get("pad_bins")
    if not isinstance(pad_bins, dict) or set(pad_bins.keys()) != set(DIMS):
        problems.append(f"pad_bins 键应为 {DIMS}")
        return problems
    for dim in DIMS:
        dist = pad_bins[dim]
        if len(dist) != N_BINS:
            problems.append(f"{dim}: bin 长度 {len(dist)} != {N_BINS}")
            continue
        s = sum(dist)
        if not (0.999 <= s <= 1.001) or any(p < 0 for p in dist):
            problems.append(f"{dim}: 分布非法 (sum={s:.4f})")
    echo = gold.get("echo")              # v5 legacy 行才带; v6 行状态在消息层, 此处不查
    if echo is not None and (len(echo) != ECHO_LEN
                             or any(not (-1.0 <= v <= 1.0) for v in echo)):
        problems.append(f"echo 应为 {ECHO_LEN} 个 [-1,1] 浮点")
    return problems


def contract_digest() -> str:
    canonical = json.dumps(
        {
            "dims": DIMS,
            "n_bins": N_BINS,
            "bin_centers": BIN_CENTERS,
            "item_grouping": ITEM_GROUPING,
            "likert": [LIKERT_MIN, LIKERT_MID, LIKERT_HALF],
            "pressure_weights": [1.0, 0.8, 0.6],
            # ── v6: 消息范式消化能力(换 checkpoint 代际即变, 部署侧必须同代) ──
            "window": vib_messages.MAX_WINDOW,
            "affect_consumed": list(vib_messages.CONSUMED_FIELDS),
            "state_divisors": list(vib_messages.STATE_DIVISORS),
        },
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_ckpt(ck: Dict, hard: bool = True) -> List[str]:
    """加载侧契约闸(README 声明的执行者): checkpoint 内指纹与当前模块不一致即拒绝。

    只比对 ckpt 里存在的键(v5 老 ckpt 无 affect_schema 键 → 跳过该键)。
    部署/导出路径应 hard=True; 跨代际对照评测(如 judge 加载 v5)可 hard=False 取警告。
    返回不一致键列表(hard 且不一致时先抛 RuntimeError)。
    """
    import battery
    import vib_messages
    expect = {"pad_digest": contract_digest(),
              "battery_digest": battery.battery_digest(),
              "affect_schema": vib_messages.schema_digest()}
    bad = [k for k, v in expect.items() if k in ck and ck[k] != v]
    if bad and hard:
        raise RuntimeError(f"契约指纹不一致: {bad} — 代码与 checkpoint 代际不匹配, 拒绝运行")
    return bad


# ── 自检 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 全 5 分(中性) → PAD 归零
    neutral = {f"I{i}": 5.0 for i in range(1, 19)}
    pad = bins_to_pad(items_to_bins(neutral))
    assert all(abs(v) < 1e-6 for v in pad.values()), pad
    print("中性 18×5.0 → PAD(0,0,0) OK")

    # 2) 与 chordia PADCalculator 公式等价(词项一致时, 期望值=均值)
    rng = __import__("random").Random(7)
    worst = 0.0
    for _ in range(200):
        scores = {f"I{i}": rng.uniform(1, 9) for i in range(1, 19)}
        bins = items_to_bins(scores)
        ours = bins_to_pad(bins)
        for dim, items in ITEM_GROUPING.items():
            ref = (sum(scores[k] for k in items) / 6.0 - 5.0) / 4.0
            worst = max(worst, abs(ours[dim] - max(-1, min(1, ref))))
    assert worst < 1e-6, worst
    print(f"与 chordia 公式等价 OK (200 随机样本最大偏差 {worst:.4f})")

    # 3) 词项分歧 → 分布展宽(不确定性可见)
    agree = items_to_bins({f"I{i}": 7.0 for i in range(1, 19)})["pleasure"]
    ent = lambda d: -sum(p * math.log(max(p, 1e-12)) for p in d)
    assert ent(agree) < 0.1
    mixed = items_to_bins({f"I{i}": (7.0 if i % 2 else 3.0) for i in range(1, 19)})["pleasure"]
    assert ent(mixed) > ent(agree) + 0.5
    print(f"分歧展宽 OK (一致熵 {ent(agree):.3f} < 分歧熵 {ent(mixed):.3f})")

    # 4) 压力公式方向性
    assert pad_to_pressure_delta(-0.5, 0.3, -0.2) > 0    # 恶化方向 → 加压
    assert pad_to_pressure_delta(0.5, -0.3, 0.2) < 0     # 好转方向 → 泄压
    print("压力增量方向性 OK")

    # 5) 校验器 + 指纹
    good = {"pad_bins": items_to_bins(neutral), "echo": [0.0] * 3}
    assert validate_gold(good) == []
    bad = {"pad_bins": {"pleasure": [0.2] * 9, "arousal": [1 / 9] * 9, "dominance": [1 / 9] * 9},
           "echo": [0.0, 2.0, 0.0]}
    assert len(validate_gold(bad)) >= 2
    print("validate_gold OK;  contract_digest:", contract_digest())
    print("✓ pad_schema 自检全部通过")
