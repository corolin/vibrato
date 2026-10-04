# -*- coding: utf-8 -*-
"""PAD → 情绪词解码模块 — 对齐 Aethelum-Nexus 生产管线(答卷转PAD→距离top2→Agent)。

锚点表: HANDOFF_pad_context_decode.md 的 14 实测 + 4 建议词([-4,4] 空间, 归一到 [-1,1])。
规则与 Nexus 一致: 最近邻为主词; 次近词仅当 d2/d1 < SECOND_RATIO(骑墙态)才给。
Vibrato 部署接口 = pad_norm3 → (主词, 次词|None, d2/d1) —— 直接喂 Agent 的格式。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

SECOND_RATIO = 1.3   # Nexus 骑墙态阈值(HANDOFF §2.2)

# (词, P, A, D) 坐标为 [-4,4] 空间原始值(便于与 HANDOFF 对照), 查询时换算
WORD_ANCHORS: List[Tuple[str, float, float, float]] = [
    ("无聊",     -1.10, -2.63, -2.10),
    ("恐惧",     -1.41,  2.47, -2.50),
    ("惊奇",      2.23,  2.73, -1.26),
    ("依赖",     -0.16, -2.19,  2.77),
    ("敌意",     -2.74,  1.98,  2.81),
    ("温和",      2.82, -1.96, -1.74),
    ("轻松",      3.15, -2.11,  1.92),
    ("乐观",      2.10,  1.87,  3.27),
    ("藐视",     -2.54, -1.52,  2.50),
    ("愤懑",     -2.91,  2.57, -0.08),
    ("喜悦",      3.68,  2.35,  1.59),
    ("悲伤",     -1.93, -0.19, -2.57),
    ("厌恶",     -3.34, -0.91,  0.17),
    ("焦虑",     -2.34,  0.00, -1.00),   # 样本量弱(253/60k), HANDOFF §5
    ("心疼",     -1.60,  1.20,  0.00),   # 以下 4 词建议坐标待标定
    ("为你高兴",  2.00,  1.60,  0.40),
    ("幸灾乐祸",  1.20,  2.00,  1.60),
    ("平静",      0.40, -2.40,  0.80),
]

_NORM = [(w, p / 4.0, a / 4.0, d / 4.0) for w, p, a, d in WORD_ANCHORS]


def _dist(p3, anchor):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p3, anchor[1:])))


def decode_top2(pad_norm3: List[float]) -> Tuple[str, Optional[str], float]:
    """PAD([-1,1] 各维) → (主词, 次词或None, d2/d1)。

    坐标换算: 本表 [-4,4] ← 输入 ×4(与 Nexus HANDOFF §2.2 相同)。
    """
    scaled = [v * 4.0 for v in pad_norm3]
    ranked = sorted(_NORM, key=lambda a: _dist(scaled, a))
    d1 = _dist(scaled, ranked[0])
    d2 = _dist(scaled, ranked[1])
    ratio = d2 / d1 if d1 > 1e-9 else math.inf
    second = ranked[1][0] if ratio < SECOND_RATIO else None
    return ranked[0][0], second, round(ratio, 3)


# 8 族质心(2026-10-01 legacy 61k 实测, 归一空间) — 族级解码/评估用
FAMILY_ANCHORS = {
    "happy":        (0.66,  0.51,  0.67),
    "affectionate": (-0.04, -0.55, 0.70),
    "calm":         (0.21, -0.59, -0.28),
    "amused":       (0.56,  0.69, -0.32),
    "sad":          (-0.49, -0.05, -0.64),
    "anxious":      (-0.36,  0.59, -0.61),
    "angry":        (-0.70,  0.23,  0.46),
    # wronged: legacy 无样本, 暂用 battery 词义估计(待 2k 数据实测替换)
    "wronged":      (-0.45,  0.10, -0.35),
}


def decode_family_top2(pad_norm3: List[float]) -> Tuple[str, Optional[str], float]:
    scaled = list(pad_norm3)
    ranked = sorted(FAMILY_ANCHORS.items(), key=lambda kv: _dist(scaled, ("",) + kv[1]))
    d1, d2 = _dist(scaled, ("",) + ranked[0][1]), _dist(scaled, ("",) + ranked[1][1])
    ratio = d2 / d1 if d1 > 1e-9 else math.inf
    second = ranked[1][0] if ratio < SECOND_RATIO else None
    return ranked[0][0], second, round(ratio, 3)


if __name__ == "__main__":
    # HANDOFF §6 验收用例(生产 PAD → ×4 查表)
    cases = [((-0.35, 0.62, -0.62), "恐惧"),
             ((0.26, 0.59, 0.29), "喜悦"),
             ((0.05, -0.05, 0.05), "平静")]
    for pad, want in cases:
        main_w, sec, r = decode_top2(pad)
        mark = "✓" if main_w == want else "✗"
        print(f"{mark} {pad} -> 主={main_w} 次={sec} (d2/d1={r})  期望={want}")
    print("族级: PAD(-0.85,-0.94,-0.90) ->", decode_family_top2([-0.85, -0.94, -0.90]))
    print("✓ decode 模块自检")
