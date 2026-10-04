# -*- coding: utf-8 -*-
"""occ.py — OCC-22 距离排序解码层(v0.1.1 choice 输出)

OCC(Ortony-Clore-Collins 1988) 22 种情绪类型; 质心从自家 occ_cases.jsonl
(2106 条 LLM 自评 PAD, 22 型各 ~96 条)实测计算。

用法:
  from occ import occ_rank
  ranked = occ_rank([0.35, -0.20, 0.10])   # pad_norm3 ∈ [-1,1]
  # → [{"id": "love", "zh": "喜爱", "d": 1.23, "rank": 1}, ...] 全 22 按距离升序

设计: 这不是训练头——是 PAD 输出上的纯解码层(最近邻排序)。
调用方自己取 top-K; 想要次近判断可看 d[1]/d[0] 比值(骑墙规则同 decode.py)。
"""

from __future__ import annotations

import math
from typing import Dict, List

# ── OCC-22 质心表 ──────────────────────────────────────────
# 来源: occ_cases.jsonl 2106 条 LLM 自评 PAD([-4,4]) / 4 → [-1,1] 后取均值。
# 精度参照: 撒娇文本 P>0 时 love/liking/joy 排前; 愤怒文本 hate/reproach 排前。
# 鬼才质心: shame D=-0.718(最弱势)/pride D=+0.656(最强势)/relief A=+0.085(最低唤醒正值)

OCC_CENTROIDS: Dict[str, List[float]] = {
    "admiration":       [+0.757, +0.565, -0.267],
    "disappointment":   [-0.649, +0.095, -0.353],
    "disliking":        [-0.382, +0.230, -0.082],
    "distress":         [-0.565, +0.110, -0.454],
    "fear":             [-0.589, +0.590, -0.609],
    "fears_confirmed":  [-0.670, +0.183, -0.461],
    "gloating":         [+0.770, +0.557, +0.422],
    "gratification":    [+0.790, +0.345, +0.442],
    "happy_for":        [+0.816, +0.602, +0.235],
    "hate":             [-0.686, +0.453, -0.024],
    "hope":             [+0.695, +0.580, +0.089],
    "joy":              [+0.854, +0.657, +0.449],
    "liking":           [+0.723, +0.405, +0.104],
    "love":             [+0.732, +0.454, -0.161],
    "pity":             [-0.513, +0.246, -0.276],
    "pride":            [+0.805, +0.610, +0.656],
    "relief":           [+0.738, +0.085, +0.261],
    "remorse":          [-0.675, +0.296, -0.526],
    "reproach":         [-0.633, +0.478, +0.215],
    "resentment":       [-0.681, +0.523, -0.403],
    "satisfaction":     [+0.749, +0.298, +0.381],
    "shame":            [-0.688, +0.554, -0.718],
}

OCC_ZH: Dict[str, str] = {
    "happy_for": "为你高兴", "gloating": "幸灾乐祸", "resentment": "怨恨",
    "pity": "心疼", "hope": "期待", "fear": "担忧", "satisfaction": "满意",
    "fears_confirmed": "应验", "relief": "释然", "disappointment": "失望",
    "joy": "喜悦", "distress": "苦恼", "pride": "骄傲", "admiration": "钦佩",
    "shame": "羞愧", "reproach": "责备", "gratification": "欣慰",
    "remorse": "懊悔", "love": "喜爱", "hate": "厌恶", "liking": "好感",
    "disliking": "轻度反感",
}

# 骑墙阈值(与 decode.py SECOND_RATIO 一致): d2/d1 < 此值时次近值得关注
SECOND_RATIO = 1.3


def occ_rank(pad_norm3: List[float], top_k: int = 0) -> List[Dict]:
    """PAD 归一化值 [-1,1] → OCC-22 按欧氏距离升序排列。

    pad_norm3: [P, A, D] 各维 [-1,1]
    top_k: >0 只返回前 K 个; 0 返回全部 22 个
    返回: [{"id", "zh", "d", "rank"}, ...] 按距离升序
    """
    p, a, d = pad_norm3[0] * 4, pad_norm3[1] * 4, pad_norm3[2] * 4  # → [-4,4] 原始空间
    scored = []
    for occ_id, c in OCC_CENTROIDS.items():
        dist = math.sqrt((p - c[0] * 4) ** 2 + (a - c[1] * 4) ** 2 + (d - c[2] * 4) ** 2)
        scored.append({"id": occ_id, "zh": OCC_ZH[occ_id], "d": round(dist, 3)})
    scored.sort(key=lambda x: x["d"])
    for i, item in enumerate(scored):
        item["rank"] = i + 1
    return scored[:top_k] if top_k > 0 else scored


def occ_top2(pad_norm3: List[float]) -> Dict:
    """便捷接口: 最近 OCC 类型 + 骑墙判断(次近值得不值得关注)。

    返回: {"top1": "love", "top1_zh": "喜爱", "d1": 1.23,
           "top2": "liking", "top2_zh": "好感", "d2": 1.45,
           "straddling": True/False}
    """
    ranked = occ_rank(pad_norm3)
    t1, t2 = ranked[0], ranked[1]
    return {
        "top1": t1["id"], "top1_zh": t1["zh"], "d1": t1["d"],
        "top2": t2["id"], "top2_zh": t2["zh"], "d2": t2["d"],
        "straddling": t2["d"] / t1["d"] < SECOND_RATIO if t1["d"] > 0 else False,
    }


# ── 自检 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. 表完整性
    assert len(OCC_CENTROIDS) == 22 and len(OCC_ZH) == 22
    assert set(OCC_CENTROIDS.keys()) == set(OCC_ZH.keys())
    for c in OCC_CENTROIDS.values():
        assert all(-1.0 <= v <= 1.0 for v in c), c

    # 2. 撒娇(正向亲昵) → love/liking/joy 应在前 5
    sajiao = occ_rank([0.30, 0.10, -0.10], top_k=5)
    top5_ids = {x["id"] for x in sajiao}
    assert "love" in top5_ids or "liking" in top5_ids, sajiao
    print(f"撒娇 [0.30, 0.10, -0.10] → top5: {[(x['id'], x['d']) for x in sajiao]}")

    # 3. 愤怒 → hate/reproach/resentment 应在前 5
    angry = occ_rank([-0.70, 0.45, 0.10], top_k=5)
    top5_angry = {x["id"] for x in angry}
    assert "hate" in top5_angry or "reproach" in top5_angry or "resentment" in top5_angry, angry
    print(f"愤怒 [-0.70, 0.45, 0.10] → top5: {[(x['id'], x['d']) for x in angry]}")

    # 4. 崩溃(极度负面低支配) → distress/shame/fear
    crash = occ_rank([-0.90, 0.30, -0.80], top_k=3)
    print(f"崩溃 [-0.90, 0.30, -0.80] → top3: {[(x['id'], x['d']) for x in crash]}")

    # 5. 原点(中性) → 应给中距值, 无 crash
    neutral = occ_top2([0.0, 0.0, 0.0])
    print(f"中性原点 → top1={neutral['top1']}({neutral['d1']:.2f}) "
          f"top2={neutral['top2']}({neutral['d2']:.2f}) 骑墙={neutral['straddling']}")

    # 6. top2 接口
    t2 = occ_top2([0.85, 0.65, 0.45])   # joy 质心附近
    assert t2["top1"] in ("joy", "happy_for", "pride"), t2
    print(f"joy 质心 → top1={t2['top1']} d1={t2['d1']} 骑墙={t2['straddling']}")

    print(f"\nOCC-22 质心数: {len(OCC_CENTROIDS)}, SECOND_RATIO={SECOND_RATIO}")
    print("✓ occ 自检全部通过")
