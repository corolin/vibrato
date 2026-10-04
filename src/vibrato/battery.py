# -*- coding: utf-8 -*-
"""Vibrato 情绪专项题库 — 类 Jev 契约(score + choice + noul 三题型) v0.1.1

Laya 的三种题型(qtype)在情绪域的完整特化:
  score  ×3  P/A/D × 9档李克特bin       (契约在 pad_schema.py, 此处引用)
  choice     8 族训练头 + OCC-22 距离排序(occ.py 解码层输出)
  noul   ×2  文本层才能判的是非信号      (宿主行为信号, 非情绪动力学)

v0.1.1 变更:
- noul 5→2: 保留 suppressed(言不由衷) + directed_at_me(指向性)——
  这两题 PAD 推不出来(定义性不可导), 只能靠文本判读;
  negative(P<0 弱导出 0.78) / needs_comfort / escalating(宿主有ΔPAD历史自己算更准)
  移除——宿主或 LLM 自行判断更可靠
- route 字段改名 host_signal, 去掉 chordia 路由错误表述(Chordia 是 agent 情绪
  动力学引擎, 不是行为路由器)

冻结纪律同 pad_schema: 题干/选项/顺序改动 = 契约破裂, battery_digest() 两侧比对。
teacher 标注: 一条消息一次调用, 18项+族+是非 合并一个 JSON(省 6 倍调用)。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from . import pad_schema

# ── choice: 情绪族(8) ───────────────────────────────────────
# Laya EMOTION_BUCKETS 7 族 + 委屈(中文伴侣聊天特有, 英文体系缺位)
FAMILY_LABELS: List[Dict[str, str]] = [
    {"id": "happy", "zh": "开心"},
    {"id": "affectionate", "zh": "亲昵"},
    {"id": "calm", "zh": "平静"},
    {"id": "amused", "zh": "好笑"},
    {"id": "sad", "zh": "低落"},
    {"id": "anxious", "zh": "焦虑"},
    {"id": "angry", "zh": "生气"},
    {"id": "wronged", "zh": "委屈"},
]
FAMILY_IDS: List[str] = [f["id"] for f in FAMILY_LABELS]
N_FAMILY = len(FAMILY_IDS)

FAMILY_INSTRUCTIONS = (
    "TARGET 消息的发送者当前的主导情绪是哪一族？只选一个。"
    "开心=愉悦满足；亲昵=对对方的喜爱亲近；平静=事务性/无明显情绪；"
    "好笑=被逗乐/玩梗；低落=难过沮丧疲惫；焦虑=担心紧张不安；"
    "生气=愤怒不满恼火；委屈=觉得被误解/被亏待/有苦说不出。"
)

# ── noul: 文本层才能判的是非信号(2) ──────────────────────
# 这两题 PAD 推不出来(定义性不可导), 只能靠文本判读:
#   suppressed: 表里错位(为真时表面PAD≠真实PAD, 一个PAD判不了"装")
#   directed_at_me: 指代只在文本里("对你失望"vs"对老板失望" PAD 逐位相同)
# id, 中文问题, 判定说明(teacher 用), 宿主行为信号
NOUL_QUESTIONS: List[Dict[str, str]] = [
    {"id": "directed_at_me", "q": "情绪是否指向对话中的我",
     "def": "情绪的靶子是接收者(我), 而非第三方/自身处境/泛泛而谈",
     "host_signal": "真→宿主可能需要道歉或担责"},
    {"id": "suppressed", "q": "是否言不由衷/压抑",
     "def": "表面措辞与真实情绪不符(敷衍/逞强/反话/故作轻松)则为真",
     "host_signal": "真→宿主轻问深挖, 勿按字面回应"},
]
NOUL_IDS: List[str] = [q["id"] for q in NOUL_QUESTIONS]
N_NOUL = len(NOUL_IDS)


def battery_digest() -> str:
    canonical = json.dumps(
        {"pad": pad_schema.contract_digest(),
         "family": {"ids": FAMILY_IDS, "instructions": FAMILY_INSTRUCTIONS},
         "noul": [{"id": q["id"], "q": q["q"]} for q in NOUL_QUESTIONS]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_teacher_prompt(state: str, prev_pad: Optional[List[float]] = None) -> str:
    """情绪专项合并标注提示词: 一次调用出 18项 + 情绪族 + 5是非。"""
    lines = [
        "你是情绪标注专家。判断下面 TARGET 消息发送者的情绪状态, 只输出 JSON。",
        "", state, "",
        "回答三部分(合成一个 JSON 对象, 禁止解释文字):",
        "1. I1-I18: PAD 情感量表 18 词项, 1.00-9.00 两位小数。",
        "   P维 I1快乐-I2高兴-I3满意-I4惬意-I5希望-I6放松(9为符合,5中立,1反面);",
        "   A维 I7兴奋-I8警觉-I9刺激-I10狂热-I11活跃-I12惊慌;",
        "   D维 I13支配-I14影响-I15领导-I16重要-I17自由-I18强力。",
        f"2. family: 主导情绪族, 只能取: {', '.join(FAMILY_IDS)}。",
        "3. noul: 两个是非判断(true/false):",
    ]
    for q in NOUL_QUESTIONS:
        lines.append(f"   {q['id']}: {q['def']}")
    lines += [
        "",
        '输出格式: {"I1":..,"I18":..,"analysis":"..","family":"..",',
        '          "noul":{"directed_at_me":true,"suppressed":false}}',
    ]
    return "\n".join(lines)


def parse_teacher_json(raw: str) -> Dict[str, Any]:
    """teacher 输出 → {items: {I1..I18}, family: id, noul: {id: bool}}。校验+裁剪。

    词项分越界(如 0.0 或归一化值混入)一律钳位到 [1,9] 而非拒收——0.0 多为
    模型把量表当 0-9 用(即极度负向), 钳位保信息保 echo 链连续。
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    items = {}
    for i in range(1, 19):
        v = float(data[f"I{i}"])
        items[f"I{i}"] = max(1.0, min(9.0, v))
    family = data["family"]
    if family not in FAMILY_IDS:
        raise ValueError(f"未知情绪族: {family}")
    noul_raw = data["noul"]
    noul = {qid: bool(noul_raw[qid]) for qid in NOUL_IDS}
    return {"items": items, "family": family, "noul": noul, "analysis": data.get("analysis", "")}


def validate_battery_gold(gold: Dict[str, Any]) -> List[str]:
    problems: List[str] = list(pad_schema.validate_gold(gold))
    if gold.get("family") not in FAMILY_IDS:
        problems.append(f"family 非法: {gold.get('family')}")
    for qid in NOUL_IDS:
        if not isinstance(gold.get("noul", {}).get(qid), bool):
            problems.append(f"noul.{qid} 缺失或非布尔")
    return problems


if __name__ == "__main__":
    # 题库自检
    assert N_FAMILY == 8 and N_NOUL == 2 and len(set(FAMILY_IDS)) == N_FAMILY
    assert set(NOUL_IDS) == {"directed_at_me", "suppressed"}
    prompt = build_teacher_prompt("TARGET message to judge:\n[them]: 哼, 随便你吧")
    for token in ["I18", "family", "wronged", "suppressed", "directed_at_me"]:
        assert token in prompt, token
    parsed = parse_teacher_json(json.dumps({
        **{f"I{i}": 5.0 for i in range(1, 19)}, "family": "wronged",
        "noul": {q: True for q in NOUL_IDS}}))
    assert parsed["family"] == "wronged" and all(parsed["noul"].values())
    assert pad_schema.bins_to_pad(pad_schema.items_to_bins(parsed["items"]))["pleasure"] == 0.0
    good = {"pad_bins": pad_schema.items_to_bins({f"I{i}": 5.0 for i in range(1, 19)}),
            "echo": [0.0] * 3, "family": "calm",
            "noul": {q: False for q in NOUL_IDS}}
    assert validate_battery_gold(good) == []
    bad = {"pad_bins": good["pad_bins"], "echo": [0.0] * 3, "family": "happy", "noul": {}}
    assert len(validate_battery_gold(bad)) == N_NOUL
    print("battery_digest:", battery_digest())
    print("题库: score×3(P/A/D×9档) + choice(8族训练+OCC-22解码) + noul×2(文本层信号)")
    print("✓ battery 自检全部通过")
