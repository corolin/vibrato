# -*- coding: utf-8 -*-
"""v1/affect/ 线上格式 — VibratoMessage 消息范式(单文件自包含)

Jev 开创了 SystemOne 的接口范式(一次前向/schema约束/标定分布), 本文件把
affect 前端的"嘴型"定下来:

    请求  messages: VibratoMessage[](时间序, 末条=user+message=判断目标)
    响应  score×3(9-bin有序分布) + choice×1(8族分布) + noul×5(是非) + conf
          + write_back(钉回刚判消息的快照) + consumed(本代实读通道)

消息双形态: 文字形态(message=原文) 与 状态形态快照(pad/pressure/vitality),
同一条消息上可并存。全文字数组=通用agent路径, 完全在分布内。
echo 已溶解: 调用方把 Vibrato 上轮输出写回上一条 user 消息的 pad 字段即是回声,
Δ/EMA 由网络从窗口内连续快照自学, 压力累积归宿主(write_back 提供 d_pressure_raw)。

两层版本铁律(接口范式与模型消化分离):
    schema_digest()   线上格式指纹 — v1 冻结, 字段只增不改
    CONSUMED_FIELDS   本代 checkpoint 消化的字段 → 进 pad_schema.contract_digest;
                      请求带未消化字段(如 vitality)合法, 优雅忽略(遮罩置零)

单位纪律: 线上说域内原生单位 — pad[-1,1] / pressure[0,100] / vitality[-30,100];
归一化是模型私事(state_vector 的除数), 调用方不得自行换算。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from .vib_state import sanitize, role_of

SCHEMA_VERSION = "v1/affect"
ROLES: Tuple[str, ...] = ("user", "assistant")

# 状态字段规约: 名 → (维度, 域内下界, 域内上界, 归一化除数)
_FIELD_SPEC: Dict[str, Tuple[int, float, float, float]] = {
    "pad":      (3, -1.0,   1.0,   1.0),   # [p,a,d] 该轮情绪位置快照
    "pressure": (1,  0.0, 100.0, 100.0),   # 累积负荷快照(chordia 压力标准)
    "vitality": (1, -30.0, 100.0, 100.0),  # 剩余能量快照(chordia-v2)
}
STATE_FIELDS: Tuple[str, ...] = ("pad", "pressure", "vitality")
STATE_DIM = 5                                        # 摊平: [p,a,d,pressure,vitality]
STATE_DIVISORS: Tuple[float, ...] = (1.0, 1.0, 1.0, 100.0, 100.0)  # 模型私有归一化

# 本代(v6)消化: 文字+pad+pressure; vitality 收而未读(v7 拿到标签源再开)
CONSUMED_FIELDS: Tuple[str, ...] = ("message", "pad", "pressure")

MAX_WINDOW = 512        # 上下文窗口(字符位+状态位), 进 pad_schema.contract_digest
_EPS = 1e-6


def validate_messages(msgs: Sequence[Dict]) -> List[Dict]:
    """校验并归一化 v1/affect 请求体, 返回深拷贝(不改动入参)。

    规则: 非空列表; role ∈ {user, assistant}; 每条至少有文字或状态之一;
    pad 三浮点[-1,1]; pressure∈[0,100]; vitality∈[-30,100](越界截断);
    末条必须是 user 且带 message(判断目标), 且不得携带状态字段——那是答案不是输入。
    """
    if not isinstance(msgs, (list, tuple)) or not msgs:
        raise ValueError("messages 应为非空列表")
    out: List[Dict] = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] 应为对象")
        role = m.get("role")
        if role not in ROLES:
            raise ValueError(f"messages[{i}].role 应为 {list(ROLES)}, 得到 {role!r}")
        r: Dict = {"role": role}
        if m.get("message") is not None:
            text = sanitize(str(m["message"]))
            if not text:
                raise ValueError(f"messages[{i}].message 为空白")
            r["message"] = text
        pad = m.get("pad")
        if pad is not None:
            if not isinstance(pad, (list, tuple)) or len(pad) != 3:
                raise ValueError(f"messages[{i}].pad 应为 [p,a,d] 三元组")
            r["pad"] = [min(1.0, max(-1.0, float(v))) for v in pad]
        for k in ("pressure", "vitality"):
            if m.get(k) is not None:
                _, lo, hi, _ = _FIELD_SPEC[k]
                r[k] = min(hi, max(lo, float(m[k])))
        if not any(k in r for k in ("message",) + STATE_FIELDS):
            raise ValueError(f"messages[{i}] 既无文字形态也无状态形态")
        out.append(r)
    last = out[-1]
    if last["role"] != "user" or "message" not in last:
        raise ValueError("末条消息必须是 user 且带 message(判断目标)")
    if any(k in last for k in STATE_FIELDS):
        raise ValueError("判断目标不得携带状态字段——那是所求答案, 不是输入")
    return out


def state_vector(m: Dict) -> Tuple[List[float], List[float]]:
    """消息 → (归一化状态值[5], 字段遮罩[5])。缺席字段=值0/遮罩0(结构性缺席, 非哨兵)。"""
    vals = [0.0] * STATE_DIM
    mask = [0.0] * STATE_DIM
    pad = m.get("pad")
    if pad is not None:
        for i in range(3):
            vals[i], mask[i] = float(pad[i]), 1.0
    if m.get("pressure") is not None:
        vals[3], mask[3] = float(m["pressure"]) / 100.0, 1.0
    # vitality: 线上合法但 v6 未消化(CONSUMED_FIELDS 之外, v7 再开)——结构性忽略:
    # 遮罩恒 0, state_in 永远看不见它。"接口收着, 网络不读"由这里强制,
    # 否则未经训练的第 5 维被激活 = 契约声明与实际输入不一致(审查#1)。
    return vals, mask


def attach_state(msgs: Sequence[Dict], idx: int, pad: Optional[Sequence[float]] = None,
                 pressure: Optional[float] = None, vitality: Optional[float] = None) -> List[Dict]:
    """write_back 落账: 把快照钉到第 idx 条消息(通常=刚判过的旧目标, 下轮请求复用)。"""
    out = [dict(m) for m in msgs]
    m = out[idx]
    if pad is not None:
        m["pad"] = [float(v) for v in pad[:3]]
    if pressure is not None:
        m["pressure"] = float(pressure)
    if vitality is not None:
        m["vitality"] = float(vitality)
    return out


def dropout_forms(msgs: Sequence[Dict], rng,
                  p_strip_state: float = 0.35, p_compress: float = 0.35) -> List[Dict]:
    """逐消息形态 dropout(训练增广): 双形态消息三选一 —
    去状态(纯文字, 模拟通用agent) / 去文字(压缩记录, 模拟状态回放) /
    全留(chordia 满血路径)。只动非目标消息, 返回新列表。"""
    norm = validate_messages(msgs)
    out: List[Dict] = []
    last = len(norm) - 1
    for i, m in enumerate(norm):
        m2 = dict(m)
        if i != last and "message" in m2 and any(k in m2 for k in STATE_FIELDS):
            r = rng.random()
            if r < p_strip_state:
                for k in STATE_FIELDS:
                    m2.pop(k, None)
            elif r < p_strip_state + p_compress:
                m2.pop("message", None)
        out.append(m2)
    return out


def _annotation(snap: Dict) -> str:
    """状态快照 → 人类可读注记(teacher 渲染用, 域内原生单位)。"""
    parts = []
    if "pad" in snap:
        parts.append("P%+.2f A%+.2f D%+.2f" % tuple(snap["pad"]))
    if "pressure" in snap:
        parts.append("压力%.1f" % snap["pressure"])
    if "vitality" in snap:
        parts.append("活力%.1f" % snap["vitality"])
    return "⟨" + " ".join(parts) + "⟩"


def _build_groups(norm: List[Dict], max_window: int) -> List[List[Dict]]:
    """规范化消息 → 每消息的编码单元组[{kind:text,prefix,content} / {kind:state,...}],
    施加窗口预算: 从最旧丢整条消息(目标永不丢); 目标自身超窗保前缀截内容头。"""
    n = len(norm)
    groups: List[Dict] = []
    for i, m in enumerate(norm):
        tag = "TARGET(需判断): " if i == n - 1 else ""
        prefix = f"{tag}{role_of(m['role'])}: "
        content = m.get("message", "")
        units: List[Dict] = [{"kind": "text", "prefix": prefix, "content": content}]
        vals, mask = state_vector(m)
        if any(mask):
            units.append({"kind": "state", "vals": vals, "mask": mask,
                          "snap": {k: m[k] for k in STATE_FIELDS if k in m}})
        groups.append({"cost": len(prefix) + len(content) + (1 if len(units) > 1 else 0) + 1,
                       "units": units})
    keep_from, acc = 0, sum(g["cost"] for g in groups)
    while acc > max_window and keep_from < n - 1:
        acc -= groups[keep_from]["cost"]
        keep_from += 1
    if acc > max_window:                    # 只剩目标仍超窗: 截内容, 保 TARGET 前缀
        tu = groups[n - 1]["units"][0]
        over = acc - max_window
        tu["content"] = "…" + tu["content"][min(len(tu["content"]), over + 1):]
    return [g["units"] for g in groups[keep_from:]]


def sequence_units(msgs: Sequence[Dict], max_window: int = MAX_WINDOW) -> List[Dict]:
    """模型侧编码单元序列: 文本单元(text=prefix+content, 与 render_text 逐字同源——
    消息组间以 "\\n" 分隔单元补齐换行) 与状态伪token单元(vals/mask, 占 1 位)。
    数据准备据此拼 char id + 状态位。"""
    norm = validate_messages(msgs)
    out: List[Dict] = []
    groups = _build_groups(norm, max_window)
    for gi, units in enumerate(groups):
        for u in units:
            if u["kind"] == "text":
                out.append({"kind": "text", "text": u["prefix"] + u["content"]})
            else:
                out.append({"kind": "state", "vals": u["vals"], "mask": u["mask"]})
        if gi < len(groups) - 1:
            out.append({"kind": "text", "text": "\n"})   # 与 render_text 的 \n join 同源
    return out


def render_text(msgs: Sequence[Dict], max_window: int = MAX_WINDOW) -> str:
    """teacher/LLM 视图: 同一窗口预算下的纯文本渲染, 状态渲染为行尾注记。"""
    norm = validate_messages(msgs)
    lines = []
    for units in _build_groups(norm, max_window):
        parts, ann = [], None
        for u in units:
            if u["kind"] == "text":
                parts.append(u["prefix"] + u["content"])
            else:
                ann = _annotation(u["snap"])
        if ann:
            parts.append(ann)
        lines.append("".join(parts))
    return "\n".join(lines)


def schema_digest() -> str:
    """线上格式指纹(v1 冻结, 字段只增不改; 与模型代际无关)。"""
    canonical = json.dumps(
        {"version": SCHEMA_VERSION, "roles": list(ROLES),
         "fields": {k: list(v) for k, v in _FIELD_SPEC.items()},
         "state_fields": list(STATE_FIELDS), "state_dim": STATE_DIM,
         "target_rule": "last=user+message+no-state"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def consumed_digest() -> str:
    """模型消化能力指纹(进 pad_schema.contract_digest, 换代即变)。"""
    canonical = json.dumps(
        {"consumed": list(CONSUMED_FIELDS), "state_dim": STATE_DIM,
         "state_divisors": list(STATE_DIVISORS)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── 自检 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    full = [
        {"role": "assistant", "message": "今天过得怎么样呀？",
         "pad": [0.4, -0.2, 0.1], "pressure": 12.0, "vitality": 80.0},
        {"role": "user", "message": "还行吧，就那样\r\n\n\n\n一堆破事"},
        {"role": "assistant", "message": "听起来有点烦，想说说吗？",
         "pad": [-0.1, 0.2, 0.0], "pressure": 18.5},
        {"role": "user", "message": "算了不说了，你陪我聊点别的"},
    ]
    norm = validate_messages(full)
    assert norm[1]["message"] == "还行吧，就那样\n\n一堆破事"   # CRLF/空行清理
    assert "\r" not in norm[1]["message"]

    # 状态向量: 满配 / 半配 / 纯文字
    v, k = state_vector(norm[0])
    assert k == [1.0, 1.0, 1.0, 1.0, 0.0] and v[3] == 0.12 and v[4] == 0.0 \
        and v[:3] == [0.4, -0.2, 0.1]                     # vitality 收而不读(遮罩恒0)
    v2, k2 = state_vector(norm[2])
    assert k2 == [1.0, 1.0, 1.0, 1.0, 0.0] and v2[4] == 0.0      # vitality 缺席=结构性0
    v3, k3 = state_vector(norm[1])
    assert k3 == [0.0] * 5

    # 双渲染同源
    units = sequence_units(full)
    texts = [u["text"] for u in units if u["kind"] == "text"]
    states = [u for u in units if u["kind"] == "state"]
    assert len(states) == 2 and "TARGET(需判断): " in texts[-1]
    rt = render_text(full)
    assert texts[0] in rt and "⟨P+0.40 A-0.20 D+0.10 压力12.0 活力80.0⟩" in rt
    assert "⟨P-0.10 A+0.20 D+0.00 压力18.5⟩" in rt
    # parity 纪律: 模型文本单元串(含换行分隔) == teacher 渲染去掉状态注记
    import re as _re
    stripped = "\n".join(_re.sub(r"⟨[^⟩]*⟩", "", ln) for ln in rt.split("\n"))
    assert "".join(texts) == stripped, "sequence/render 同源破裂"
    print("── render_text ──")
    print(rt)

    # 非法输入逐一拒绝
    bads = [
        [],
        [{"role": "system", "message": "x"}],
        [{"role": "user", "message": "   "}],
        [{"role": "user", "message": "ok", "pad": [0.1, 0.2]}],
        [{"role": "assistant", "message": "ok", "pad": [0.1, 0.2, 9.0]},
         {"role": "user", "message": "判"}],                              # pad 越界→截断, 合法
        [{"role": "assistant"}],                                            # 无任何形态
        [{"role": "assistant", "message": "回我"}],                          # 末条非 user
        [{"role": "user", "message": "判我", "pad": [0, 0, 0]}],            # 目标带状态
    ]
    n_reject = 0
    for b in bads:
        try:
            validate_messages(b)
        except ValueError:
            n_reject += 1
    assert n_reject == 7, f"应拒 7 例, 实拒 {n_reject}"
    assert validate_messages(bads[4])[0]["pad"][2] == 1.0                   # 越界截断

    # 窗口: 超预算丢最旧整条, 目标保留
    long_msgs = ([{"role": "assistant", "message": "早" + "a" * 200, "pad": [0.1, 0, 0]},
                  {"role": "user", "message": "b" * 200}] * 3) + \
                [{"role": "user", "message": "最后这条必须活下来"}]
    u2 = sequence_units(long_msgs)
    all_text = "".join(x["text"] for x in u2 if x["kind"] == "text")
    # 目标必活; 预算内尽量多留新消息(三对历史只丢最旧两对); 总量守恒
    assert "最后这条必须活下来" in all_text and all_text.count("早") == 1
    assert len(all_text) + sum(1 for x in u2 if x["kind"] == "state") <= MAX_WINDOW + 8
    # 单条目标超窗: 截内容头, 保 TARGET 前缀
    u3 = sequence_units([{"role": "user", "message": "x" * 700}])
    t3 = u3[0]["text"]
    assert t3.startswith("TARGET(需判断): [用户]: …") and len(t3) <= MAX_WINDOW

    # 形态 dropout: 比率 + 结果可再校验
    rng = random.Random(11)
    trials, strip, compress, keep = 3000, 0, 0, 0
    for _ in range(trials):
        aug = dropout_forms(full, rng)
        validate_messages(aug)                          # 任何增广结果仍合法
        mid = aug[0]
        if "message" in mid and not any(k in mid for k in STATE_FIELDS):
            strip += 1
        elif "message" not in mid:
            compress += 1
        else:
            keep += 1
    assert abs(strip / trials - 0.35) < 0.03 and abs(compress / trials - 0.35) < 0.03
    print(f"dropout_forms: strip={strip/trials:.2f} compress={compress/trials:.2f} "
          f"keep={keep/trials:.2f}")

    # write_back 落账: 旧目标成中段消息后可挂状态
    wb = attach_state(full, 1, pad=[0.1, -0.3, -0.2], pressure=23.4)
    assert wb[1]["pad"] == [0.1, -0.3, -0.2] and validate_messages(wb) is not None

    d1, d2 = schema_digest(), schema_digest()
    assert d1 == d2
    print("schema_digest:", d1[:16], "| consumed_digest:", consumed_digest()[:16],
          "| consumed:", list(CONSUMED_FIELDS))
    print("✓ vib_messages 自检全部通过 (v1/affect)")
