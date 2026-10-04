# -*- coding: utf-8 -*-
"""prep_v6 — v6 训练数据组装器 (v1/affect 消息范式, piece ④-b)

把五路原料组装成 v6 训练/验证行:
  conversations/convos_a/convos_b.jsonl   原始会话(消息时间序)
  labels_a/labels_b.jsonl                 用户轮 teacher 标签 (id="{convo}-u{消息全局idx:02d}")
  agent_labels.jsonl                      agent 侧标注 (label_agent.py)
  legacy_balanced.jsonl                   61k 单行(纯文字路径)
  occ_dataset.jsonl                       OCC 2106 单行(纯文字路径)

行格式(存原始消息, 不存编码——dropout_forms 每 epoch 重采样后编码):
  {"id", "group"(会话id或legacy/occ), "messages":[VibratoMessage 满血全形态],
   "gold":{"pad_bins", "family"?, "noul"?}}

用户侧压力合成(训练先验, 宿主真值仍由调用方累积):
  pressure_t = clamp(0.9×pressure_{t-1} + 20×ΔPressure_raw(Δpad_t), 0, 100)
  Δpad 取同会话相邻两个已标用户轮; 首轮从 0 起步。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional

from vibrato import pad_schema
from vibrato import vib_messages

CONVO_FILES = ["data/conversations.jsonl", "data/convos_a.jsonl", "data/convos_b.jsonl"]
LABEL_FILES = ["data/labels_a.jsonl", "data/labels_b.jsonl"]
LEGACY_FILES = ["data/legacy_balanced.jsonl"]
OCC_FILES = ["data/occ_dataset.jsonl"]
SAJIAO_FILES = ["data/sajiao_rows.jsonl", "data/sajiao2_rows.jsonl", "data/flat_rows.jsonl"]  # v6.1/6.2/6.3 形态补刀
V5_PREFIX = "TARGET(需判断): [用户]: "
PRESSURE_DECAY, PRESSURE_GAIN = 0.9, 20.0
LEGACY_AFF_CAP = 2000                     # legacy affectionate 降采样上限(P≈0 漂移源)


def load_jsonl(paths):
    rows = []
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                rows.extend(json.loads(ln) for ln in fh if ln.strip())
    return rows


def pad_of(pad_bins) -> List[float]:
    p = pad_schema.bins_to_pad(pad_bins)
    return [p[d] for d in pad_schema.DIMS]


def build_convo_rows(convos, labels, agent_labels) -> List[Dict]:
    by_convo: Dict[str, List[Dict]] = {}
    for r in labels:
        cid, _, uidx = r["id"].rpartition("-u")
        by_convo.setdefault(cid, []).append({**r, "_msg_idx": int(uidx)})
    agent_by_convo = {a["id"]: {rt["idx"]: rt for rt in a["ratings"]}
                      for a in agent_labels}
    rows: List[Dict] = []
    for cid, lrows in by_convo.items():
        convo = convos.get(cid)
        if convo is None:
            print(f"  [skip] 标签 {cid} 找不到原始会话")
            continue
        msgs = convo["messages"]
        agent = agent_by_convo.get(cid, {})
        lrows.sort(key=lambda r: r["_msg_idx"])
        # 用户侧压力合成(沿已标轮链)
        prev_pad, pressure = None, 0.0
        state_by_idx: Dict[int, Dict] = {}
        for r in lrows:
            pad = pad_of(r["pad_bins"])
            if prev_pad is not None:
                dp = pad_schema.pad_to_pressure_delta(
                    pad[0] - prev_pad[0], pad[1] - prev_pad[1], pad[2] - prev_pad[2])
                pressure = max(0.0, min(100.0, PRESSURE_DECAY * pressure
                                        + PRESSURE_GAIN * dp))
            state_by_idx[r["_msg_idx"]] = {"pad": pad, "pressure": round(pressure, 2)}
            prev_pad = pad
        for r in lrows:
            t = r["_msg_idx"]
            # 同串平价纪律: gold 是 teacher 在"目标前 6 条"视窗下打的(vib_state.
            # MAX_CONTEXT_MSGS=6, build_context 与 teacher 同串)。训练行裁到同一
            # 视窗——远于 teacher 所见的上下文对 gold 是噪声(实测 512 全窗 famAcc
            # 掉 0.13)。512 是线上容量, 不是训练窗。
            vmsgs: List[Dict] = []
            for j in range(max(0, t - 6), t + 1):
                m = msgs[j]
                vm = {"role": "user" if m["side"] == "user" else "assistant",
                      "message": str(m.get("text", ""))}
                if j < t:                                   # 目标不携带状态(契约)
                    if j in state_by_idx:
                        vm.update(state_by_idx[j])
                    if j in agent:
                        vm["pad"] = agent[j]["pad"]
                        vm["pressure"] = round(agent[j]["pressure"], 2)
                vmsgs.append(vm)
            gold = {"pad_bins": r["pad_bins"], "family": r.get("family")}
            if r.get("noul"):
                gold["noul"] = r["noul"]
            if "low_consistency" in r:              # conf 头监督(teacher 一致性)
                gold["low_consistency"] = r["low_consistency"]
            rows.append({"id": r["id"], "group": cid, "messages": vmsgs, "gold": gold})
    return rows


def build_single_rows(label_rows, group) -> List[Dict]:
    """legacy/occ/sajiao 单行: 剥 v5 前缀取原文, 重走 v6 消息范式(单条编码与 v5 前缀同串)。
    legacy 的 affectionate 行降采样到 LEGACY_AFF_CAP(P≈0 标签漂移是撒娇极性反转的
    源头, 正确先验由 sajiao 批供给)。"""
    if group == "legacy":
        aff = [r for r in label_rows if r.get("family") == "affectionate"]
        rest = [r for r in label_rows if r.get("family") != "affectionate"]
        if len(aff) > LEGACY_AFF_CAP:
            keep = set(r["id"] for r in random.Random(42).sample(aff, LEGACY_AFF_CAP))
            label_rows = rest + [r for r in aff if r["id"] in keep]
    out = []
    for r in label_rows:
        ti = r["text_input"]
        if not ti.startswith(V5_PREFIX):
            print(f"  [skip] {r['id']} 非 v5 单行格式")
            continue
        text = ti[len(V5_PREFIX):]
        gold = {"pad_bins": r["pad_bins"], "family": r.get("family")}
        if r.get("noul"):
            gold["noul"] = r["noul"]
        if "low_consistency" in r:                  # conf 头监督(teacher 一致性)
            gold["low_consistency"] = r["low_consistency"]
        out.append({"id": r["id"], "group": group,
                    "messages": [{"role": "user", "message": text}], "gold": gold})
    return out


def split_rows(rows, val_ratio: float = 0.1):
    """会话行按会话组划分(防泄漏); legacy/occ 单行互相独立, 按行 id 划分
    (否则两个巨型单组一次哈希全进同侧, val 集大小随运气漂移)。"""
    def key(r):
        g = r["group"]
        return g if g not in ("legacy", "occ", "sajiao") else r["id"]
    keys = sorted({key(r) for r in rows})
    val_keys = {k for k in keys
                if int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) % 100
                < val_ratio * 100}
    tr = [r for r in rows if key(r) not in val_keys]
    va = [r for r in rows if key(r) in val_keys]
    return tr, va


def build_vocab(rows, cap: int = 21000) -> Dict[str, int]:
    """字符词表: id0=pad(保留), 按频次降序取前 cap-1。"""
    freq = Counter()
    for r in rows:
        for u in vib_messages.sequence_units(r["messages"]):
            if u["kind"] == "text":
                freq.update(u["text"])
    vocab = {"\x00": 0}
    for ch, _ in freq.most_common(cap - 1):
        if ch not in vocab and ch != "\x00":
            vocab[ch] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="v6")
    ap.add_argument("--vocab-cap", type=int, default=21000)
    args = ap.parse_args()

    convos = {c["id"]: c for c in load_jsonl(CONVO_FILES)}
    labels = load_jsonl(LABEL_FILES)
    agent_labels = load_jsonl(["data/agent_labels.jsonl"])
    legacy = load_jsonl(LEGACY_FILES)
    occ = load_jsonl(OCC_FILES)
    saajiao = load_jsonl(SAJIAO_FILES)

    convo_rows = build_convo_rows(convos, labels, agent_labels)
    legacy_rows = build_single_rows(legacy, "legacy")
    occ_rows = build_single_rows(occ, "occ")
    sajiao_rows = build_single_rows(saajiao, "sajiao")
    print(f"会话行 {len(convo_rows)} | legacy {len(legacy_rows)} | occ {len(occ_rows)} "
          f"| sajiao {len(sajiao_rows)}")

    # 契约自证: 全部行通过 v1/affect 校验(满血形态)
    for r in convo_rows + legacy_rows + occ_rows:
        vib_messages.validate_messages(r["messages"])
    # 单行同串: v6 编码 == v5 前缀渲染
    demo = legacy_rows[0]["messages"][0]["message"]
    unit = vib_messages.sequence_units([{"role": "user", "message": demo}])[0]
    assert unit["kind"] == "text" and unit["text"] == V5_PREFIX + demo
    print("契约自证 OK (validate + 单行同串)")

    all_rows = convo_rows + legacy_rows + occ_rows + sajiao_rows
    tr, va = split_rows(all_rows)
    with open(f"{args.out_prefix}_train.jsonl", "w", encoding="utf-8") as fh:
        for r in tr:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out_prefix}_val.jsonl", "w", encoding="utf-8") as fh:
        for r in va:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 词表只用训练行构建(审查#5: 含 val 会泄漏字符覆盖分布; val 未见字符走 UNK)
    vocab = build_vocab(tr, args.vocab_cap)
    with open(f"{args.out_prefix}_vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    n_state = sum(1 for r in convo_rows
                  if any(any(k in m for k in vib_messages.STATE_FIELDS)
                         for m in r["messages"][:-1]))
    print(f"train {len(tr)} / val {len(va)} | vocab {len(vocab)} | "
          f"带状态快照会话行 {n_state}/{len(convo_rows)}")
    print(f"→ {args.out_prefix}_train.jsonl / _val.jsonl / _vocab.json")


if __name__ == "__main__":
    main()
