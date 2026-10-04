# -*- coding: utf-8 -*-
"""偷懒路径评估 — score 王道策略的定量验证。

三问:
  1) PAD→族 质心最近邻解码的天花板(用 gold PAD 测, 排除模型误差)
  2) 哪些 noul 可由 PAD/echo 阈值导出(与 battery 行的标注对齐率)
  3) 端到端: v2 模型预测的 PAD → 解码族 vs 文本族头, 差距多少

用法(服务器): cd /root/vibrato && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/vibrato-torch-lib \
  python3 eval_lazy_path.py --ckpt ckpt_v2/vibrato.pt --legacy legacy_dataset.jsonl --battery snap_battery.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import pad_schema  # noqa: E402
from vibrato.battery import FAMILY_IDS  # noqa: E402


def pad_norm_of(row):
    return [pad_schema.normalize_likert(pad_schema.expected_from_bins(row["pad_bins"][d]))
            for d in pad_schema.DIMS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2/vibrato.pt")
    ap.add_argument("--legacy", default="data/legacy_dataset.jsonl")
    ap.add_argument("--battery", default="snap_battery.jsonl")
    args = ap.parse_args()

    legacy = [json.loads(l) for l in open(args.legacy, encoding="utf-8") if l.strip()]
    battery = [json.loads(l) for l in open(args.battery, encoding="utf-8") if l.strip()]

    # ── 1) 8族质心(训练半区计算, 测试半区评估, 防泄漏) ──
    rng = random.Random(7)
    idx = list(range(len(legacy)))
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = set(idx[:half]), idx[half:]

    import math
    cent = {f: [0.0, 0.0, 0.0] for f in FAMILY_IDS}
    cnt = {f: 0 for f in FAMILY_IDS}
    for i in train_idx:
        r = legacy[i]
        p = pad_norm_of(r)
        for k in range(3):
            cent[r["family"]][k] += p[k]
        cnt[r["family"]] += 1
    for f in FAMILY_IDS:
        if cnt[f]:
            cent[f] = [v / cnt[f] for v in cent[f]]
    print("8族PAD质心(legacy实测):")
    for f in FAMILY_IDS:
        print(f"  {f:13s} ({cent[f][0]:+.2f},{cent[f][1]:+.2f},{cent[f][2]:+.2f})  n={cnt[f]}")

    def decode_family(p3):
        best, bd = None, 1e9
        for f in FAMILY_IDS:
            d = sum((a - b) ** 2 for a, b in zip(p3, cent[f]))
            if d < bd:
                best, bd = f, d
        return best

    # 测试半区: gold PAD → 解码族 vs 词标族
    hit = tot = 0
    per_fam = {f: [0, 0] for f in FAMILY_IDS}
    for i in test_idx:
        r = legacy[i]
        pred = decode_family(pad_norm_of(r))
        tot += 1
        hit += pred == r["family"]
        per_fam[r["family"]][1] += 1
        per_fam[r["family"]][0] += pred == r["family"]
    print(f"\n[1] gold-PAD→质心解码族: {hit}/{tot} = {hit/tot:.3f} (偷懒路径天花板)")
    for f in FAMILY_IDS:
        if per_fam[f][1]:
            print(f"    {f:13s} {per_fam[f][0]}/{per_fam[f][1]} = {per_fam[f][0]/per_fam[f][1]:.2f}")

    # ── 2) noul 可导出性(battery 行对齐率) ──
    print(f"\n[2] noul 从 PAD/echo 阈值导出的对齐率(battery n={len(battery)}):")
    rules = [
        ("negative",   "P<0",           lambda r, p: p[0] < 0),
        ("escalating", "A升或P降",       lambda r, p: (p[1] - r["echo"][1] > 0.1) or (p[0] - r["echo"][0] < -0.1)),
        ("needs_comfort", "不可导",      lambda r, p: False),
        ("directed_at_me", "不可导",     lambda r, p: False),
        ("suppressed", "不可导",         lambda r, p: False),
    ]
    for qid, label, fn in rules:
        agree = sum(1 for r in battery if bool(fn(r, pad_norm_of(r))) == bool(r["noul"][qid]))
        base = max(sum(1 for r in battery if r["noul"][qid]),
                   sum(1 for r in battery if not r["noul"][qid]))
        print(f"    {qid:14s} {label:8s} {agree}/{len(battery)} = {agree/len(battery):.2f}  (全猜多数={base/len(battery):.2f})")

    # ── 3) 端到端: v2 预测 PAD → 解码 ──
    import torch
    from net import VibratoNet, collate_pad
    ck = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]), max_len=ck["max_len"]).cuda()
    net.load_state_dict(ck["model"])
    net.eval()
    enc = lambda t: [ck["vocab"].get(ch, 1) for ch in t][-ck["max_len"] + 1:]
    hit2 = tot2 = 0
    mae = 0.0
    with torch.no_grad():
        for i in test_idx[:3000]:
            r = legacy[i]
            t = torch.tensor([enc(r["text_input"])]).cuda()
            e = torch.tensor([r["echo"]]).cuda()
            out = net(t, e)
            p = out["pad_norm"][0].tolist()
            gold_p = pad_norm_of(r)
            mae += sum(abs(a - b) for a, b in zip(p, gold_p)) / 3
            tot2 += 1
            hit2 += decode_family(p) == r["family"]
    print(f"\n[3] v2端到端(3000样本): PAD MAE={mae/tot2:.3f} | 预测PAD→解码族 acc={hit2/tot2:.3f}"
          f" | 对照: 文本族头在legacy val≈0.98")


if __name__ == "__main__":
    main()
