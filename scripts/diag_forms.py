# -*- coding: utf-8 -*-
"""diag_forms — v6 分形态诊断: 同一 battery-val, 满血 / 纯文字 两条路分开测。

定位 famAcc 掉价的归属: 状态/bge 通路拖累, 还是文字通路本身退化。
用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge CUDA_VISIBLE_DEVICES=0 \
      python3 diag_forms.py ckpt_v61/vibrato.pt
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import pad_schema
import train as T
from vibrato import vib_messages
from vibrato.net import VibratoNet, collate_pad

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt_v61/vibrato.pt"
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # 0=用 ckpt 原生窗口
ck = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
vocab, unk_id = ck["vocab"], ck["unk_id"]
net = VibratoNet(vocab_size=len(vocab) + 1, max_len=ck["max_len"],
                 feat_dim=ck["feat_dim"]).cuda()
net.load_state_dict(ck["model"])
net.eval()
store = T.FeatStore("feats_v6.mm", "feats_v6_index.json") if ck["feat_dim"] else None

rows = [json.loads(ln) for ln in open("data/v6_val.jsonl", encoding="utf-8") if ln.strip()]
rows = [r for r in rows if "noul" in r["gold"]]
print(f"battery-val {len(rows)} 行, ckpt={ckpt_path}")


def strip_states(msgs):
    out = []
    for m in msgs:
        m2 = {"role": m["role"]}
        if "message" in m:
            m2["message"] = m["message"]
        out.append(m2)
    return out


@torch.no_grad()
def run(mode: str):
    """mode: full=满血 | text=纯文字 | state=有状态无bge | bge=无状态有bge"""
    correct, total, mae_sum = 0, 0, 0.0
    wrong_samples = []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        items, all_segs = [], []
        for r in chunk:
            msgs = r["messages"] if mode in ("full", "state") else strip_states(r["messages"])
            it, segs = T.encode_row({"messages": msgs, "gold": r["gold"]},
                                    vocab, unk_id, WIN or ck["max_len"], use_feats=bool(ck["feat_dim"]))
            items.append(it)
            all_segs.append(segs)
        b = collate_pad(items)
        feats = None
        if ck["feat_dim"]:
            feats = np.zeros((len(chunk), b["ids"].shape[1], ck["feat_dim"]), dtype="float32")
            if mode in ("full", "bge"):               # state/text 模式=零张量(无bge信号)
                for bi, segs in enumerate(all_segs):
                    for s, e, h in segs:
                        if h is None:
                            continue
                        f = store.get(h)
                        if f is not None:
                            feats[bi, s:e] = f
        sp = b["state_pos"].cuda() if b["state_pos"] is not None else None
        sv = b["state_vals"].cuda() if b["state_vals"] is not None else None
        sm = b["state_mask"].cuda() if b["state_mask"] is not None else None
        ft = torch.from_numpy(feats).cuda() if feats is not None else None
        out = net(b["ids"].cuda(), sp, sv, sm, b["echo"].cuda(), ft, lengths=b["lengths"].cuda())
        fam = out["family_probs"].argmax(-1).cpu()
        from battery import FAMILY_IDS
        for bi, r in enumerate(chunk):
            gold = FAMILY_IDS.index(r["gold"]["family"])
            pred = fam[bi].item()
            total += 1
            correct += int(pred == gold)
            centers = torch.tensor(pad_schema.BIN_CENTERS)
            gold_pad = ((torch.tensor([r["gold"]["pad_bins"][d] for d in pad_schema.DIMS])
                         * centers).sum(-1) - 5.0) / 4.0
            mae_sum += (out["pad_norm"][bi].cpu() - gold_pad.clamp(-1, 1)).abs().mean().item()
            if pred != gold and len(wrong_samples) < 5:
                wrong_samples.append((r["id"], FAMILY_IDS[pred], r["gold"]["family"]))
    print(f"[{mode:4s}] famAcc={correct/total:.3f} padMAE={mae_sum/total:.3f}")
    for w in wrong_samples:
        print("   错例:", w)


run("full")
run("state")
run("bge")
run("text")
