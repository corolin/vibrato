# -*- coding: utf-8 -*-
"""eval_head2head — 同一批 battery-val 行, v5/v6 各用自家格式打分(消除切分差异)。

v5 格式: build_context(6条窗) + echo(上一已标用户轮的 pad); net_v5, ckpt_v51。
v6 格式: v1/affect 满血消息; net v6, ckpt_v61。
用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge CUDA_VISIBLE_DEVICES=0 \
      python3 eval_head2head.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import pad_schema
from vibrato import vib_state
import train as T
from vibrato import vib_messages
from vibrato.battery import FAMILY_IDS
from net_v5 import VibratoNet as NetV5, collate_pad as collate_v5
from vibrato.net import VibratoNet as NetV6, collate_pad as collate_v6

rows = [json.loads(ln) for ln in open("data/v6_val.jsonl", encoding="utf-8") if ln.strip()]
rows = [r for r in rows if "noul" in r["gold"]]
print(f"battery-val {len(rows)} 行(与 diag_forms 同一批)")

centers = torch.tensor(pad_schema.BIN_CENTERS)


def gold_of(r):
    fam = FAMILY_IDS.index(r["gold"]["family"])
    gp = ((torch.tensor([r["gold"]["pad_bins"][d] for d in pad_schema.DIMS]) * centers)
          .sum(-1) - 5.0) / 4.0
    return fam, gp.clamp(-1, 1)


@torch.no_grad()
def eval_v5():
    ck = torch.load("checkpoints/ckpt_v51/vibrato.pt", map_location="cuda:0", weights_only=False)
    vocab = ck["vocab"]
    net = NetV5(vocab_size=len(vocab), max_len=ck["max_len"]).cuda()
    net.load_state_dict(ck["model"])
    net.eval()
    def enc(t):
        return [vocab.get(ch, 1) for ch in t][-ck["max_len"]:]
    correct = total = 0
    mae = 0.0
    for i in range(0, len(rows), 16):
        items = []
        chunk = rows[i:i + 16]
        for r in chunk:
            msgs = [{"side": ("user" if m["role"] == "user" else "ai"),
                     "text": m.get("message", "")} for m in r["messages"]]
            target_idx = len(msgs) - 1
            text_input = vib_state.build_context(msgs, target_idx)
            prev_pad = next((m["pad"] for m in reversed(r["messages"][:-1])
                             if m["role"] == "user" and "pad" in m), None)
            echo = vib_state.echo_from_prev(prev_pad)
            items.append({"ids": enc(text_input), "echo": echo,
                          "pad_bins": r["gold"]["pad_bins"],
                          "family": r["gold"]["family"]})
        b = collate_v5(items)
        out = net(b["ids"].cuda(), b["echo"].cuda(), lengths=b["lengths"].cuda())
        fam = out["family_probs"].argmax(-1).cpu()
        for bi, r in enumerate(chunk):
            g, gp = gold_of(r)
            correct += int(fam[bi].item() == g)
            total += 1
            mae += (out["pad_norm"][bi].cpu() - gp).abs().mean().item()
    print(f"[v5  ] famAcc={correct/total:.3f} padMAE={mae/total:.3f}")


@torch.no_grad()
def eval_v6():
    ck = torch.load("ckpt_v61/vibrato.pt", map_location="cuda:0", weights_only=False)
    vocab, unk_id = ck["vocab"], ck["unk_id"]
    net = NetV6(vocab_size=len(vocab) + 1, max_len=ck["max_len"],
                feat_dim=ck["feat_dim"]).cuda()
    net.load_state_dict(ck["model"])
    net.eval()
    store = T.FeatStore("feats_v6.mm", "feats_v6_index.json")
    correct = total = 0
    mae = 0.0
    for i in range(0, len(rows), 16):
        items, all_segs = [], []
        chunk = rows[i:i + 16]
        for r in chunk:
            it, segs = T.encode_row(r, vocab, unk_id, ck["max_len"], use_feats=True)
            items.append(it)
            all_segs.append(segs)
        b = collate_v6(items)
        feats = np.zeros((len(chunk), b["ids"].shape[1], ck["feat_dim"]), dtype="float32")
        for bi, segs in enumerate(all_segs):
            for s, e, h in segs:
                f = store.get(h) if h else None
                if f is not None:
                    feats[bi, s:e] = f
        sp = b["state_pos"].cuda() if b["state_pos"] is not None else None
        sv = b["state_vals"].cuda() if b["state_vals"] is not None else None
        sm = b["state_mask"].cuda() if b["state_mask"] is not None else None
        out = net(b["ids"].cuda(), sp, sv, sm, b["echo"].cuda(), torch.from_numpy(feats).cuda(),
                  lengths=b["lengths"].cuda())
        fam = out["family_probs"].argmax(-1).cpu()
        for bi, r in enumerate(chunk):
            g, gp = gold_of(r)
            correct += int(fam[bi].item() == g)
            total += 1
            mae += (out["pad_norm"][bi].cpu() - gp).abs().mean().item()
    print(f"[v6  ] famAcc={correct/total:.3f} padMAE={mae/total:.3f}")


eval_v5()
eval_v6()
