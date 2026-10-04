# -*- coding: utf-8 -*-
"""fit_v5 — 从零训练 v5 格式网络于 v6 的同一 train/val 切分(公平性审判)。

v6 行 → v5 格式(build_context 6条窗 + echo + 192尾截); NetV5 从零训练。
若其在 v6-val 上也只得 ~0.63, 则 ckpt_v51 头对头的 0.766 系跨版本泄漏。
用法: PYTHONPATH=/root/vibrato-torch-lib CUDA_VISIBLE_DEVICES=0 python3 fit_v5.py
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/root/vibrato")
import torch
import torch.nn.functional as F

from vibrato import pad_schema
from vibrato import vib_state
from vibrato.battery import FAMILY_IDS, N_NOUL, NOUL_IDS, battery_digest
from net_v5 import VibratoNet, collate_pad


def render(rows):
    out = []
    for r in rows:
        msgs = [{"side": ("user" if m["role"] == "user" else "ai"),
                 "text": m.get("message", "")} for m in r["messages"]]
        text_input = vib_state.build_context(msgs, len(msgs) - 1)
        prev_pad = next((m["pad"] for m in reversed(r["messages"][:-1])
                         if m["role"] == "user" and "pad" in m), None)
        out.append({"id": r["id"], "text_input": text_input,
                    "echo": vib_state.echo_from_prev(prev_pad),
                    "pad_bins": r["gold"]["pad_bins"],
                    **({"family": r["gold"]["family"]} if r["gold"].get("family") else {}),
                    **({"noul": r["gold"]["noul"]} if r["gold"].get("noul") else {}),
                    "conf": (None if "low_consistency" not in r["gold"]
                             else (0.0 if r["gold"]["low_consistency"] else 1.0))})
    return out


def main():
    dev = torch.device("cuda:0")
    train_rows = render([json.loads(l) for l in open("data/v6_train.jsonl", encoding="utf-8")
                         if l.strip()])
    val_rows = render([json.loads(l) for l in open("data/v6_val.jsonl", encoding="utf-8")
                       if l.strip()])
    print(f"v5格式: train {len(train_rows)} / val {len(val_rows)} | "
          f"题库 {battery_digest()[:8]}")

    vocab = {"<pad>": 0, "<unk>": 1}
    for r in train_rows:
        for ch in r["text_input"]:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    print(f"vocab {len(vocab)}")

    def enc(t, ml=192):
        return [vocab.get(ch, 1) for ch in t][-ml:]

    def to_batch(rs):
        items = [{"ids": enc(r["text_input"]), "echo": r["echo"],
                  "pad_bins": r["pad_bins"],
                  **({"family": r["family"]} if "family" in r else {}),
                  **({"noul": r["noul"]} if "noul" in r else {})}
                 for r in rs]
        b = collate_pad(items)
        b["conf"] = (torch.tensor([r["conf"] for r in rs]) if all(r["conf"] is not None
                   for r in rs) else None)
        return b

    net = VibratoNet(vocab_size=len(vocab), max_len=192).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=8 * (len(train_rows) // 16 + 1))

    def step(rs, train):
        b = to_batch(rs)
        ids, echo, tgt = b["ids"].to(dev), b["echo"].to(dev), b["target"].to(dev)
        if train:
            echo = (echo + torch.randn_like(echo) * 0.1).clamp_(-1, 1)
        out = net(ids, echo, lengths=b["lengths"].to(dev))
        loss = -(tgt * out["pad_bins"].clamp_min(1e-9).log()).sum(-1).mean()
        if b["family"] is not None:
            loss = loss + F.nll_loss(out["family_probs"].clamp_min(1e-9).log(),
                                     b["family"].to(dev))
        if b["noul"] is not None:
            nt = b["noul"].to(dev)
            loss = loss + -(torch.stack([1 - nt, nt], -1)
                            * out["noul_probs"].clamp_min(1e-9).log()).sum(-1).mean()
        if b["conf"] is not None:
            loss = loss + 0.5 * F.binary_cross_entropy(out["conf"], b["conf"].to(dev))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        with torch.no_grad():
            centers = torch.tensor(pad_schema.BIN_CENTERS, device=dev)
            gp = ((tgt * centers).sum(-1) - 5.0) / 4.0
            mae = (out["pad_norm"] - gp.clamp(-1, 1)).abs().mean()
            fa = ((out["family_probs"].argmax(-1) == b["family"].to(dev)).float().mean()
                  if b["family"] is not None else torch.tensor(0.0, device=dev))
            na = (((out["noul_probs"][..., 1] > 0.5) == b["noul"].to(dev).bool())
                  .float().mean() if b["noul"] is not None
                  else torch.tensor(0.0, device=dev))
        return loss, mae, fa, na

    for ep in range(8):
        net.train()
        import random
        random.Random(ep).shuffle(train_rows)
        t0 = time.time()
        for i in range(0, len(train_rows), 16):
            step(train_rows[i:i + 16], True)
        net.eval()
        with torch.no_grad():
            va = [step(val_rows[i:i + 16], False) for i in range(0, len(val_rows), 16)]
        avg = lambda xs, i: sum(x[i].item() for x in xs) / max(1, len(xs))
        print(f"ep{ep+1}/8 val loss={avg(va,0):.4f} padMAE={avg(va,1):.3f} "
              f"famAcc={avg(va,2):.2f} noulAcc={avg(va,3):.2f} ({time.time()-t0:.0f}s)",
              flush=True)

    # 二段: battery-only 微调(v5 冠军配方 lr 3e-4 × 6)
    bt = [r for r in train_rows if "noul" in r]
    bv = [r for r in val_rows if "noul" in r]
    print(f"二段 battery: train {len(bt)} / val {len(bv)}")
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6 * (len(bt) // 16 + 1))
    for ep in range(6):
        net.train()
        random.Random(100 + ep).shuffle(bt)
        for i in range(0, len(bt), 16):
            step(bt[i:i + 16], True)
        net.eval()
        with torch.no_grad():
            va = [step(bv[i:i + 16], False) for i in range(0, len(bv), 16)]
        avg = lambda xs, i: sum(x[i].item() for x in xs) / max(1, len(xs))
        print(f"s2 ep{ep+1}/6 battery-val loss={avg(va,0):.4f} padMAE={avg(va,1):.3f} "
              f"famAcc={avg(va,2):.2f} noulAcc={avg(va,3):.2f}", flush=True)

    import os
    os.makedirs("ckpt_v5fair", exist_ok=True)
    torch.save({"model": net.state_dict(), "vocab": vocab, "max_len": 192},
               "ckpt_v5fair/vibrato.pt")
    print("已保存 ckpt_v5fair")


if __name__ == "__main__":
    main()
