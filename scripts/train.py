# -*- coding: utf-8 -*-
"""Vibrato v6 训练 — v1/affect 消息范式 + bge 冻结知识注入。

输入: prep_v6.py 产物 v6_train/val.jsonl {id, group, messages, gold}
      + v6_vocab.json + feats_v6.mm/_index.json (feat_precompute.py, --no-feats 可跳过)

纪律:
- 逐epoch形态增广: 训练行重采样 dropout_forms(纯文字/压缩记录/满血三路),
  val 恒满血(测 chordia 满血路径); 无状态历史的行(legacy/occ)增广恒等→编码缓存
- bge 特征按文本单元 sha1 查 fp16 memmap 重组; 整行 15% 置零(bge-dropout,
  保纯文字通路); 状态值 σ=0.05 轻噪声(对快照误差鲁棒, val 不加)
- conf 监督 = gold.low_consistency(battery 行才有); --battery-only 配 --init 二段微调

设备: P100 = CUDA_VISIBLE_DEVICES=0 cuda:0(fp32; Pascal 无 fp16 算力, 勿 autocast)。
用法: cd /root/vibrato && PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge \
      CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt_v6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import battery  # noqa: E402
from vibrato import pad_schema  # noqa: E402
from vibrato import vib_messages  # noqa: E402
from vibrato.feat_bge import FEAT_DIM  # noqa: E402
from vibrato.net import VibratoNet, collate_pad  # noqa: E402

PAD_ID = 0
BGE_DROPOUT = 0.15
STATE_NOISE = 0.05


class FeatStore:
    """fp16 memmap 只读 + sha1 索引; 未命中返回 None(调用方置零行)。"""

    def __init__(self, mm_path: str, index_path: str, dim: int = None):
        import numpy as np
        self.np = np
        self.index = json.load(open(index_path, encoding="utf-8"))
        self.dim = dim or FEAT_DIM
        n_rows = os.path.getsize(mm_path) // (self.dim * 2)
        self.mm = self.np.memmap(mm_path, dtype=self.np.float16, mode="r",
                                 shape=(n_rows, self.dim))

    def get(self, h: str):
        ent = self.index.get(h)
        if ent is None:
            return None
        off, ln = ent
        return self.mm[off:off + ln]


def encode_row(row, vocab, unk_id, max_len, augment_rng=None, use_feats=True,
               p_strip=0.35, p_compress=0.35):
    """messages → (item, segs)。augment_rng 非 None 时先做形态增广。
    带状态行(几乎全是 battery 行)是满血路径的唯一教师, 增广力度宜轻
    (默认 0.15/0.10: 通用鲁棒性由 47k 纯文字 legacy 行承担, 不靠打折扣 battery)。"""
    msgs = row["messages"]
    # echo 锚定从原始消息提取(增广前)——快通路不随形态降级消失, 与 v5 语义一致:
    # 目标前最后一条带 pad 的 user 消息
    echo = next((m["pad"] for m in reversed(msgs[:-1])
                 if m["role"] == "user" and "pad" in m), None)
    if augment_rng is not None:
        msgs = vib_messages.dropout_forms(msgs, augment_rng,
                                          p_strip_state=p_strip, p_compress=p_compress)
    units = vib_messages.sequence_units(msgs, max_window=max_len)
    ids, spos, svals, smask = [], [], [], []
    segs = []          # (start, end, unit_hash) 文本段表, feats 拼装用
    for u in units:
        if u["kind"] == "text":
            start = len(ids)
            for ch in u["text"]:
                ids.append(vocab.get(ch, unk_id))
                spos.append(False)
            segs.append((start, len(ids),
                         hashlib.sha1(u["text"].encode()).hexdigest() if use_feats else None))
            svals.extend([None] * (len(ids) - start))
            smask.extend([None] * (len(ids) - start))
        else:
            ids.append(PAD_ID)
            spos.append(True)
            svals.append(u["vals"])
            smask.append(u["mask"])
    if len(ids) > max_len:                   # 防御性尾截(窗口预算应已保证不触发)
        keep = slice(len(ids) - max_len, len(ids))
        ids, spos = ids[keep], spos[keep]
        svals, smask = svals[keep], smask[keep]
        segs = [(max(0, s - (len(spos) - max_len)), max(0, e - (len(spos) - max_len)), h)
                for s, e, h in segs if e > len(spos) - max_len]
    item = {"ids": ids, "state_pos": spos, "echo": echo or [0.0, 0.0, 0.0],
            "pad_bins": row["gold"]["pad_bins"]}
    if any(spos):
        z = [0.0] * vib_messages.STATE_DIM
        item["state_vals"] = [v if v is not None else z for v in svals]
        item["state_mask"] = [v if v is not None else z for v in smask]
    if "family" in row["gold"]:
        item["family"] = row["gold"]["family"]
    if "noul" in row["gold"]:
        item["noul"] = row["gold"]["noul"]
    if "low_consistency" in row["gold"]:
        item["conf"] = 0.0 if row["gold"]["low_consistency"] else 1.0
    return item, segs


def build_batch(rows, store, rng, train, np):
    items, batch_segs = [], []
    for r in rows:
        if r.get("_static_item") is not None:
            it, segs = r["_static_item"], r["_static_segs"]
        else:
            aug = (rng if (train and build_batch.augment) else None)
            it, segs = encode_row(r, build_batch.vocab, build_batch.unk_id,
                                  build_batch.max_len, augment_rng=aug,
                                  use_feats=build_batch.use_feats,
                                  p_strip=build_batch.p_strip,
                                  p_compress=build_batch.p_compress)
        items.append(it)
        batch_segs.append(segs)
    b = collate_pad(items)
    b["conf"] = (None if not all("conf" in it for it in items) else
                 np.array([it["conf"] for it in items], dtype="float32"))
    feats = None
    if build_batch.use_feats:
        B, L = b["ids"].shape
        feats = np.zeros((B, L, build_batch.fdim), dtype="float32")
        for i, segs in enumerate(batch_segs):
            if train and rng.random() < BGE_DROPOUT:
                continue                     # 整行 bge-dropout → 纯文字通路
            for s, e, h in segs:
                if h is None:
                    continue
                f = store.get(h) if store else None
                if f is not None:
                    feats[i, s:e] = f
    return b, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v6_train.jsonl")
    ap.add_argument("--val", default="data/v6_val.jsonl")
    ap.add_argument("--vocab", default="data/v6_vocab.json")
    ap.add_argument("--out", default="ckpt_v6")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-feats", action="store_true", help="不用 bge (feat_dim=0)")
    ap.add_argument("--feats-mm", default="feats_v6.mm")
    ap.add_argument("--feats-index", default="feats_v6_index.json")
    ap.add_argument("--feat-dim", type=int, default=None,
                    help="覆盖特征维度(默认读 ckpt/1024)")
    ap.add_argument("--battery-only", action="store_true", help="只用带 noul 的行(二段微调)")
    ap.add_argument("--no-augment", action="store_true",
                    help="关闭 dropout_forms 增广(二段满血专精用)")
    ap.add_argument("--p-strip", type=float, default=0.15,
                    help="增广: 去状态概率(带状态行=battery, 宜轻)")
    ap.add_argument("--p-compress", type=float, default=0.10,
                    help="增广: 压缩为纯状态概率")
    ap.add_argument("--init", default=None, help="从 v6 ckpt 初始化权重")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    def load(p):
        with open(p, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]
    train_rows = load(args.train)
    val_rows = load(args.val)
    if args.battery_only:
        train_rows = [r for r in train_rows if "noul" in r["gold"]]
        val_rows = [r for r in val_rows if "noul" in r["gold"]]
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    unk_id = len(vocab)
    print(f"train {len(train_rows)} / val {len(val_rows)} | vocab {len(vocab)}+unk | "
          f"题库指纹 {battery.battery_digest()[:16]} | 刻度指纹 {pad_schema.contract_digest()[:16]}")

    use_feats = not args.no_feats
    store = None
    if use_feats:
        store = FeatStore(args.feats_mm, args.feats_index, dim=args.feat_dim)
        print(f"feats: {len(store.index)} 唯一单元, memmap {len(store.mm):,}×{store.dim}")

    # 无状态历史的行: dropout_forms 恒等 → 编码一次缓存(legacy/occ 占大头)
    n_static = 0
    for r in train_rows + val_rows:
        if not any(any(k in m for k in vib_messages.STATE_FIELDS)
                   for m in r["messages"][:-1]):
            r["_static_item"], r["_static_segs"] = encode_row(
                r, vocab, unk_id, args.max_len, use_feats=use_feats)
            n_static += 1
    print(f"静态编码缓存 {n_static}/{len(train_rows) + len(val_rows)} 行(无状态历史)")

    # 批组装参数走函数属性(避免全局)
    build_batch.vocab, build_batch.unk_id = vocab, unk_id
    build_batch.max_len, build_batch.use_feats = args.max_len, use_feats
    build_batch.augment = not args.no_augment
    build_batch.p_strip, build_batch.p_compress = args.p_strip, args.p_compress

    fdim = store.dim if store else FEAT_DIM
    build_batch.fdim = fdim if use_feats else 0
    net = VibratoNet(vocab_size=len(vocab) + 1, max_len=args.max_len,
                     feat_dim=fdim if use_feats else 0).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        print(f"已从 {args.init} 载入权重(二段微调)")
    print(f"VibratoNet v6: {sum(p.numel() for p in net.parameters()):,} params "
          f"(feat_dim={fdim if use_feats else 0})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, -(-len(train_rows) // args.batch)))  # ceil:含尾批

    def step(batch, feats, train):
        ids = batch["ids"].to(device)
        lengths = batch["lengths"].to(device)
        sp = batch["state_pos"].to(device) if batch["state_pos"] is not None else None
        sv = batch["state_vals"].to(device) if batch["state_vals"] is not None else None
        sm = batch["state_mask"].to(device) if batch["state_mask"] is not None else None
        ft = torch.from_numpy(feats).to(device) if feats is not None else None
        target = batch["target"].to(device)
        ep = batch["echo"].to(device)
        if train:                                            # echo 加噪(v5 纪律沿用)
            ep = (ep + torch.randn_like(ep) * 0.1).clamp_(-1, 1)
        if train and sv is not None:                       # 状态轻噪声(val 不加)
            sv = sv + torch.randn_like(sv) * STATE_NOISE * sm
        out = net(ids, sp, sv, sm, ep, ft, lengths=lengths)
        l_ce = -(target * out["pad_bins"].clamp_min(1e-9).log()).sum(-1).mean()
        loss = l_ce
        if batch["family"] is not None:
            loss = loss + F.nll_loss(out["family_probs"].clamp_min(1e-9).log(),
                                     batch["family"].to(device))
        if batch["noul"] is not None:
            nt = batch["noul"].to(device)
            noul_t = torch.stack([1 - nt, nt], -1)
            loss = loss + -(noul_t * out["noul_probs"].clamp_min(1e-9).log()).sum(-1).mean()
        if batch["conf"] is not None:
            ct = torch.from_numpy(batch["conf"]).to(device)
            loss = loss + 0.5 * F.binary_cross_entropy(out["conf"], ct)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        with torch.no_grad():
            centers = torch.tensor(pad_schema.BIN_CENTERS, device=device)
            gold_pad = ((target * centers).sum(-1) - 5.0) / 4.0
            pad_mae = (out["pad_norm"] - gold_pad.clamp(-1, 1)).abs().mean()
            fam_acc = ((out["family_probs"].argmax(-1) == batch["family"].to(device))
                       .float().mean() if batch["family"] is not None
                       else torch.tensor(0.0, device=device))
            noul_acc = (((out["noul_probs"][..., 1] > 0.5) == batch["noul"].to(device).bool())
                        .float().mean() if batch["noul"] is not None
                        else torch.tensor(0.0, device=device))
        return loss, pad_mae, fam_acc, noul_acc

    for epoch in range(args.epochs):
        net.train()
        rng.shuffle(train_rows)
        tr, t0 = [], time.time()
        for i in range(0, len(train_rows), args.batch):
            b, ft = build_batch(train_rows[i:i + args.batch], store, rng, True, np)
            tr.append(step(b, ft, True))
        net.eval()
        with torch.no_grad():
            va = [step(*build_batch(val_rows[i:i + args.batch], store, rng, False, np), False)
                  for i in range(0, len(val_rows), args.batch)]
        avg = lambda xs, i: sum(x[i].item() for x in xs) / max(1, len(xs))
        print(f"ep{epoch+1}/{args.epochs} | train loss={avg(tr,0):.4f} | "
              f"val loss={avg(va,0):.4f} padMAE={avg(va,1):.3f} "
              f"famAcc={avg(va,2):.2f} noulAcc={avg(va,3):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": net.state_dict(), "vocab": vocab, "unk_id": unk_id,
                "max_len": args.max_len, "feat_dim": fdim if use_feats else 0,
                "battery_digest": battery.battery_digest(),
                "pad_digest": pad_schema.contract_digest(),
                "affect_schema": vib_messages.schema_digest(),
                "consumed": list(vib_messages.CONSUMED_FIELDS)},
               os.path.join(args.out, "vibrato.pt"))
    with open(os.path.join(args.out, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump({"train": len(train_rows), "val": len(val_rows), "vocab": len(vocab),
                   "epochs": args.epochs, "feats": use_feats,
                   "final_val": {"loss": avg(va, 0), "pad_mae": avg(va, 1),
                                 "family_acc": avg(va, 2), "noul_acc": avg(va, 3)}},
                  f, ensure_ascii=False, indent=2)
    print(f"已保存 {args.out}/vibrato.pt + train_report.json")


if __name__ == "__main__":
    main()
