# -*- coding: utf-8 -*-
"""finetune_cvas — CVAS 5折交叉验证微调线(学术口径): 与零样本线对照。

每折: 其余4折微调 ckpt_v6o(只监督 V/A 两维的9-bin软标签, D无标注不监督,
family/noul 头无标签自然冻结), 本折测 Pearson r。语料无官方划分, 均分5块
做折(与 CVAS 原文同法)。文本 t2s 转简(贴字符词表; bge 双体通吃)。

用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge CUDA_VISIBLE_DEVICES=0 \
      python3 finetune_cvas.py [--epochs 4]
"""
from __future__ import annotations

import argparse
import csv
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch
import torch.nn.functional as F

from vibrato import pad_schema
from vibrato import vib_messages
from eval_bench import make_batch
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

CKPT = "checkpoints/ckpt_v6o/vibrato.pt"
FOLD = "/root/cemo/ChineseEmoBank/CVAS_SD/CVAS_{}.csv"


def load_folds():
    from opencc import OpenCC
    t2s = OpenCC("t2s")
    folds = []
    for k in range(1, 6):
        rows = []
        with open(FOLD.format(k), encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                text = t2s.convert(r["Text"]).strip()
                if text:
                    rows.append((text, float(r["Valence_Mean"]), float(r["Arousal_Mean"])))
        folds.append(rows)
    return folds


@torch.no_grad()
def predict(net, ck, fe, texts, batch=24):
    pads = []
    for i in range(0, len(texts), batch):
        chunk = [[{"role": "user", "message": t}] for t in texts[i:i + batch]]
        ids, sp, sv, sm, ft, Ls = make_batch(chunk, ck, fe)
        out = net(ids.cuda(), sp.cuda(), sv.cuda(), sm.cuda(),
                  torch.zeros(len(chunk), 3).cuda(),
                  torch.from_numpy(ft).cuda(), lengths=Ls.cuda())
        pads.extend(out["pad_norm"].cpu().tolist())
    return pads


def pearson(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    folds = load_folds()
    ck0 = torch.load(CKPT, map_location="cpu", weights_only=False)
    fe = BgeFeaturizer()
    centers = torch.tensor(pad_schema.BIN_CENTERS).cuda()

    rs_v, rs_a, fold_detail = [], [], []
    for k in range(5):
        test = folds[k]
        train = [r for j in range(5) if j != k for r in folds[j]]
        net = VibratoNet(vocab_size=len(ck0["vocab"]) + 1, max_len=ck0["max_len"],
                         feat_dim=ck0["feat_dim"]).cuda()
        net.load_state_dict(ck0["model"])
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        # 目标: V→pleasure, A→arousal (D 无标注, 不进损失)
        tgts = []
        for _, v, a in train:
            tgts.append({"p": pad_schema.likert_to_bins(v), "a": pad_schema.likert_to_bins(a)})
        idx = np.arange(len(train))
        for ep in range(args.epochs):
            net.train()
            np.random.shuffle(idx)
            for i in range(0, len(idx), 24):
                sel = idx[i:i + 24]
                chunk = [[{"role": "user", "message": train[j][0]}] for j in sel]
                ids, sp, sv, sm, ft, Ls = make_batch(chunk, ck0, fe)
                out = net(ids.cuda(), sp.cuda(), sv.cuda(), sm.cuda(),
                          torch.zeros(len(sel), 3).cuda(),
                          torch.from_numpy(ft).cuda(), lengths=Ls.cuda())
                tv = torch.stack([torch.tensor(tgts[j]["p"]) for j in sel]).cuda()
                ta = torch.stack([torch.tensor(tgts[j]["a"]) for j in sel]).cuda()
                loss = -(tv * out["pad_bins"][:, 0].clamp_min(1e-9).log()).sum(-1).mean() \
                     + -(ta * out["pad_bins"][:, 1].clamp_min(1e-9).log()).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        pads = predict(net, ck0, fe, [t for t, *_ in test])
        v_pred = [p[0] * 4 + 5 for p in pads]
        a_pred = [p[1] * 4 + 5 for p in pads]
        rv = pearson(v_pred, [r[1] for r in test])
        ra = pearson(a_pred, [r[2] for r in test])
        rs_v.append(rv)
        rs_a.append(ra)
        fold_detail.append(f"f{k+1}: V={rv:.3f} A={ra:.3f} (train {len(train)}/test {len(test)})")
        print(fold_detail[-1], flush=True)

    mv, ma = np.mean(rs_v), np.mean(rs_a)
    sv, sa = np.std(rs_v), np.std(rs_a)
    print(f"\n== CVAS 5折CV微调 == V: {mv:.3f}±{sv:.3f}  A: {ma:.3f}±{sa:.3f}")
    print(f"   零样本线: V=0.517 A=0.098 | 学界域内水位: V≈0.70 A≈0.45")


if __name__ == "__main__":
    main()
