# -*- coding: utf-8 -*-
"""eval_bench — 标准测评集: Chinese-EmoBank(维度) + CPED(分类), 零训练样本冷评。

维度场(CVAS单句/CVAT多句): 逐句冷读(零echo零状态+bge), Pearson r 对
Valence/Arousal 众数标注 —— 与学界水位(V≈0.70/A≈0.45)直接可比。
文本繁体, t2s 转简双跑(opencc); 高置信子集(V/A 的 SD<1.0)作次口径。
分类场(CPED valid split): 对话上下文(说话人=user, 对方=assistant, 6条窗),
family 头零样本跨词表映射, 报 accuracy/macro-F1/逐类recall + 多数类基线。

用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge CUDA_VISIBLE_DEVICES=0 \
      python3 eval_bench.py cvas|cvat|cped [--raw-trad]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import vib_messages
from vibrato.battery import FAMILY_IDS
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

CKPT = "checkpoints/ckpt_v6o/vibrato.pt"

CPED_EMOTION_MAP = {
    "happy": "happy", "grateful": "affectionate",
    "relaxed": "calm", "neutral": "calm",
    "sadness": "sad", "depress": "sad",
    "worried": "anxious", "fear": "anxious",
    "anger": "angry", "disgust": "angry",
    "astonished": "amused",
    # negative-other / positive-other: 无对应族, 排除
}


def load_net():
    ck = torch.load(CKPT, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    return ck, net


def make_batch(rows_msgs, ck, fe):
    """[{messages, extra}] → 单批张量。messages 已是 v1/affect 形态。"""
    items, feats_list, Ls = [], [], []
    for msgs in rows_msgs:
        units = vib_messages.sequence_units(msgs, max_window=ck["max_len"])
        ids, spos, svals, smask = [], [], [], []
        for u in units:
            if u["kind"] == "text":
                for ch in u["text"]:
                    ids.append(ck["vocab"].get(ch, ck["unk_id"]))
                    spos.append(False)
                svals.extend([None] * len(u["text"]))
                smask.extend([None] * len(u["text"]))
            else:
                ids.append(0)
                spos.append(True)
                svals.append(u["vals"])
                smask.append(u["mask"])
        Ls.append(len(ids))
        items.append((ids, spos, svals, smask, units))
    B, L = len(items), max(Ls)
    ids_t = torch.zeros(B, L, dtype=torch.long)
    sp_t = torch.zeros(B, L, dtype=torch.bool)
    sv_t = torch.zeros(B, L, 5)
    sm_t = torch.zeros(B, L, 5)
    feats = np.zeros((B, L, ck["feat_dim"]), dtype="float32")
    for i, (ids, spos, svals, smask, units) in enumerate(items):
        n = len(ids)
        ids_t[i, :n] = torch.tensor(ids)
        sp_t[i, :n] = torch.tensor(spos)
        z = [0.0] * 5
        sv_t[i, :n] = torch.tensor([v if v is not None else z for v in svals])
        sm_t[i, :n] = torch.tensor([v if v is not None else z for v in smask])
        pos = 0
        for u in units:
            if u["kind"] == "text":
                if u["text"].strip():
                    f = fe.char_feats(u["text"], max_chars=n).cpu().numpy()
                    feats[i, pos:pos + len(u["text"])] = f[:len(u["text"])]
                pos += len(u["text"])
            else:
                pos += 1
    return ids_t, sp_t, sv_t, sm_t, feats, torch.tensor(Ls)


@torch.no_grad()
def run_all(rows_msgs, ck, net, fe, batch=24):
    pads, fams = [], []
    for i in range(0, len(rows_msgs), batch):
        chunk = rows_msgs[i:i + batch]
        ids, sp, sv, sm, ft, Ls = make_batch(chunk, ck, fe)
        out = net(ids.cuda(), sp.cuda(), sv.cuda(), sm.cuda(),
                  torch.zeros(len(chunk), 3).cuda(),
                  torch.from_numpy(ft).cuda(), lengths=Ls.cuda())
        pads.extend(out["pad_norm"].cpu().tolist())
        fams.extend(out["family_probs"].argmax(-1).cpu().tolist())
        print(f"  {min(i + batch, len(rows_msgs))}/{len(rows_msgs)}", flush=True)
    return pads, fams


def pearson(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def eval_dimensional(csv_path, tag, trad_raw):
    from opencc import OpenCC
    t2s = OpenCC("t2s")
    ck, net = load_net()
    fe = BgeFeaturizer()
    rows = []
    with open(csv_path, encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            text = (r["Text"] if trad_raw else t2s.convert(r["Text"])).strip()
            if text:
                rows.append((text, float(r["Valence_Mean"]), float(r["Arousal_Mean"]),
                             float(r.get("Valence_SD", 9)), float(r.get("Arousal_SD", 9))))
    print(f"{tag}: {len(rows)} 句 ({'繁体原文' if trad_raw else 't2s转简'})")
    msgs = [[{"role": "user", "message": t}] for t, *_ in rows]
    pads, _ = run_all(msgs, ck, net, fe)
    v_pred = [p[0] * 4 + 5 for p in pads]      # [-1,1]→[1,9] 仿射, r 不变
    a_pred = [p[1] * 4 + 5 for p in pads]
    v_gold = [r[1] for r in rows]
    a_gold = [r[2] for r in rows]
    r_v, r_a = pearson(v_pred, v_gold), pearson(a_pred, a_gold)
    hi = [i for i, r in enumerate(rows) if r[3] < 1.0 and r[4] < 1.0]
    r_vh = pearson([v_pred[i] for i in hi], [v_gold[i] for i in hi]) if len(hi) > 10 else float("nan")
    r_ah = pearson([a_pred[i] for i in hi], [a_gold[i] for i in hi]) if len(hi) > 10 else float("nan")
    print(f"== {tag} == Pearson r: V={r_v:.3f} A={r_a:.3f} | "
          f"高置信子集(SD<1, n={len(hi)}): V={r_vh:.3f} A={r_ah:.3f}")
    print("   学界水位: V≈0.70 A≈0.45 (SemEval-2026 / EmoBank 系)")


def eval_cped(trad_raw):
    from opencc import OpenCC
    t2s = OpenCC("t2s")
    ck, net = load_net()
    fe = BgeFeaturizer()
    conv = t2s.convert if not trad_raw else (lambda x: x)
    # valid split 按对话组块, 取可映射情绪的 utterance
    import random
    rng = random.Random(42)
    by_dialogue = {}
    with open("/root/cped/data/CPED/valid_split.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            by_dialogue.setdefault(r["Dialogue_ID"], []).append(r)
    cases = []
    for did, utts in by_dialogue.items():
        for i, r in enumerate(utts):
            fam = CPED_EMOTION_MAP.get(r["Emotion"])
            if not fam:
                continue
            hist = utts[max(0, i - 6):i]
            msgs = [{"role": ("user" if h["Speaker"] == r["Speaker"] else "assistant"),
                     "message": conv(h["Utterance"])} for h in hist]
            msgs.append({"role": "user", "message": conv(r["Utterance"])})
            cases.append({"gold": fam, "cped": r["Emotion"], "msgs": msgs})
    # 分层抽样 ≤250/映射族
    by_gold = {}
    for c in cases:
        by_gold.setdefault(c["gold"], []).append(c)
    sample = []
    for g, lst in sorted(by_gold.items()):
        rng.shuffle(lst)
        sample.extend(lst[:250])
    rng.shuffle(sample)
    print(f"CPED valid: 可映射 {len(cases)} 句, 分层抽样 {len(sample)} "
          f"({ {g: min(250, len(l)) for g, l in sorted(by_gold.items())} })")
    pads, fams = run_all([c["msgs"] for c in sample], ck, net, fe)
    golds = [FAMILY_IDS.index(c["gold"]) for c in sample]
    n = len(sample)
    acc = sum(int(p == g) for p, g in zip(fams, golds)) / n
    # 逐类 recall + macro-F1(粗)
    per = {}
    for f in sorted(set(golds)):
        tp = sum(int(p == g == f) for p, g in zip(fams, golds))
        fp = sum(int(p == f and g != f) for p, g in zip(fams, golds))
        fn = sum(int(g == f and p != f) for p, g in zip(fams, golds))
        per[FAMILY_IDS[f]] = (tp / max(1, tp + fn), tp / max(1, tp + fp))
    macro_f1 = sum((2 * r * p / max(1e-9, r + p)) for r, p in per.values()) / len(per)
    maj = max(set(golds), key=golds.count)
    print(f"== CPED(零样本跨词表映射) == acc={acc:.3f} macro-F1={macro_f1:.3f} "
          f"多数类基线={golds.count(maj)/n:.3f} ({FAMILY_IDS[maj]})")
    for fam, (rec, prec) in per.items():
        print(f"   {fam:12s} recall={rec:.2f} precision={prec:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["cvas", "cvat", "cped"])
    ap.add_argument("--raw-trad", action="store_true", help="不转简体, 直接繁体")
    a = ap.parse_args()
    if a.bench == "cvas":
        eval_dimensional("/root/cemo/ChineseEmoBank/CVAS_SD/CVAS_all.csv", "CVAS", a.raw_trad)
    elif a.bench == "cvat":
        eval_dimensional("/root/cemo/ChineseEmoBank/CVAT_SD/CVAT_all_SD.csv", "CVAT", a.raw_trad)
    else:
        eval_cped(a.raw_trad)
