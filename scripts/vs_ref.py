# -*- coding: utf-8 -*-
"""vs_ref — 全员对裁判(ref4b)的统一口径表: top1/top2/noul + 延迟 + token。

judge.py 的表是"各挑战者 vs vib"；本表把 vib 也拉进考场, 所有人对同一位裁判
(本地4B答卷)交卷——vib 自己的 top1/top2/noul 由此而来。
用法: PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 python3 vs_ref.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, "/root/vibrato")
import torch

from vibrato import battery
from answer_sheet import questions, echo_chain_from_ref
from vibrato.decode import decode_top2

def tag_ok(path, ref_tag):
    name = os.path.basename(path)[8:-6]
    return name != ref_tag


def load_sheet(tag):
    return [json.loads(l) for l in open(f"data/answers_{tag}.jsonl", encoding="utf-8")
            if l.strip()]


def vib_answers(ck_path="checkpoints/ckpt_v6o/vibrato.pt"):
    import numpy as np
    import vib_messages
    from feat_bge import BgeFeaturizer
    from net import VibratoNet
    ck = torch.load(ck_path, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    fe = BgeFeaturizer()
    convos = {c["id"]: c for c in (json.loads(l) for l
                                   in open("data/test_convos.jsonl", encoding="utf-8")
                                   if l.strip())}
    qs, echo = questions(), echo_chain_from_ref()
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (cid, idx, _t) in enumerate(qs):
            msgs = convos[cid]["messages"][max(0, idx - 6):idx + 1]
            vmsgs, pin = [], None
            for j, m in enumerate(msgs):
                vm = {"role": "user" if m["side"] == "user" else "assistant",
                      "message": str(m.get("text", ""))}
                if j < len(msgs) - 1 and vm["role"] == "user":
                    pin = len(vmsgs)
                vmsgs.append(vm)
            if pin is not None and i > 0:
                vmsgs[pin]["pad"] = echo[i]
            units = vib_messages.sequence_units(vmsgs, max_window=ck["max_len"])
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
            L = len(ids)
            z = [0.0] * 5
            feats = np.zeros((1, L, ck["feat_dim"]), dtype="float32")
            pos = 0
            for u in units:
                if u["kind"] == "text":
                    if u["text"].strip():
                        feats[0, pos:pos + len(u["text"])] = \
                            fe.char_feats(u["text"], max_chars=L).cpu().numpy()[:len(u["text"])]
                    pos += len(u["text"])
                else:
                    pos += 1
            o = net(torch.tensor([ids]).cuda(), torch.tensor([spos]).cuda(),
                    torch.tensor([[v if v is not None else z for v in svals]]).cuda(),
                    torch.tensor([[v if v is not None else z for v in smask]]).cuda(),
                    torch.tensor([echo[i]]).cuda(),
                    torch.from_numpy(feats).cuda(), lengths=torch.tensor([L]).cuda())
            out.append({"pad": o["pad_norm"][0].tolist(),
                        "noul": {q: bool(o["noul_probs"][0, j, 1] > 0.5)
                                 for j, q in enumerate(battery.NOUL_IDS)},
                        "latency_s": 0.0, "usage": {}})
    return out, (time.perf_counter() - t0) / len(qs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref4b", help="裁判答卷 tag(如 zcode)")
    a = ap.parse_args()
    ref = load_sheet(a.ref)
    rw = [decode_top2(r["pad"])[:2] for r in ref]
    vib, vib_lat = vib_answers()
    rows = []
    for tag in sorted(os.path.basename(p)[8:-6] for p in glob.glob("data/answers_*.jsonl")
                      if tag_ok(path := p, a.ref)):
        sheet = load_sheet(tag)
        rows.append(dict(tag=tag, sheet=sheet))
    rows.append(dict(tag="vibrato", sheet=vib))
    print(f"全员对裁判 ref4b (n={len(ref)}): top1/top2=词解码一致, noul=五旗一致\n")
    print(f"{'选手':<12} {'延迟':>7} {'top1':>6} {'top2':>6} {'noul':>6} {'token'}")
    res = []
    for r in rows:
        s = r["sheet"]
        n = min(len(s), len(ref))
        t1 = t2 = nu = 0.0
        for i in range(n):
            if r["tag"] == "vibrato":
                w1, w2 = decode_top2(s[i]["pad"])[:2]
            else:
                w1, w2 = decode_top2(s[i]["pad"])[:2]
            t1 += w1 == rw[i][0]
            t2 += rw[i][0] in (w1, w2)
            nu += sum(s[i]["noul"][q] == ref[i]["noul"][q]
                      for q in battery.NOUL_IDS) / 5
        lat = vib_lat if r["tag"] == "vibrato" else (
            sum(x.get("latency_s", 0) for x in s) / max(1, len(s)))
        tok = sum((x["usage"].get("prompt") or 0) + (x["usage"].get("completion") or 0)
                  for x in s)
        res.append((t1 / n, r["tag"], t2 / n, nu / n, lat, tok))
    for t1, tag, t2, nu, lat, tok in sorted(res, reverse=True):
        print(f"{tag:<12} {lat:>6.1f}s {t1:>6.2f} {t2:>6.2f} {nu:>6.2f} "
              f"{tok if tok else '0(本地)'}")


if __name__ == "__main__":
    main()
