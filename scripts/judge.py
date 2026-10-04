# -*- coding: utf-8 -*-
"""离线判卷器 — 答卷制另一半: 本地判卷, 零API, Vibrato每出新版随时重算。

用法: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=... python3 judge.py --ckpt ckpt_v51/vibrato.pt
自动发现 answers_*.jsonl(ref4b 为裁判卷), 逐挑战者出对比表 + token台账。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import battery  # noqa: E402
from vibrato import pad_schema  # noqa: E402
from answer_sheet import questions  # noqa: E402
from answer_sheet import echo_chain_from_ref  # noqa: E402
from vibrato.decode import decode_top2  # noqa: E402


def load_sheet(tag: str):
    return [json.loads(l) for l in open(f"data/answers_{tag}.jsonl", encoding="utf-8")
            if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=4, help="每挑战者分歧样本数")
    args = ap.parse_args()

    import torch
    from net import VibratoNet

    ck = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)
    is_v6 = "unk_id" in ck
    bad = pad_schema.verify_ckpt(ck, hard=is_v6)   # v6 硬闸; v5 跨代际对照取警告
    if bad:
        print(f"[warn] 契约指纹不一致(跨代际对照): {bad}")
    net = VibratoNet(vocab_size=len(ck["vocab"]) + (1 if is_v6 else 0),
                     max_len=ck["max_len"], feat_dim=ck.get("feat_dim", 0)).cuda().eval()
    net.load_state_dict(ck["model"])
    vocab, unk = ck["vocab"], ck.get("unk_id", 1)

    qs = questions()
    echo = echo_chain_from_ref()
    ref = load_sheet("ref4b")

    # Vibrato 答卷(一次算好, 对所有挑战者复用) + 计时
    vib = []
    t0 = time.perf_counter()
    if not is_v6:
        with torch.no_grad():
            for i, (_, _, text) in enumerate(qs):
                ids = torch.tensor([[vocab.get(ch, 1) for ch in text][-ck["max_len"] + 1:]]).cuda()
                e = torch.tensor([echo[i]]).cuda()
                out = net(ids, e)
                pad = out["pad_norm"][0].tolist()
                w1, w2, _ = decode_top2(pad)
                vib.append({"pad": pad, "w1": w1, "w2": w2,
                            "noul": {q: bool(out["noul_probs"][0, j, 1] > 0.5)
                                     for j, q in enumerate(battery.NOUL_IDS)}})
    else:
        # v6: v1/affect 消息范式作答。ref链PAD钉到上一条user消息(与echo锚同源,
        # 口径与v5判卷可比); 6条平价窗; bge特征现场算(满血路径)
        import numpy as np
        import vib_messages
        from feat_bge import BgeFeaturizer
        fe = BgeFeaturizer()
        convos = {c["id"]: c for c in (json.loads(l) for l
                                       in open("data/test_convos.jsonl", encoding="utf-8")
                                       if l.strip())}
        with torch.no_grad():
            for i, (cid, idx, _text) in enumerate(qs):
                msgs = convos[cid]["messages"][max(0, idx - 6):idx + 1]
                vmsgs, pin = [], None
                for j, m in enumerate(msgs):
                    vm = {"role": "user" if m["side"] == "user" else "assistant",
                          "message": str(m.get("text", ""))}
                    if j < len(msgs) - 1 and vm["role"] == "user":
                        pin = len(vmsgs)            # 最近一条历史user消息, 待钉ref链PAD
                    vmsgs.append(vm)
                if pin is not None and i > 0:
                    vmsgs[pin]["pad"] = echo[i]
                units = vib_messages.sequence_units(vmsgs, max_window=ck["max_len"])
                ids, spos, svals, smask = [], [], [], []
                for u in units:
                    if u["kind"] == "text":
                        for ch in u["text"]:
                            ids.append(vocab.get(ch, unk))
                            spos.append(False)
                        svals.extend([None] * len(u["text"]))
                        smask.extend([None] * len(u["text"]))
                    else:
                        ids.append(0)
                        spos.append(True)
                        svals.append(u["vals"])
                        smask.append(u["mask"])
                L = len(ids)
                z = [0.0] * vib_messages.STATE_DIM
                sv = [v if v is not None else z for v in svals]
                sm = [v if v is not None else z for v in smask]
                feats = None
                if ck["feat_dim"]:
                    feats = np.zeros((1, L, ck["feat_dim"]), dtype="float32")
                    pos = 0
                    for u in units:
                        if u["kind"] == "text":
                            if u["text"].strip():       # 空白单元(换行分隔)零特征, 与训练侧一致
                                f = fe.char_feats(u["text"], max_chars=L).cpu().numpy()
                                feats[0, pos:pos + len(u["text"])] = f[:len(u["text"])]
                            pos += len(u["text"])
                        else:
                            pos += 1
                out = net(torch.tensor([ids]).cuda(),
                          torch.tensor([spos]).cuda(), torch.tensor([sv]).cuda(),
                          torch.tensor([sm]).cuda(), torch.tensor([echo[i]]).cuda(),
                          torch.from_numpy(feats).cuda() if feats is not None else None,
                          lengths=torch.tensor([L]).cuda())
                pad = out["pad_norm"][0].tolist()
                w1, w2, _ = decode_top2(pad)
                vib.append({"pad": pad, "w1": w1, "w2": w2,
                            "noul": {q: bool(out["noul_probs"][0, j, 1] > 0.5)
                                     for j, q in enumerate(battery.NOUL_IDS)}})
    vib_ms = (time.perf_counter() - t0) / len(qs) * 1000

    print(f"判卷: {args.ckpt.rsplit('/',1)[-1]} {'(v1/affect)' if is_v6 else '(v5)'} | "
          f"n={len(qs)} | Vibrato {vib_ms:.1f}ms/条\n")
    tags = sorted(os.path.basename(p)[8:-6] for p in glob.glob("data/answers_*.jsonl")
                  if "ref4b" not in p)
    header = f"{'挑战者':<12} {'延迟':>8} {'top1':>6} {'top2':>6} {'noul':>6} {'MAE共识':>8} {'共识n':>5} {'token(P+C)'}"
    print(header)
    print("-" * len(header))
    for tag in tags:
        sheet = load_sheet(tag)
        n = min(len(sheet), len(qs))
        lat = sum(r.get("latency_s", 0) for r in sheet) / max(1, len(sheet))
        tok_p = sum(r["usage"].get("prompt") or 0 for r in sheet)
        tok_c = sum(r["usage"].get("completion") or 0 for r in sheet)
        top1 = top2 = noul_sum = 0.0
        cons_v = cons_q = cons_n = 0
        dis = []
        for i in range(n):
            r = sheet[i]
            v = vib[i]
            top1 += v["w1"] == r["family"] if False else 0  # family口径不同, 用词解码对比
            rw1, rw2, _ = decode_top2(r["pad"])
            top1 += v["w1"] == rw1
            top2 += rw1 in (v["w1"], v["w2"])
            noul_sum += sum(v["noul"][q] == r["noul"][q] for q in battery.NOUL_IDS) / 5
            # 共识判项: 挑战者PAD与裁判PAD均值差<0.30
            d_ref = sum(abs(a - b) for a, b in zip(r["pad"], ref[i]["pad"])) / 3
            if d_ref < 0.30:
                cons_n += 1
                cons_v += sum(abs(a - b) for a, b in zip(v["pad"], ref[i]["pad"])) / 3
                cons_q += d_ref
            if v["w1"] != rw1 and len(dis) < args.samples:
                dis.append((r["text_tail"][-36:], v["w1"], v["w2"], rw1))
        print(f"{tag:<12} {lat:>6.1f}s {top1/n:>6.2f} {top2/n:>6.2f} {noul_sum/n:>6.2f} "
              f"{(cons_q/max(1,cons_n)):>8.3f}/{(cons_v/max(1,cons_n)):.3f} {cons_n:>5} "
              f"{tok_p}+{tok_c}")
        for t, v1, v2, r1 in dis:
            print(f"    「…{t}」 vib={v1}/{v2} vs {r1}")
    print(f"\n[ref裁判4b] token≈0(本地) | Vibrato {vib_ms:.1f}ms vs 各挑战者秒级")
    print("注: MAE列=挑战者/Vibrato 各自与裁判PAD的偏差(仅共识判项); top1/top2=情绪词解码一致率")


if __name__ == "__main__":
    main()
