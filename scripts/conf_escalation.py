# -*- coding: utf-8 -*-
"""conf_escalation — System One 升级曲线: 低置信样本转交 LLM 后 r̄ 涨多少。

对 40 题: vib 答卷(含 conf) 按 conf 升序, 把最低 X% 换成升级模型的 pad,
其余保留 vib —— 对独立裁判 zcode 算混合 r̄。量化"conf 知道自己何时不该答"。
用法: PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 python3 conf_escalation.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import vib_messages
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet


def vib_conf(ck_path="checkpoints/ckpt_v6o/vibrato.pt"):
    import json as _json
    ck = torch.load(ck_path, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    fe = BgeFeaturizer()
    convos = {c["id"]: c for c in (_json.loads(l) for l
                                   in open("data/test_convos.jsonl", encoding="utf-8")
                                   if l.strip())}
    from answer_sheet import questions, echo_chain_from_ref
    qs, echo = questions(), echo_chain_from_ref()
    pads, confs = [], []
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
            pads.append(o["pad_norm"][0].tolist())
            confs.append(float(o["conf"][0]))
    return np.array(pads), np.array(confs)


def rbar(P, R):
    return float(np.mean([np.corrcoef(P[:, d], R[:, d])[0, 1] for d in range(3)]))


def main():
    zc = [json.loads(l) for l in open("data/answers_zcode.jsonl", encoding="utf-8")]
    R = np.array([r["pad"] for r in zc])
    V, C = vib_conf()
    escs = {t: np.array([json.loads(l)["pad"] for l in
                         open(f"data/answers_{t}.jsonl", encoding="utf-8") if l.strip()][:40])
            for t in ("sonnet55", "glm53")}
    order = np.argsort(C)          # conf 最低在前
    print(f"vib conf 范围 [{C.min():.2f}, {C.max():.2f}] 均值 {C.mean():.2f}")
    print(f"{'升级比例':>6} {'API调用/40题':>10} {'→sonnet55 r̄':>12} {'→glm53 r̄':>10}")
    for frac in (0.0, 0.1, 0.2, 0.3, 0.5):
        k = int(round(40 * frac))
        take = order[:k]
        line = f"{frac:>6.0%} {k:>10d}"
        for t, P_esc in escs.items():
            P = V.copy()
            P[take] = P_esc[take]
            line += f" {rbar(P, R):>12.3f}" if t == "sonnet55" else f" {rbar(P, R):>10.3f}"
        print(line)


if __name__ == "__main__":
    main()
