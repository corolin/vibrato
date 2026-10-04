# -*- coding: utf-8 -*-
"""eval_backbone — 嵌入底座对比: 同一 Vibrato 训练管线 + 不同底座 → 对独立裁判 r̄。

用法: PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 python3 eval_backbone.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import vib_messages
from answer_sheet import questions, echo_chain_from_ref
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

CONFIGS = [
    ("bge-m3 568M", "checkpoints/ckpt_v6o/vibrato.pt", "bge", 1024),
    ("bekko-a25m 123M", "ckpt_bekko25_s2/vibrato.pt",
     "/root/.cache/huggingface/hub/models--hotchpotch--bekko-embedding-v1-a25m/snapshots/44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c", 384),
    ("bekko-a8m 106M", "ckpt_bekko8_s2/vibrato.pt",
     "/root/.cache/huggingface/hub/models--hotchpotch--bekko-embedding-v1-a8m/snapshots/c721113d59a1d91b447450324f51c4b3332c924a", 384),
    ("e5-small 118M", "checkpoints/ckpt_e5s_s2/vibrato.pt",
     "/root/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3", 384),
]

zc = [json.loads(l) for l in open("data/answers_zcode.jsonl", encoding="utf-8")]
R = np.array([r["pad"] for r in zc])
convos = {c["id"]: c for c in (json.loads(l) for l
                               in open("data/test_convos.jsonl", encoding="utf-8")
                               if l.strip())}
qs, echo = questions(), echo_chain_from_ref()

print(f"{'底座':<18} {'dim':>4} {'r̄':>6} {'r(P)':>6} {'r(A)':>6} {'r(D)':>6} {'MAE':>6} {'词翻':>4}")
for name, ckpt, feat_src, dim in CONFIGS:
    ck = torch.load(ckpt, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    if feat_src == "bge":
        fe = BgeFeaturizer()
    else:
        from feat_any import AnyFeaturizer
        fe = AnyFeaturizer(feat_src)
    pads = []
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
            feats = np.zeros((L, ck["feat_dim"]), dtype=np.float32)
            pos = 0
            for u in units:
                if u["kind"] == "text":
                    if u["text"].strip():
                        f = fe.char_feats(u["text"], max_chars=L).cpu().numpy()
                        feats[pos:pos + len(u["text"])] = f[:len(u["text"])]
                    pos += len(u["text"])
                else:
                    pos += 1
            o = net(torch.tensor([ids]).cuda(), torch.tensor([spos]).cuda(),
                    torch.tensor([[v if v is not None else z for v in svals]]).cuda(),
                    torch.tensor([[v if v is not None else z for v in smask]]).cuda(),
                    torch.tensor([echo[i]]).cuda(),
                    torch.from_numpy(feats).cuda().unsqueeze(0),
                    lengths=torch.tensor([L]).cuda())
            pads.append(o["pad_norm"][0].cpu().tolist())
    del fe, net
    torch.cuda.empty_cache()
    P = np.array(pads)
    corr = [float(np.corrcoef(P[:, d], R[:, d])[0, 1]) for d in range(3)]
    mae = float(np.abs(P - R).mean())
    from decode import decode_top2
    flips = sum(decode_top2(P[i].tolist())[0] != decode_top2(R[i].tolist())[0]
                for i in range(40))
    print(f"{name:<18} {dim:>4} {np.mean(corr):>6.3f} {corr[0]:>6.3f} {corr[1]:>6.3f} "
          f"{corr[2]:>6.3f} {mae:>6.3f} {flips:>4}")
