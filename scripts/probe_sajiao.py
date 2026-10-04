# -*- coding: utf-8 -*-
"""probe_sajiao — 撒娇极性病灶定位: echo链污染 vs 知识缺口。

对判卷里的亲昵案, 三种配置对比: (a)判卷原样(ref链echo+钉pad) (b)零echo无状态
(单条冷读) (c)零echo+满6条窗。输出词解码 + family头两强 + PAD。
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/vibrato")
import numpy as np
import torch

from vibrato import battery
from vibrato import vib_messages
from answer_sheet import questions, echo_chain_from_ref
from vibrato.decode import decode_top2
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

ck = torch.load("ckpt_v6h/vibrato.pt", map_location="cuda:0", weights_only=False)
vocab, unk = ck["vocab"], ck["unk_id"]
net = VibratoNet(vocab_size=len(vocab) + 1, max_len=ck["max_len"],
                 feat_dim=ck["feat_dim"]).cuda().eval()
net.load_state_dict(ck["model"])
fe = BgeFeaturizer()
qs = questions()
echo = echo_chain_from_ref()
convos = {c["id"]: c for c in (json.loads(l) for l
                               in open("data/test_convos.jsonl", encoding="utf-8") if l.strip())}

KEYS = ["终于等到你啦", "抱枕嘛", "快抱抱", "睡前故事", "在忙啥呢"]


@torch.no_grad()
def run(vmsgs, echo_vec):
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
    out = net(torch.tensor([ids]).cuda(), torch.tensor([spos]).cuda(),
              torch.tensor([[v if v is not None else z for v in svals]]).cuda(),
              torch.tensor([[v if v is not None else z for v in smask]]).cuda(),
              torch.tensor([echo_vec]).cuda(),
              torch.from_numpy(feats).cuda(), lengths=torch.tensor([L]).cuda())
    fam = out["family_probs"][0].cpu()
    top2 = fam.topk(2)
    pad = out["pad_norm"][0].tolist()
    w1, w2, _ = decode_top2(pad)
    return w1, w2, pad, [(battery.FAMILY_IDS[i], v.item()) for i, v in
                         zip(top2.indices, top2.values)]


print(f"{'案':22s} {'判卷原样':16s} {'冷读(零echo单条)':16s} {'冷读+窗':16s} family头")
for i, (cid, idx, _t) in enumerate(qs):
    text = convos[cid]["messages"][idx].get("text", "")
    if not any(k in text for k in KEYS):
        continue
    msgs = convos[cid]["messages"][max(0, idx - 6):idx + 1]
    # (a) 判卷原样
    va, pin = [], None
    for j, m in enumerate(msgs):
        vm = {"role": "user" if m["side"] == "user" else "assistant",
              "message": str(m.get("text", ""))}
        if j < len(msgs) - 1 and vm["role"] == "user":
            pin = len(va)
        va.append(vm)
    if pin is not None and i > 0:
        va[pin]["pad"] = echo[i]
    a = run(va, echo[i] if i > 0 else [0.0, 0.0, 0.0])
    # (b) 冷读单条
    b = run([{"role": "user", "message": text}], [0.0, 0.0, 0.0])
    # (c) 冷读满窗
    vc = [{"role": ("user" if m["side"] == "user" else "assistant"),
           "message": str(m.get("text", ""))} for m in msgs]
    c = run(vc, [0.0, 0.0, 0.0])
    tail = text[-14:]
    print(f"{tail:22s} {a[0]+'/'+a[1]:16s} {b[0]+'/'+b[1]:16s} {c[0]+'/'+c[1]:16s} "
          f"{a[3][0][0]}/{a[3][1][0]} | 冷读family={b[3][0][0]}")
    print(f"{'':22s} PADa={[round(x,2) for x in a[2]]} PADb={[round(x,2) for x in b[2]]}")
