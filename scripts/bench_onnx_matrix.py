# -*- coding: utf-8 -*-
"""bench_onnx_matrix — ONNX 全矩阵基准: 精度(fp32 vs int8, 判头×bge) + p50/p90/p99 + RSS。

精度口径 = 40 题真实判卷输入, 端到端 PAD 与生产基线(torch bge GPU + torch 判头)的偏差
+ 词解码 top1 翻转数; bge 另报特征级余弦。内存 = /proc VmRSS 加载增量。
用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge:/root/vibrato-ort \
      CUDA_VISIBLE_DEVICES=0 python3 bench_onnx_matrix.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from vibrato import vib_messages
from answer_sheet import questions, echo_chain_from_ref
from vibrato.decode import decode_top2
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

CKPT = "checkpoints/ckpt_v6o/vibrato.pt"
NOUL_KEYS = None


def rss_mb():
    with open("/proc/self/status") as fh:
        for ln in fh:
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) / 1024
    return 0.0


def pstats(ts):
    ts = sorted(ts)[max(0, len(ts) // 10):]          # 掐掉前10%冷启动
    n = len(ts)
    q = lambda p: ts[min(n - 1, int(n * p))] * 1000
    return q(0.5), q(0.9), q(0.99)


def build_cases():
    """40 题真实输入(判卷同款) + torch-GPU bge 参考特征。"""
    ck = torch.load(CKPT, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    fe = BgeFeaturizer()
    convos = {c["id"]: c for c in (json.loads(l) for l
                                   in open("data/test_convos.jsonl", encoding="utf-8")
                                   if l.strip())}
    qs, echo = questions(), echo_chain_from_ref()
    cases = []
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
            feats = np.zeros((L, ck["feat_dim"]), dtype="float32")
            pos = 0
            unit_texts = []
            for u in units:
                if u["kind"] == "text":
                    if u["text"].strip():
                        f = fe.char_feats(u["text"], max_chars=L).cpu().numpy()
                        feats[pos:pos + len(u["text"])] = f[:len(u["text"])]
                        unit_texts.append((pos, len(u["text"]), u["text"]))
                    pos += len(u["text"])
                else:
                    pos += 1
            o = net(torch.tensor([ids]).cuda(), torch.tensor([spos]).cuda(),
                    torch.tensor([[v if v is not None else z for v in svals]]).cuda(),
                    torch.tensor([[v if v is not None else z for v in smask]]).cuda(),
                    torch.tensor([echo[i]]).cuda(),
                    torch.from_numpy(feats).cuda().unsqueeze(0),
                    lengths=torch.tensor([L]).cuda())
            cases.append(dict(ids=ids, spos=spos, svals=svals, smask=smask,
                              echo=echo[i], feats=feats, L=L, unit_texts=unit_texts,
                              pad_ref=o["pad_norm"][0].cpu().numpy()))
    return cases, ck


def bge_feats_onnx(sess, tok, text, max_chars):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
              truncation=True, max_length=1024, return_tensors="np")
    offsets = enc.pop("offset_mapping")[0].tolist()
    feeds = {k: v for k, v in enc.items()}
    hidden = sess.run(None, feeds)[0][0]              # (T, 1024)
    out = np.zeros((min(len(text), max_chars), hidden.shape[1]), dtype="float32")
    n = out.shape[0]
    for t, (s, e) in enumerate(offsets):
        lo, hi = s, min(e, n)
        if hi > lo:
            out[lo:hi] = hidden[t]                      # (1024,) 广播到字符区间
    return out


def main():
    print("构建 40 题真实输入(含 torch-GPU 参考特征)…", flush=True)
    cases, ck = build_cases()

    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/models/bge-m3")

    def head_sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    r0 = rss_mb()
    hs = {}
    for tag in ("fp32", "int8"):
        hs[tag] = head_sess(f"models/v6/vibrato_v6_{tag}.onnx")
    r_head = rss_mb() - r0
    print(f"判头会话加载 RSS 增量: {r_head:.0f} MB")

    bs = {}
    r0 = rss_mb()
    for tag in ("fp32", "int8"):
        so = ort.SessionOptions()
        bs[tag] = ort.InferenceSession(f"bge_onnx/bge_m3_{tag}.onnx", so,
                                       providers=["CPUExecutionProvider"])
    r_bge = rss_mb() - r0
    print(f"bge 会话加载 RSS 增量: {r_bge:.0f} MB (fp32+int8 常驻)")

    # ── bge 特征级偏差(onnx CPU vs torch GPU 参考) + 延迟 ──
    print("\n== bge-m3 特征级: onnx-CPU vs torch-GPU 参考 ==")
    feat_cache = {}
    for tag in ("fp32", "int8"):
        devs, ts = [], []
        for c in cases:
            for pos, n, text in c["unit_texts"]:
                t0 = time.perf_counter()
                f = bge_feats_onnx(bs[tag], tok, text, c["L"])
                ts.append(time.perf_counter() - t0)
                ref = c["feats"][pos:pos + n]
                m = min(len(f), n)
                a, b = f[:m], ref[:m]
                cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                                         * np.linalg.norm(b, axis=-1) + 1e-9)
                devs.extend(cos.tolist())
        p50, p90, p99 = pstats(ts)
        print(f"bge-{tag}: 逐token余弦 均值={np.mean(devs):.5f} 最小={np.min(devs):.4f} "
              f"| {len(ts)}单元 p50/p90/p99 = {p50:.1f}/{p90:.1f}/{p99:.1f} ms")
        feat_cache[tag] = devs

    # ── 端到端五组合 ──
    print("\n== 端到端(40题): PAD 偏差 vs 生产基线(torch bge GPU + torch 判头) ==")

    def run_head(sess, feats):
        ids = np.array([c["ids"] for c in cases], dtype=object)
        out = []
        for c, F in zip(cases, feats):
            o = sess.run(None, {
                "ids": np.array([c["ids"]], dtype=np.int64),
                "state_pos": np.array([c["spos"]], dtype=np.float32),
                "state_vals": np.array([[v if v is not None else [0.0]*5
                                         for v in c["svals"]]], dtype=np.float32),
                "state_mask": np.array([[v if v is not None else [0.0]*5
                                         for v in c["smask"]]], dtype=np.float32),
                "echo_prev": np.array([c["echo"]], dtype=np.float32),
                "feats": F.astype(np.float32)[None]})[2]        # pad_norm
            out.append(o[0])
        return np.array(out)

    ref_pads = np.array([c["pad_ref"] for c in cases])
    ref_words = [decode_top2(p.tolist())[0] for p in ref_pads]
    torch_feats = [c["feats"] for c in cases]
    zero_feats = [np.zeros_like(c["feats"]) for c in cases]

    # bge-onnx 特征重组(按题)
    onnx_feats = {"fp32": [], "int8": []}
    for tag in ("fp32", "int8"):
        for c in cases:
            F = np.zeros_like(c["feats"])
            for pos, n, text in c["unit_texts"]:
                f = bge_feats_onnx(bs[tag], tok, text, c["L"])
                F[pos:pos + n] = f[:n]
            onnx_feats[tag].append(F)

    combos = [
        ("A 判头fp32 + bge参考", hs["fp32"], torch_feats),
        ("B 判头int8 + bge参考", hs["int8"], torch_feats),
        ("C 判头fp32 + bge-fp32-CPU", hs["fp32"], onnx_feats["fp32"]),
        ("D 判头int8 + bge-int8-CPU(全int8满血)", hs["int8"], onnx_feats["int8"]),
        ("E 判头int8 + 零特征(纯文字)", hs["int8"], zero_feats),
    ]
    results = {}
    for name, sess, feats in combos:
        P = run_head(sess, feats)
        dev = np.abs(P - ref_pads)
        flips = sum(decode_top2(p.tolist())[0] != w for p, w in zip(P, ref_words))
        results[name] = P
        print(f"{name:<28} PAD均值偏差={dev.mean():.4f} 最大={dev.max():.4f} "
              f"| 词解码top1翻转 {flips}/40")

    # ── 判头延迟(真实长度逐题) ──
    print("\n== 判头延迟(40题真实长度, CPU单线程) ==")
    for tag in ("fp32", "int8"):
        sess = hs[tag]
        feeds0 = [{
            "ids": np.array([c["ids"]], dtype=np.int64),
            "state_pos": np.array([c["spos"]], dtype=np.float32),
            "state_vals": np.array([[v if v is not None else [0.0]*5
                                     for v in c["svals"]]], dtype=np.float32),
            "state_mask": np.array([[v if v is not None else [0.0]*5
                                     for v in c["smask"]]], dtype=np.float32),
            "echo_prev": np.array([c["echo"]], dtype=np.float32),
            "feats": c["feats"].astype(np.float32)[None]} for c in cases]
        for _ in range(3):
            for f in feeds0:
                sess.run(None, f)
        ts = []
        for _ in range(5):
            for f in feeds0:
                t0 = time.perf_counter()
                sess.run(None, f)
                ts.append(time.perf_counter() - t0)
        p50, p90, p99 = pstats(ts)
        Ls = sorted(c["L"] for c in cases)
        print(f"判头-{tag}: p50/p90/p99 = {p50:.2f}/{p90:.2f}/{p99:.2f} ms "
              f"(题长 p50={Ls[len(Ls)//2]}, max={Ls[-1]})")

    # ── 满血全 int8 端到端延迟(bge int8 + 判头 int8, 纯 CPU) ──
    print("\n== 纯CPU满血档端到端(bge-int8 + 判头-int8) ==")
    ts = []
    for c in cases:
        t0 = time.perf_counter()
        F = np.zeros_like(c["feats"])
        for pos, n, text in c["unit_texts"]:
            F[pos:pos + n] = bge_feats_onnx(bs["int8"], tok, text, c["L"])[:n]
        hs["int8"].run(None, {
            "ids": np.array([c["ids"]], dtype=np.int64),
            "state_pos": np.array([c["spos"]], dtype=np.float32),
            "state_vals": np.array([[v if v is not None else [0.0]*5
                                     for v in c["svals"]]], dtype=np.float32),
            "state_mask": np.array([[v if v is not None else [0.0]*5
                                     for v in c["smask"]]], dtype=np.float32),
            "echo_prev": np.array([c["echo"]], dtype=np.float32),
            "feats": F.astype(np.float32)[None]})
        ts.append(time.perf_counter() - t0)
    p50, p90, p99 = pstats(ts)
    print(f"端到端 p50/p90/p99 = {p50:.0f}/{p90:.0f}/{p99:.0f} ms/题")


if __name__ == "__main__":
    main()
