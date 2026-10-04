# -*- coding: utf-8 -*-
"""run_full — bundle 满血运行时(零 torch): onnxruntime + tokenizers + numpy。

内嵌双判头(fp32/int8 可选) + bge-m3 int8 + tokenizer.json + 词表。
依赖: pip install onnxruntime tokenizers numpy

用法:
  python run_full.py --head int8 --messages '[{"role":"user","message":"今天累死啦，快抱抱"}]'
  python run_full.py --head fp32 --demo          # 内置三例
输出: 每轮 PAD/族/noul/conf, 并按 v1/affect 把 pad 写回上一条 user 消息(回声链)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vibrato import vib_messages                      # 纯 python, 无重依赖
from vibrato.battery import FAMILY_IDS, NOUL_IDS

DEMOS = [
    [{"role": "user", "message": "今天累死啦，快抱抱"}],
    [{"role": "user", "message": "你刚才为什么要那样敷衍我？给我说清楚！"},
     {"role": "assistant", "message": "抱歉，我没有想敷衍你的意思。"},
     {"role": "user", "message": "算了，说了也没用。"}],
    [{"role": "assistant", "message": "今天过得怎么样呀？", "pad": [0.4, -0.2, 0.1]},
     {"role": "user", "message": "还行吧，就那样，一堆破事"}],
]


def load(head: str):
    mdir = os.path.join(HERE, "model_v6")
    vocab = json.load(open(os.path.join(mdir, "vocab.json"), encoding="utf-8"))
    tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer", "tokenizer.json"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    head_sess = ort.InferenceSession(
        os.path.join(mdir, f"models/v6/vibrato_v6_{head}.onnx"), so,
        providers=["CPUExecutionProvider"])
    bge = ort.InferenceSession(
        os.path.join(mdir, "models/v6/bge_m3_int8.onnx"), so,
        providers=["CPUExecutionProvider"])
    return vocab, len(vocab), tok, head_sess, bge


def bge_chars(bge, tok, text, max_chars):
    n = min(len(text), max_chars)
    out = np.zeros((max_chars, 1024), dtype=np.float32)   # 由调用方按总长分配
    enc = tok.encode(text, add_special_tokens=False)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.ones_like(ids)
    hidden = bge.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
    for (s, e), vec in zip(enc.offsets, hidden):
        lo, hi = s, min(e, n)
        if hi > lo:
            out[lo:hi] = vec
    return out


def score(messages, vocab, unk, tok, head_sess, bge, max_len=512):
    vib_messages.validate_messages(messages)
    echo = next((m["pad"] for m in reversed(messages[:-1])
                 if m["role"] == "user" and "pad" in m), [0.0] * 3)
    units = vib_messages.sequence_units(messages, max_window=max_len)
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
    feats = np.zeros((L, 1024), dtype=np.float32)
    pos = 0
    for u in units:
        if u["kind"] == "text":
            if u["text"].strip():
                f = bge_chars(bge, tok, u["text"], L)
                n = min(len(u["text"]), L - pos)
                feats[pos:pos + n] = f[:n]
            pos += len(u["text"])
        else:
            pos += 1
    o = head_sess.run(None, {
        "ids": np.array([ids], dtype=np.int64),
        "state_pos": np.array([spos], dtype=np.float32),
        "state_vals": np.array([[v if v is not None else z for v in svals]], dtype=np.float32),
        "state_mask": np.array([[v if v is not None else z for v in smask]], dtype=np.float32),
        "echo_prev": np.array([echo], dtype=np.float32),
        "feats": feats[None]})
    pad = o[2][0].tolist()
    fam = FAMILY_IDS[int(o[3][0].argmax())]
    noul = {q: bool(o[4][0, j, 1] > 0.5) for j, q in enumerate(NOUL_IDS)}
    return pad, fam, noul, float(o[5][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--messages", default=None, help="JSON 数组(v1/affect)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    vocab, unk, tok, head_sess, bge = load(a.head)
    convos = DEMOS if a.demo else [json.loads(a.messages)]
    for ci, msgs in enumerate(convos):
        for turn in range(len(msgs)):
            upto = msgs[:turn + 1]
            if upto[-1]["role"] != "user" or "message" not in upto[-1]:
                continue
            pad, fam, noul, conf = score(upto, vocab, unk, tok, head_sess, bge)
            print(f"[例{ci + 1}] 「{upto[-1]['message'][:18]}…」 "
                  f"PAD=[{pad[0]:+.2f},{pad[1]:+.2f},{pad[2]:+.2f}] 族={fam} "
                  f"conf={conf:.2f} noul={[k for k, v in noul.items() if v]}")
            msgs[turn]["pad"] = [round(x, 4) for x in pad]     # write_back 回声链


if __name__ == "__main__":
    main()
