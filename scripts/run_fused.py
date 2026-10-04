# -*- coding: utf-8 -*-
"""run_fused — 融合单文件运行时: 一个 ONNX(bge int8+判头 int8 全融合), 一次 run。

依赖: pip install onnxruntime tokenizers numpy
用法:
  python run_fused.py --demo
  python run_fused.py --messages '[{"role":"user","message":"…"}]'
输入契约: 原六项 + bge_ids/bge_mask [Nseg,T] + char_seg/char_tok [B,L]
         (调用方用 tokenizer.json 分词并按 offsets 建映射; 分词是数据不是模型)。
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
from vibrato import vib_messages
from vibrato.battery import FAMILY_IDS, NOUL_IDS

MODEL = os.path.join(HERE, "models/v6/vibrato_fused_int8.onnx")
VOCAB = os.path.join(HERE, "vocab.json")
TOK = os.path.join(HERE, "tokenizer.json")

DEMOS = [
    [{"role": "user", "message": "今天累死啦，快抱抱"}],
    [{"role": "user", "message": "终于等到你啦~ 今天累死本宝宝了，快抱抱！"}],
    [{"role": "user", "message": "你刚才为什么要那样敷衍我？给我说清楚！"},
     {"role": "assistant", "message": "抱歉，我没有想敷衍你的意思。"},
     {"role": "user", "message": "算了，说了也没用。"}],
]


def load():
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    tok = Tokenizer.from_file(TOK)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return vocab, len(vocab), tok, sess


def score(messages, vocab, unk, tok, sess, max_len=512):
    echo = next((m["pad"] for m in reversed(messages[:-1])
                 if m["role"] == "user" and "pad" in m), [0.0] * 3)
    units = vib_messages.sequence_units(messages, max_window=max_len)
    ids, spos, svals, smask, segs = [], [], [], [], []
    char_seg, char_tok = [], []
    for u in units:
        if u["kind"] == "text":
            for ch in u["text"]:
                ids.append(vocab.get(ch, unk))
                spos.append(False)
            svals.extend([None] * len(u["text"]))
            smask.extend([None] * len(u["text"]))
            if u["text"].strip():
                enc = tok.encode(u["text"], add_special_tokens=False)
                s = len(segs)
                segs.append(enc)
                cs = [-1] * len(u["text"])
                ct = [-1] * len(u["text"])
                for t, (a, b) in enumerate(enc.offsets):
                    for k in range(a, min(b, len(u["text"]))):
                        cs[k], ct[k] = s, t
                char_seg.extend(cs)
                char_tok.extend(ct)
            else:
                char_seg.extend([-1] * len(u["text"]))
                char_tok.extend([-1] * len(u["text"]))
        else:
            ids.append(0)
            spos.append(True)
            svals.append(u["vals"])
            smask.append(u["mask"])
            char_seg.append(-1)
            char_tok.append(-1)
    T = max(len(e.ids) for e in segs) if segs else 1
    bge_ids = np.zeros((len(segs), T), dtype=np.int64)
    bge_mask = np.zeros((len(segs), T), dtype=np.int64)
    for i, e in enumerate(segs):
        bge_ids[i, :len(e.ids)] = e.ids
        bge_mask[i, :len(e.ids)] = 1
    z = [0.0] * 5
    o = sess.run(None, {
        "ids": np.array([ids], dtype=np.int64),
        "state_pos": np.array([spos], dtype=np.float32),
        "state_vals": np.array([[v if v is not None else z for v in svals]],
                               dtype=np.float32),
        "state_mask": np.array([[v if v is not None else z for v in smask]],
                                dtype=np.float32),
        "echo_prev": np.array([echo], dtype=np.float32),
        "bge_ids": bge_ids, "bge_mask": bge_mask,
        "char_seg": np.array([char_seg], dtype=np.int64),
        "char_tok": np.array([char_tok], dtype=np.int64)})
    pad = o[2][0].tolist()
    return (pad, FAMILY_IDS[int(o[3][0].argmax())],
            {q: bool(o[4][0, j, 1] > 0.5) for j, q in enumerate(NOUL_IDS)},
            float(o[5][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", default=None)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    vocab, unk, tok, sess = load()
    convos = DEMOS if a.demo else [json.loads(a.messages)]
    for ci, msgs in enumerate(convos):
        for turn in range(len(msgs)):
            upto = msgs[:turn + 1]
            if upto[-1]["role"] != "user" or "message" not in upto[-1]:
                continue
            pad, fam, noul, conf = score(upto, vocab, unk, tok, sess)
            print(f"[例{ci + 1}] 「{upto[-1]['message'][:18]}…」 "
                  f"PAD=[{pad[0]:+.2f},{pad[1]:+.2f},{pad[2]:+.2f}] 族={fam} "
                  f"conf={conf:.2f} noul={[k for k, v in noul.items() if v]}")
            msgs[turn]["pad"] = [round(x, 4) for x in pad]


if __name__ == "__main__":
    main()
