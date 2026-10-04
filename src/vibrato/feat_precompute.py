# -*- coding: utf-8 -*-
"""feat_precompute — v6 训练集 bge 特征预计算 (piece ④-c 前置)

bge-m3 冻结(on-the-fly 每epoch 46k×8ep×~15ms ≈ 92min 太贵), 一次性预计算
唯一文本单元特征 → fp16 memmap + sha1 索引。dropout_forms 只整删单元、不改文本,
故按单元哈希存特征, 任意增广形态都能重组。

输出: feats_v6.mm (fp16, [总字符位, 1024]) + feats_v6_index.json {sha1(text): [off, len]}
用法: cd /root/vibrato && PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge \
      CUDA_VISIBLE_DEVICES=0 python3 feat_precompute.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from . import vib_messages  # noqa: E402

TRAIN, VAL = "data/v6_train.jsonl", "data/v6_val.jsonl"
MM, INDEX = "feats_v6.mm", "feats_v6_index.json"
from .feat_bge import BgeFeaturizer, FEAT_DIM  # noqa: E402


def unit_texts(rows):
    seen = {}
    for r in rows:
        for u in vib_messages.sequence_units(r["messages"]):
            if u["kind"] == "text" and u["text"].strip():
                seen.setdefault(u["text"], None)
    return list(seen)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--featurizer", default="bge",
                    help="bge=默认BgeFeaturizer; 否则=模型目录路径(AnyFeaturizer)")
    ap.add_argument("--tag", default=None, help="输出名后缀(如 bekko25 → feats_bekko25.mm)")
    a = ap.parse_args()
    mm_name = MM if a.tag is None else f"feats_{a.tag}.mm"
    idx_name = INDEX if a.tag is None else f"feats_{a.tag}_index.json"
    if a.featurizer == "bge":
        fe = BgeFeaturizer()
    else:
        from feat_any import AnyFeaturizer
        fe = AnyFeaturizer(a.featurizer)
        print(f"底座: {a.featurizer} dim={fe.dim}")
    rows = []
    for p in (TRAIN, VAL):
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.loads(ln) for ln in fh if ln.strip())
    texts = unit_texts(rows)
    n_chars = sum(len(t) for t in texts)
    print(f"{len(rows)} 行 → {len(texts)} 唯一文本单元, {n_chars:,} 字符位, "
          f"预计 memmap {n_chars * FEAT_DIM * 2 / 1e9:.2f} GB")

    done = {}
    if os.path.exists(idx_name):
        done = json.load(open(idx_name, encoding="utf-8"))
        texts = [t for t in texts if hashlib.sha1(t.encode()).hexdigest() not in done]
        print(f"断点续跑: 已有 {len(done)} 单元, 剩 {len(texts)}")

    with open(mm_name, "ab") as mm:                     # 追加写; 索引记录偏移(单位: 字符位)
        t0 = time.time()
        for i, t in enumerate(texts):
            f = fe.char_feats(t).half().cpu()      # (len, 1024) fp16
            assert f.shape == (len(t), fe.dim), (t[:20], f.shape)
            off = os.path.getsize(mm_name) // (fe.dim * 2)
            mm.write(f.numpy().tobytes())
            mm.flush()
            done[hashlib.sha1(t.encode()).hexdigest()] = [off, len(t)]
            if (i + 1) % 2000 == 0:
                json.dump(done, open(idx_name, "w", encoding="utf-8"))
                print(f"  {i + 1}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(done, open(idx_name, "w", encoding="utf-8"))
    print(f"完成: {len(done)} 单元 → {MM} + {INDEX}, 用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
