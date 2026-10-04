# -*- coding: utf-8 -*-
"""chordia 60k 金矿转换器 — pad_synthetic_output-test.jsonl → vibrato 训练行。

字段桥接:
  user_input_text → text_input (vib_state 孤立消息格式, 训练/推理同源)
  user_pad [-4,4] → 李克特 L = pad + 5 → likert_to_bins 帐篷核
  user_emotion    → 8 族映射(未映射词丢弃并计数)
  echo = 零向量(孤立消息); noul/conf 无标签 → 键省略(collate 可选键)

用法: python convert_legacy.py --in pad_synthetic_output-test.jsonl --out legacy_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import pad_schema  # noqa: E402
from vibrato.battery import FAMILY_IDS  # noqa: E402
from vibrato.vib_state import build_context  # noqa: E402

# 14 个实测词 → 8 族(语义映射; 未映射的罕见词丢弃)
WORD_TO_FAMILY = {
    "喜悦": "happy", "乐观": "happy",
    "依赖": "affectionate",
    "温和": "calm", "轻松": "calm", "无聊": "calm", "bored": "calm",
    "惊奇": "amused",
    "悲伤": "sad",
    "焦虑": "anxious", "恐惧": "anxious",
    "敌意": "angry", "愤懑": "angry", "厌恶": "angry", "藐视": "angry",
    "委屈": "wronged",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default="data/legacy_dataset.jsonl")
    args = ap.parse_args()

    n_ok = n_drop_word = n_drop_pad = n_drop_text = 0
    dropped_words: dict = {}
    with open(args.in_path, encoding="utf-8-sig") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = str(d.get("user_input_text", "")).strip()
            word = str(d.get("user_emotion", ""))
            pad = d.get("user_pad")
            if not text or len(text) < 2:
                n_drop_text += 1
                continue
            family = WORD_TO_FAMILY.get(word)
            if family is None:
                n_drop_word += 1
                dropped_words[word] = dropped_words.get(word, 0) + 1
                continue
            try:
                p, a, dd = (float(x) for x in pad[:3])
                pad_bins = {dim: pad_schema.likert_to_bins(v + 5.0)
                            for dim, v in zip(pad_schema.DIMS, (p, a, dd))}
            except Exception:
                n_drop_pad += 1
                continue
            msg = [{"side": "user", "text": text[:400]}]
            row = {
                "id": f"legacy-{i:06d}",
                "text_input": build_context(msg, 0),
                "echo": [0.0, 0.0, 0.0],
                "pad_bins": pad_bins,
                "family": family,
                "meta": {"source": "chordia60k", "word": word},
            }
            # 自检: 分布归一 + 族合法
            assert all(abs(sum(v) - 1.0) < 1e-6 for v in pad_bins.values())
            assert family in FAMILY_IDS
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"转换完成: {n_ok} 行 -> {args.out_path}")
    print(f"丢弃: 未映射词 {n_drop_word} {sorted(dropped_words.items(), key=lambda kv: -kv[1])[:5]}"
          f" | PAD异常 {n_drop_pad} | 空文本 {n_drop_text}")


if __name__ == "__main__":
    main()
