# -*- coding: utf-8 -*-
"""answer_clef — Clef 答卷收集：v1/affect 消息 → Clef state + questions → PAD。

Clef 是通用 schema 分类器(不输出 PAD 数值)，所以 score 题用李克特 18 项
（与 battery.py teacher prompt 同一量表），choice 题问 8 族——这样答卷
可直接进 judge.py / vs_ref.py 现有管线。

用法: python3 answer_clef.py --base https://decision.mirages.cc --key sk-clef-...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import pad_schema
from vibrato.battery import FAMILY_IDS
from vibrato.vib_state import build_context

BASE = "https://decision.mirages.cc"
KEY = ""


def call_clef(state: str, questions: dict) -> dict:
    payload = {"model": "clef-flash", "state": state, "questions": questions}
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r


def build_questions() -> dict:
    """构建 18 项 score + 8 族 choice（与 battery teacher 同一量表）。"""
    questions = {}
    # score: 18 项李克特 1-9
    item_defs = [
        "I1 快乐", "I2 高兴", "I3 满意", "I4 惬意", "I5 希望", "I6 放松",
        "I7 兴奋", "I8 警觉", "I9 刺激", "I10 狂热", "I11 活跃", "I12 惊慌",
        "I13 支配", "I14 影响", "I15 领导", "I16 重要", "I17 自由", "I18 强力",
    ]
    for i, name in enumerate(item_defs, 1):
        questions[f"I{i}"] = {
            "type": "score",
            "instructions": f"TARGET 消息发送者的「{name}」程度（1=完全不符合, 9=完全符合）",
        }
    # choice: 8 族
    questions["family"] = {
        "type": "choice",
        "instructions": "TARGET 消息发送者的主导情绪族",
        "criteria": {fid: fid for fid in FAMILY_IDS},
    }
    return questions


def answers_to_row(answers: dict, idx: int, convo_id: str, text_tail: str,
                   latency_s: float) -> dict:
    """Clef answers → answer_sheet 格式行（pad_bins + family）。"""
    # 18 项 → pad_bins
    items = {}
    for i in range(1, 19):
        v = answers.get(f"I{i}", {}).get("score", 5.0)
        items[f"I{i}"] = max(1.0, min(9.0, float(v)))
    bins = pad_schema.items_to_bins(items)
    pad = pad_schema.bins_to_pad(bins)
    fam = answers.get("family", {}).get("choice", "calm")
    if fam not in FAMILY_IDS:
        fam = "calm"
    return {
        "idx": idx, "convo": convo_id, "text_tail": text_tail,
        "latency_s": round(latency_s, 2),
        "usage": {"prompt": 0, "completion": 0},
        "items": items,
        "pad": [pad[d] for d in pad_schema.DIMS],
        "pad_bins": bins,
        "family": fam,
        "noul": {"directed_at_me": False, "suppressed": False},  # Clef 不做 noul
    }


def main():
    global BASE, KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", default="clef-flash")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    BASE, KEY = args.base, args.key
    out_file = args.out or f"data/answers_{args.tag}.jsonl"

    from answer_sheet import questions, echo_chain_from_ref
    qs = questions()
    echo = echo_chain_from_ref()
    questions_schema = build_questions()

    done = 0
    if os.path.exists(out_file):
        done = sum(1 for l in open(out_file, encoding="utf-8") if l.strip())
    print(f"题数 {len(qs)}, 已答 {done}")

    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(done, len(qs)):
            convo_id, idx, text = qs[i]
            t0 = time.time()
            try:
                r = call_clef(text, questions_schema)
                row = answers_to_row(r["answers"], i, convo_id, text[-60:],
                                     time.time() - t0)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(qs)}] {row['latency_s']}s "
                          f"pad={row['pad'][:2]} fam={row['family']}", flush=True)
            except Exception as e:
                print(f"  [{i}] 失败: {e}", file=sys.stderr)
                time.sleep(2)
    print(f"完成 → {out_file}")


if __name__ == "__main__":
    main()
