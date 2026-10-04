# -*- coding: utf-8 -*-
"""teacher 全题库标注冒烟: battery.build_teacher_prompt(18项+族+是非一次出) → 4B。

用法(服务器上): cd /root/vibrato && python3 teacher_smoke.py
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import battery  # noqa: E402  (内部引用 pad_schema)
from vibrato import pad_schema  # noqa: E402

ENDPOINT = "http://localhost:11434/api/chat"
MODEL = os.environ.get("TEACHER_MODEL", "qwen3.5:4b")

TESTS = [
    ("哈哈哈哈今天也太顺利了吧，中午还抽中了奶茶！", None),
    ("算了，无所谓了，你忙你的吧。", [-0.1, -0.5, 0.0]),
    ("这个方案你再这样改下去我真的要炸了，说了多少遍前提不一样！", [-0.3, 0.4, 0.1]),
    ("哦，我没事，挺好的，真的。", [-0.4, -0.2, -0.3]),
]


def label(state: str, prev_pad):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": battery.build_teacher_prompt(state, prev_pad)}],
        "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0.3, "num_predict": 640},
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    dt = time.time() - t0
    raw = r["message"]["content"].strip()
    return battery.parse_teacher_json(raw), dt, r.get("eval_count", 0)


def main():
    zh = {f["id"]: f["zh"] for f in battery.FAMILY_LABELS}
    print(f"题库指纹 {battery.battery_digest()[:16]} | teacher={MODEL}\n")
    for text, prev in TESTS:
        state = f"TARGET message to judge:\n[them]: {text}"
        parsed, dt, ntok = label(state, prev)
        bins = pad_schema.items_to_bins(parsed["items"])
        pad = pad_schema.bins_to_pad(bins)
        dp = pad_schema.pad_to_pressure_delta(
            pad["pleasure"], pad["arousal"], pad["dominance"])
        flags = [q["id"] for q in battery.NOUL_QUESTIONS if parsed["noul"][q["id"]]]
        print(f"[{dt:.1f}s / {ntok} tok] {text}")
        print(f"  情绪族: {zh[parsed['family']]}  PAD="
              f"({pad['pleasure']:+.2f},{pad['arousal']:+.2f},{pad['dominance']:+.2f})  "
              f"压力Δ={dp:+.2f}")
        print(f"  是非=真: {flags if flags else '(无)'}  | {parsed['analysis'][:40]}")
        print()


if __name__ == "__main__":
    main()
