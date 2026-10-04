# -*- coding: utf-8 -*-
"""gen_sajiao — 亲昵撒娇定向补数据 (v6.1 补刀, 修正 affectionate 族 P 维标签偏置)

背景: 训练集 affectionate 族 5675 行 gold P 均值 -0.005(legacy "依赖"词质心
P-0.16 的语义漂移拖平), 而五天王共识与 OCC love 自评(P+2.93/4)都说亲昵是
愉悦的。本脚本产 2000 条亲昵撒娇消息(自评PAD, 不引导P), 行内转 v5 单行
训练行(family=affectionate), 供 prep_v6 并入。

用法: OCC_BASE_URL=https://open.cherryin.net/v1 OCC_API_KEY=sk-.. \
      python3 gen_sajiao.py --n 2000
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.request

from vibrato import pad_schema

OUT = "data/sajiao_rows.jsonl"
V5_PREFIX = "TARGET(需判断): [用户]: "

FLAVORS = [
    ("clingy", "粘人求陪伴(要抱抱/不许走/再聊一会儿)"),
    ("praise", "求夸奖求关注(夸夸我/看我嘛/我棒不棒)"),
    ("comfort", "求安慰式撒娇(委屈巴巴/要哄/哼)"),
    ("reunion", "重逢撒娇(终于等到你/想死你了/你去哪了)"),
    ("bedtime", "睡前软乎(讲睡前故事/晚安吻/软乎乎)"),
    ("tease", "撒娇打闹(你是坏蛋/不理你了/哼唧)"),
]

PROMPT = """你是中文聊天语料作家。写 {n} 条"用户发给AI伴侣"的微信风格消息, 全部是同一种情绪: 亲昵撒娇({flavor})——{desc}。

要求:
1. 每条独立成句, 8~40字, 口语化, 可带语气词/emoji(🥺~嘛 哼 嘿嘿), 像真的在跟AI伴侣撒娇
2. 场景多样化: 工作/学习/游戏/日常/感情/追剧/天气, 不要重复句式
3. 撒娇对象是AI伴侣本身
4. 每条同时给出该消息的PAD自评 pad:[p,a,d], 各维∈[-4,4]:
   p愉悦(-4很负面~4很正面), a唤醒(-4很平静~4很激动), d支配(-4顺从粘人~4强势主导)

只输出JSON: {{"cases":[{{"text":"…","pad":[p,a,d]}}, …]}}"""


def call_api(base: str, key: str, prompt: str):
    payload = {
        "model": "deepseek/deepseek-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3000, "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return (r["choices"][0]["message"]["content"].strip(),
            int((r.get("usage") or {}).get("prompt_tokens", 0)),
            int((r.get("usage") or {}).get("completion_tokens", 0)))


def to_row(i, case, flavor):
    text = str(case["text"]).strip()
    pad = [max(-4.0, min(4.0, float(v))) for v in case["pad"][:3]]
    if not text:
        return None
    bins = {d: pad_schema.likert_to_bins(pad[k] + 5.0)   # [-4,4]+5 → [1,9], legacy 同式
            for k, d in enumerate(pad_schema.DIMS)}
    return {"id": f"sj-{i:05d}", "text_input": V5_PREFIX + text,
            "echo": [0.0, 0.0, 0.0], "pad_bins": bins,
            "family": "affectionate", "meta": {"flavor": flavor}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    base = (os.environ.get("OCC_BASE_URL") or os.environ.get("TEACHER_BASE_URL")
            or "https://open.cherryin.net/v1")
    key = os.environ.get("OCC_API_KEY") or os.environ.get("TEACHER_API_KEY")
    if not key:
        sys.exit("缺少 API key (OCC_API_KEY/TEACHER_API_KEY)")

    n_batches = (args.n + args.batch - 1) // args.batch
    rng = random.Random(7)
    jobs = [(FLAVORS[i % len(FLAVORS)][0], FLAVORS[i % len(FLAVORS)][1])
            for i in range(n_batches)]          # 风味轮换
    print(f"{args.n} 条 ÷ {args.batch}/批 = {n_batches} 批, 风味轮换 {len(FLAVORS)} 种")

    lock = threading.Lock()
    fh = open(OUT, "w", encoding="utf-8")
    counter, tok, t0 = [0], [0, 0], time.time()

    def work(bi):
        fl, desc = jobs[bi]
        for attempt in range(2):
            try:
                raw, pt, ct = call_api(
                    base, key,
                    PROMPT.format(n=args.batch, flavor=fl, desc=desc))
                cases = json.loads(raw.strip().strip("`").removeprefix("json").strip())["cases"]
                with lock:
                    tok[0] += pt
                    tok[1] += ct
                    for c in cases[:args.batch]:
                        row = to_row(counter[0], c, fl)
                        if row:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            counter[0] += 1
                return
            except Exception as e:
                if attempt == 1:
                    print(f"[WARN] 批{bi}({fl}) 失败: {str(e)[:70]}", file=sys.stderr)
                time.sleep(2)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, range(n_batches)))
    fh.close()
    print(f"完成: {counter[0]} 行 → {OUT}, token {tok[0]}+{tok[1]}, "
          f"用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
