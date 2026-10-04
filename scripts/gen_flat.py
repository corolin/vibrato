# -*- coding: utf-8 -*-
"""gen_flat — 疲惫平淡形态补数据 (v6.3, 第三刀)

v6.2 后最后分歧簇: 疲惫平淡回复(还行吧/知道了/随便吧)被判 无聊——但 累≠无聊
(无聊质心 A-2.63 极低唤醒; 累是中低唤醒, 正解应落 平静/悲伤)。本批 1200 条
三风味, pad+family 双自评, 教 累→calm/sad 而非 无聊。

用法: OCC_BASE_URL=https://open.cherryin.net/v1 OCC_API_KEY=sk-.. \
      python3 gen_flat.py --n 1200
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request

from vibrato import pad_schema
from vibrato.battery import FAMILY_IDS

OUT = "data/flat_rows.jsonl"
V5_PREFIX = "TARGET(需判断): [用户]: "

FLAVORS = [
    ("tired_reply", "疲惫平淡的回应——还行吧/嗯/知道了/就这样吧/先这样, 累但不想多说"),
    ("low_energy_warm", "低电量但有暖意——还不错啦/还好还好/今天没那么糟/有点累但是开心"),
    ("flat_ack", "心不在焉的应付——哦/嗯嗯/好嘞/随便/都行, 注意力不在这"),
]

FAMILY_LINE = ("happy(喜悦) affectionate(亲昵撒娇) calm(平静温和) amused(逗趣玩闹) "
               "sad(难过) anxious(焦虑担心) angry(生气) wronged(委屈)")

PROMPT = """你是中文聊天语料作家。写 {n} 条"用户发给AI伴侣"的微信风格消息, 全部属于同一种形态: {flavor}——{desc}。

要求:
1. 每条独立成句, 4~25字, 口语化, 可带省略号/语气词, 像真的在跟AI伴侣发消息
2. 场景多样化: 加班/深夜/下课/通勤/游戏间隙/饭点, 不要重复句式
3. 注意区分: 累是"低电量的平静或低落"(唤醒中低, 不是兴奋), 无聊是"闲得发慌"(唤醒更低更空)——按真实感受自评, 不要都往无聊上靠
4. 每条给出自评: pad:[p,a,d] 各维∈[-4,4] (p愉悦 -4很负面~4很正面; a唤醒 -4很平静~4很激动; d支配 -4顺从~4强势), 以及 family 从这些里选一: {families}

只输出JSON: {{"cases":[{{"text":"…","pad":[p,a,d],"family":"calm"}}, …]}}"""


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
    fam = str(case.get("family", "")).strip().lower()
    if not text or fam not in FAMILY_IDS:
        return None
    pad = [max(-4.0, min(4.0, float(v))) for v in case["pad"][:3]]
    bins = {d: pad_schema.likert_to_bins(pad[k] + 5.0)
            for k, d in enumerate(pad_schema.DIMS)}
    return {"id": f"f3-{i:05d}", "text_input": V5_PREFIX + text,
            "echo": [0.0, 0.0, 0.0], "pad_bins": bins,
            "family": fam, "meta": {"flavor": flavor}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    base = (os.environ.get("OCC_BASE_URL") or os.environ.get("TEACHER_BASE_URL")
            or "https://open.cherryin.net/v1")
    key = os.environ.get("OCC_API_KEY") or os.environ.get("TEACHER_API_KEY")
    if not key:
        sys.exit("缺少 API key (OCC_API_KEY/TEACHER_API_KEY)")

    n_batches = (args.n + args.batch - 1) // args.batch
    jobs = [FLAVORS[i % len(FLAVORS)] for i in range(n_batches)]
    print(f"{args.n} 条 ÷ {args.batch}/批 = {n_batches} 批, 风味 {len(FLAVORS)} 种")

    lock = threading.Lock()
    fh = open(OUT, "w", encoding="utf-8")
    counter, tok, t0 = [0], [0, 0], time.time()

    def work(bi):
        fl, desc = jobs[bi]
        for attempt in range(2):
            try:
                raw, pt, ct = call_api(
                    base, key,
                    PROMPT.format(n=args.batch, flavor=fl, desc=desc, families=FAMILY_LINE))
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
