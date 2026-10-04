# -*- coding: utf-8 -*-
"""gen_sajiao2 — 形态缺口定向补数据 (v6.2, 顽固案追杀)

v6.1 后仍极性反转的两形态: 叹气式正向开场("啊...终于等到你啦")与随口寒暄
("在忙啥呢")。本批 1500 条覆盖三风味, **family 也由生成器自评**(8族选一,
寒暄不该硬塞 affectionate), pad 自评同 gen_sajiao。

用法: OCC_BASE_URL=https://open.cherryin.net/v1 OCC_API_KEY=sk-.. \
      python3 gen_sajiao2.py --n 1500
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

OUT = "data/sajiao2_rows.jsonl"
V5_PREFIX = "TARGET(需判断): [用户]: "

FLAVORS = [
    ("sigh_open", "叹气式正向开场——以 啊.../诶.../哈~/呜哇 等语气词开头, 后接期待/重逢/松了口气的正向内容"),
    ("checkin", "随口寒暄关心——在忙啥呢/在干嘛呀/吃饭了没/睡了么/好久没聊了, 温和关注但不粘人"),
    ("relief", "松了口气——可算下班了/还好还好/吓死我了结果没事/虚惊一场"),
]

FAMILY_LINE = ("happy(喜悦) affectionate(亲昵撒娇) calm(平静温和) amused(逗趣玩闹) "
               "sad(难过) anxious(焦虑担心) angry(生气) wronged(委屈)")

PROMPT = """你是中文聊天语料作家。写 {n} 条"用户发给AI伴侣"的微信风格消息, 全部属于同一种形态: {flavor}——{desc}。

要求:
1. 每条独立成句, 6~35字, 口语化, 可带语气词/省略号/emoji, 像真的在跟AI伴侣发消息
2. 场景多样化: 工作/学习/游戏/日常/感情/追剧/天气/深夜/清晨, 不要重复句式
3. 情绪整体是**中性偏暖**的: 轻度愉悦、放松、关注, 不是激烈情绪
4. 每条给出自评: pad:[p,a,d] 各维∈[-4,4] (p愉悦 -4很负面~4很正面; a唤醒 -4很平静~4很激动; d支配 -4顺从~4强势), 以及 family 从这些里选一: {families}

只输出JSON: {{"cases":[{{"text":"…","pad":[p,a,d],"family":"happy"}}, …]}}"""


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
    return {"id": f"s2-{i:05d}", "text_input": V5_PREFIX + text,
            "echo": [0.0, 0.0, 0.0], "pad_bins": bins,
            "family": fam, "meta": {"flavor": flavor}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
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
