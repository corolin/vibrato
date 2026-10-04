# -*- coding: utf-8 -*-
"""OCC-22 全谱用例生成器 — 2000 条用户↔AI伴侣消息, deepseek-flash 量产。

22 型各 ~91 条, 每条带 pad:[p,a,d]∈[-4,4](生成时同窗自评, 单次通过零标注成本)。
thinking 关闭省 token。增量落盘断点续跑。输出 occ_cases.jsonl: {occ, zh, text, pad}

用法: OCC_BASE_URL=https://open.cherryin.net/v1 OCC_API_KEY=sk-.. \
      python3 gen_occ.py --per-type 91
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

OCC_22 = [
    ("happy_for", "为你高兴", "为他人遇到的好事由衷开心(朋友升职/恋人成功)"),
    ("gloating", "幸灾乐祸", "对讨厌之人的倒霉暗自得意嘴上还俏皮"),
    ("resentment", "怨恨", "别人得了好处而自己没有, 不平与酸"),
    ("pity", "心疼怜悯", "为别人的不幸难过, 想安慰"),
    ("hope", "期待希望", "盼望好事发生(周末约会/结果公布前)"),
    ("fear", "担忧害怕", "怕坏事发生(体检报告/面试结果)"),
    ("satisfaction", "满意", "结果如预期的好, 踏实"),
    ("fears_confirmed", "应验", "担心的事真的发生了, 心一沉"),
    ("relief", "释然松口气", "担心的事没发生, 长出一口气"),
    ("disappointment", "失望", "盼望的事落空"),
    ("joy", "喜悦", "单纯的高兴, 好事正在发生"),
    ("distress", "苦恼烦闷", "单纯的难受憋闷说不清"),
    ("pride", "骄傲自豪", "自己干得漂亮想被夸"),
    ("admiration", "钦佩崇拜", "别人真厉害(偶像/大佬/身边的TA)"),
    ("shame", "羞愧", "自己做了丢人的事抬不起头"),
    ("reproach", "责备不满", "别人做得不对, 怄气或指出"),
    ("gratification", "欣慰满足", "努力有了回报, 值了"),
    ("remorse", "懊悔", "自己做错了, 又悔又难受"),
    ("love", "喜爱亲近", "喜欢对方, 撒娇粘人表达爱"),
    ("hate", "厌恶", "讨厌某人某事, 嫌弃"),
    ("liking", "好感喜欢", "轻度喜欢(新游戏/小猫/一首歌)"),
    ("disliking", "轻度反感", "轻度讨厌(小事膈应)"),
]

PROMPT = """你是中文聊天语料作家。写 {n} 条"用户发给AI伴侣"的微信风格消息, 全部表达同一种情绪: {zh}({occ})——{definition}。

要求:
1. 每条独立成句, 8~40字, 口语化, 可带语气词/emoji, 像真的在跟AI伴侣聊天
2. 场景多样化: 工作/学习/游戏/日常/感情/追剧/天气/社交, 不要重复句式
3. 话题提及的对象可以是AI本身/第三方/自己/事物
4. 每条同时给出该消息的PAD坐标 pad:[p,a,d], 各维∈[-4,4](p愉悦正负/a唤醒强弱/d支配高低)

只输出JSON: {{"cases":[{{"text":"…","pad":[p,a,d]}}, …]}}"""


def call_api(base: str, key: str, prompt: str) -> str:
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
    return r["choices"][0]["message"]["content"].strip()


def parse(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)["cases"]


def gen_batch(base, key, occ, zh, definition, n, out_lock, counts):
    prompt = PROMPT.format(n=n, zh=zh, occ=occ, definition=definition)
    for attempt in range(3):
        try:
            cases = parse(call_api(base, key, prompt))
            ok = []
            for c in cases:
                t = str(c.get("text", "")).strip()
                pad = c.get("pad")
                if t and len(t) >= 4 and isinstance(pad, list) and len(pad) == 3:
                    try:
                        pad = [max(-4.0, min(4.0, float(x))) for x in pad]
                        ok.append({"occ": occ, "zh": zh, "text": t[:200], "pad": pad})
                    except (TypeError, ValueError):
                        continue
            with out_lock:
                with open(OUT, "a", encoding="utf-8") as f:
                    for c in ok:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                counts[occ] = counts.get(occ, 0) + len(ok)
                done = sum(min(v, PER) for v in counts.values())
                print(f"[{occ}] +{len(ok)} (累计{counts[occ]}/{PER}) 总进度{done}/{22*PER}",
                      flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"[{occ}] 失败: {str(exc)[:70]}", file=sys.stderr, flush=True)
            else:
                time.sleep(3)


def main() -> None:
    global OUT, PER
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=91)
    ap.add_argument("--out", default="data/occ_cases.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    OUT, PER = args.out, args.per_type

    base = os.environ.get("OCC_BASE_URL", "https://open.cherryin.net/v1")
    key = os.environ.get("OCC_API_KEY", "")
    if not key:
        sys.exit("缺 OCC_API_KEY")

    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            if line.strip():
                occ = json.loads(line)["occ"]
                counts[occ] = counts.get(occ, 0) + 1
    print(f"续跑现状: {sum(counts.values())} 条 | 目标 {22*PER}", flush=True)

    import threading
    out_lock = threading.Lock()
    jobs = []
    for occ, zh, definition in OCC_22:
        need = PER - counts.get(occ, 0)
        for _ in range(0, need, args.batch_size):
            n = min(args.batch_size, need - (len([j for j in jobs])) % max(need, 1))
            jobs.append((occ, zh, definition, min(args.batch_size, PER)))
    # 简化: 每型按缺口整批生成(每批batch_size条), 多退少补由续跑消化
    jobs = []
    for occ, zh, definition in OCC_22:
        need = max(0, PER - counts.get(occ, 0))
        for _ in range((need + args.batch_size - 1) // args.batch_size):
            jobs.append((occ, zh, definition, args.batch_size))

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(gen_batch, base, key, occ, zh, d, n, out_lock, counts)
                for occ, zh, d, n in jobs]
        for f in futs:
            f.result()
    total = sum(1 for _ in open(OUT, encoding="utf-8"))
    print(f"完成: 文件现有 {total} 条 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
