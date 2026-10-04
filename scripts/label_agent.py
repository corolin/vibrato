# -*- coding: utf-8 -*-
"""label_agent — agent(AI伴侣)侧逐回复 PAD/压力标注 (v6 piece ④-a)

v1/affect 消息范式需要 assistant 消息带状态快照(pad/pressure)。本脚本用
deepseek-flash(关思考) 对每段会话一次性标注全部 AI 回复: 标的是 **AI 自己**
在说出该回复那一刻的情绪状态(共情下随用户波动), 不是用户的状态。

输入: conversations.jsonl / convos_a.jsonl / convos_b.jsonl ({id, scenario, messages})
输出: agent_labels.jsonl {"id", "ratings":[{"idx","pad":[-1,1],"pressure"}], "usage"}
      idx = 该回复在原 messages 列表的全局下标; pad 由 LLM 的 [-4,4] 评分 /4 归一。
增量落盘断点续跑, 容错解析(夹紧/缺项重试一次), 并发 6。

用法: set -a; . /root/vibrato/.showdown_env; set +a; python3 label_agent.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import urllib.request

CONVO_FILES = ["data/conversations.jsonl", "data/convos_a.jsonl", "data/convos_b.jsonl"]
OUT = "data/agent_labels.jsonl"

PROMPT = """你是情绪标注专家。以下是一段用户与AI伴侣的对话。请对【AI的每一条回复】标注: AI 在说出这条回复那一刻自身的情绪状态。

PAD 三维(各∈[-4,4]): p愉悦(-4很负面~4很正面), a唤醒(-4很平静~4很激动), d支配(-4很弱势顺从~4很强势主导)。
pressure: AI此刻的累积情绪负荷 0~100(持续安抚疲惫/激动的用户会升高, 轻松闲聊低)。

注意:
1. 标的是 AI 自己的状态, 不是用户的状态(共情型伴侣会随用户情绪同向波动, 但强度通常弱于用户)
2. idx 用对话里方括号标注的消息全序号

对话:
{dialogue}

只输出JSON: {{"ratings":[{{"idx":0,"pad":[p,a,d],"pressure":12.0}}, ...]}} (覆盖上面每一条 [AI] 消息)"""


def call_api(base: str, key: str, prompt: str) -> tuple:
    payload = {
        "model": "deepseek/deepseek-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000, "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    usage = r.get("usage") or {}
    return (r["choices"][0]["message"]["content"].strip(),
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


def parse_ratings(raw: str, ai_idxs):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    ratings = {}
    for rt in data.get("ratings", []):
        try:
            idx = int(rt["idx"])
            pad = [max(-1.0, min(1.0, float(v) / 4.0)) for v in rt["pad"][:3]]
            pressure = max(0.0, min(100.0, float(rt.get("pressure", 0.0))))
            ratings[idx] = {"idx": idx, "pad": pad, "pressure": pressure}
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    missing = [i for i in ai_idxs if i not in ratings]
    return ratings, missing


def label_one(base, key, convo):
    msgs = convo["messages"]
    lines = []
    ai_idxs = []
    for i, m in enumerate(msgs):
        tag = "[用户]" if m["side"] == "user" else "[AI]"
        if m["side"] != "user":
            ai_idxs.append(i)
        lines.append(f"[{i}] {tag}: {str(m.get('text', '')).strip()}")
    prompt = PROMPT.format(dialogue="\n".join(lines))
    usage_acc = [0, 0]
    for attempt in range(2):
        try:
            raw, pt, ct = call_api(base, key, prompt)
            usage_acc[0] += pt
            usage_acc[1] += ct
            ratings, missing = parse_ratings(raw, ai_idxs)
            if not missing:
                return {"id": convo["id"], "ratings": [ratings[i] for i in ai_idxs],
                        "usage": {"prompt_tokens": usage_acc[0],
                                  "completion_tokens": usage_acc[1]}}
        except Exception as e:
            if attempt == 1:
                print(f"[WARN] {convo['id']} 两轮失败: {e}", file=sys.stderr)
                return None
            time.sleep(2)
    print(f"[WARN] {convo['id']} 缺 idx: {missing}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    base = (os.environ.get("AGENT_BASE_URL") or os.environ.get("OCC_BASE_URL")
            or os.environ.get("TEACHER_BASE_URL") or "https://open.cherryin.net/v1")
    key = (os.environ.get("AGENT_API_KEY") or os.environ.get("OCC_API_KEY")
           or os.environ.get("TEACHER_API_KEY"))
    if not key:
        sys.exit("缺少 API key (AGENT_API_KEY/OCC_API_KEY/TEACHER_API_KEY)")

    convos = []
    for f in CONVO_FILES:
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                convos.extend(json.loads(ln) for ln in fh if ln.strip())
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            done = {json.loads(ln)["id"] for ln in fh if ln.strip()}
    todo = [c for c in convos if c["id"] not in done]
    print(f"会话 {len(convos)} 段, 已标 {len(done)}, 待标 {len(todo)}")

    lock = threading.Lock()
    fh = open(args.out, "a", encoding="utf-8")
    t0, n_ok, n_fail = time.time(), 0, 0

    def work(c):
        nonlocal n_ok, n_fail
        r = label_one(base, key, c)
        with lock:
            if r:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
            else:
                n_fail += 1
            if (n_ok + n_fail) % 50 == 0:
                print(f"  进度 {n_ok + n_fail}/{len(todo)} ok={n_ok} "
                      f"({(time.time() - t0):.0f}s)", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"完成: ok={n_ok} fail={n_fail}, 用时 {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
