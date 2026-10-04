# -*- coding: utf-8 -*-
"""答卷收集器 — 各路选手把同一张考卷答完存档, 判卷离线进行(用户架构)。

两模式:
  ref 模式(本地4B裁判, 免API): python3 answer_sheet.py ref
  挑战者模式: python3 answer_sheet.py challenge --model Qwen/Qwen2.5-7B-Instruct --tag q7
    (--base sf|cherry, 思考开关自动适配: 400则去掉重试)

考题: test_convos.jsonl 的 user 消息(v5.1 复赛同款 40 题); echo 链统一用 ref 的 PAD
(所有选手同题同上下文, 答卷严格可比)。token 用量逐行记录+汇总。

输出: answers_<tag>.jsonl {idx, text_tail, items, family, noul, pad, usage, latency_s}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import battery  # noqa: E402
from vibrato import pad_schema  # noqa: E402
from vibrato.vib_state import build_context, echo_from_prev  # noqa: E402

SF_BASE = "https://api.siliconflow.cn/v1"
CHERRY_BASE = "https://open.cherryin.net/v1"
TEST_CONVOS = "data/test_convos.jsonl"
REF_FILE = "data/answers_ref4b.jsonl"
LIMIT = 40
NO_THINK = {"thinking": {"type": "disabled"}, "enable_thinking": False}


def questions():
    convos = [json.loads(l) for l in open(TEST_CONVOS, encoding="utf-8") if l.strip()]
    out = []
    for convo in convos:
        for idx, m in enumerate(convo["messages"]):
            if m["side"] == "user" and len(out) < LIMIT:
                out.append((convo["id"], idx, build_context(convo["messages"], idx)))
    return out


def echo_chain_from_ref():
    """ref 的 PAD 序列 → 每题 echo(与 eval_showdown 完全一致)。"""
    echo, prev = [], None
    for line in open(REF_FILE, encoding="utf-8"):
        if line.strip():
            prev = json.loads(line)["pad"]
            echo.append(echo_from_prev(prev))
    if len(echo) != len(questions()):
        sys.exit(f"ref 答卷数 {len(echo)} != 题数 {len(questions())}, 先跑 ref 模式")
    return echo


def parse_judge(raw: str) -> dict:
    """容错版判卷解析: items缺省5.0/钳位, family透传不校验(混元会自创族名), noul缺省False。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    d = json.loads(text)  # 截断的JSON仍会炸 → 由collect层重试兜底
    items = {f"I{i}": max(1.0, min(9.0, float(d.get(f"I{i}", 5.0))))
             for i in range(1, 19)}
    noul_raw = d.get("noul", {})
    noul = {q: bool(noul_raw.get(q, False)) for q in battery.NOUL_IDS}
    family = str(d.get("family", "unknown"))
    bins = pad_schema.items_to_bins(items)
    pad = pad_schema.bins_to_pad(bins)
    return {"items": items, "family": family, "noul": noul,
            "pad": [pad[dim] for dim in pad_schema.DIMS]}


def call_ref(prompt: str):
    payload = {"model": os.environ.get("TEACHER_MODEL", "qwen3.5:4b-q8_0"),
               "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": False, "format": "json",
               "options": {"temperature": 0.3, "num_predict": 700}}
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    return r["message"]["content"].strip(), {}, time.perf_counter() - t0


def call_api(base: str, key: str, model: str, prompt: str, no_think: bool):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3, "max_tokens": 2400,
               "response_format": {"type": "json_object"}}
    if no_think:
        payload.update(NO_THINK)
    for attempt in range(2):
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        t0 = time.perf_counter()
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return (r["choices"][0]["message"]["content"].strip(), r.get("usage", {}),
                    time.perf_counter() - t0)
        except urllib.error.HTTPError as e:
            if e.code in (400, 500) and no_think:   # 500: GPT通道对thinking字段直接炸(gateway适配器崩)
                payload.pop("thinking", None); payload.pop("enable_thinking", None)
                no_think = False
                continue
            raise
    raise RuntimeError("unreachable")


def collect(tag: str, call_fn) -> None:
    qs = questions()
    out_file = f"data/answers_{tag}.jsonl"
    done = 0
    if os.path.exists(out_file):
        done = sum(1 for l in open(out_file, encoding="utf-8") if l.strip())
    if tag == "ref4b":
        # ref 链: echo[0]=零向量, echo[i]=ref 第 i-1 答的 PAD; 断点续跑从盘上回放
        rows = [json.loads(l) for l in open(out_file, encoding="utf-8") if l.strip()] \
            if os.path.exists(out_file) else []
        echo = [echo_from_prev(None)] + [echo_from_prev(r["pad"]) for r in rows]
    else:
        echo = echo_chain_from_ref()
    tot_p = tot_c = n_rows = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(done, len(qs)):
            convo_id, idx, text = qs[i]
            prompt = battery.build_teacher_prompt(text, echo[i] if i < len(echo) else echo_from_prev(None))
            for _try in range(2):
                try:
                    raw, usage, lat = call_fn(prompt)
                    parsed = parse_judge(raw)
                    break
                except Exception:
                    if _try:
                        raise
                    time.sleep(2)
            row = {"idx": i, "convo": convo_id, "text_tail": text[-60:],
                   "latency_s": round(lat, 2), "usage": {
                       "prompt": usage.get("prompt_tokens"),
                       "completion": usage.get("completion_tokens")}}
            row.update(parsed)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n_rows += 1
            tot_p += usage.get("prompt_tokens") or 0
            tot_c += usage.get("completion_tokens") or 0
            echo.append(echo_from_prev(row["pad"]))  # ref 链延伸(挑战者模式此行无副作用)
            print(f"[{i+1}/{len(qs)}] {tag} {row['latency_s']}s "
                  f"(+{usage.get('completion_tokens', 0)}tok)", flush=True)
    print(f"完成 answers_{tag}.jsonl: 新增 {n_rows} 行 | 本轮token: "
          f"prompt≈{tot_p} completion≈{tot_c}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "challenge"])
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--base", default="sf", choices=["sf", "cherry"])
    args = ap.parse_args()
    if args.mode == "ref":
        collect("ref4b", call_ref)
        return
    assert args.model and args.tag
    base = SF_BASE if args.base == "sf" else CHERRY_BASE
    key = (os.environ.get("TEACHER_API_KEY") if args.base == "sf"
           else os.environ.get("OCC_API_KEY", "")) or \
        open("/root/vibrato/.showdown_env").read().strip().split("=", 1)[1]
    collect(args.tag, lambda p: call_api(base, key, args.model, p, True))


if __name__ == "__main__":
    main()
