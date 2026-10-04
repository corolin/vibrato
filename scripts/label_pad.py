# -*- coding: utf-8 -*-
"""Vibrato 数据标注 — 全题库双采样 + echo 链 + 自一致性过滤。

流程: conversations.jsonl 的每段对话按序遍历 user 消息:
  1. vib_state.build_context 组输入文本(训练/teacher/推理同源)
  2. echo = 上一条 user 消息的归一化 PAD(标签链; 首条为零向量)
  3. teacher ×2 采样(temperature=0.4) → battery.parse_teacher_json
  4. 一致性: PAD 三维逐维 TV + family 是否一致 + noul 不一致数
  5. 合并: pad_bins=两样本分布均值(归一), family/noul 一致取之/不一致取样本1并标记

输出 vibrato_dataset.jsonl:
  {"id","text_input","echo","pad_bins","family","noul","low_consistency","meta"}

用法(服务器): cd /root/vibrato && python3 label_pad.py --in conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibrato import battery  # noqa: E402
from vibrato import pad_schema  # noqa: E402
from vibrato.vib_state import build_context, echo_from_prev  # noqa: E402

ENDPOINT = os.environ.get("TEACHER_ENDPOINT", "http://localhost:11434/api/chat")
MODEL = os.environ.get("TEACHER_MODEL", "qwen3.5:4b")
SAMPLES = 2
TV_FLAG = 0.50          # 单维 TV 超过参与标记(冒烟实测 teacher 自然方差下 0.30 过严)
NOUL_DIS_FLAG = 3       # noul 不一致数达到才标记(族边界模糊, 单族分歧不标)


def chat(prompt: str) -> str:
    """双通道: 设了 TEACHER_BASE_URL 走 OpenAI 兼容 API(免费8B等), 否则 ollama 原生。"""
    import urllib.error
    base = os.environ.get("TEACHER_BASE_URL")
    key = os.environ.get("TEACHER_API_KEY", "")
    if base:
        url = base.rstrip("/") + "/chat/completions"
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.4, "max_tokens": 900,
                   "response_format": {"type": "json_object"}}
        if os.environ.get("TEACHER_NO_THINK") == "1":
            payload["enable_thinking"] = False
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
    else:
        url = ENDPOINT
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "think": False, "format": "json",
                   "options": {"temperature": 0.4, "num_predict": 700}}
        headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    if base:
        return r["choices"][0]["message"]["content"].strip()
    return r["message"]["content"].strip()


def tv(p: List[float], q: List[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def sample_once(text_input: str, echo: List[float]) -> Dict[str, Any]:
    return battery.parse_teacher_json(
        chat(battery.build_teacher_prompt(text_input, echo)))


def label_conversation(convo: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = convo["messages"]
    rows: List[Dict[str, Any]] = []
    prev_pad: Optional[List[float]] = None
    for idx, m in enumerate(msgs):
        if m["side"] != "user":
            continue
        text_input = build_context(msgs, idx)
        echo = echo_from_prev(prev_pad)
        samples: List[Dict[str, Any]] = []
        for attempt in range(4):
            try:
                samples.append(sample_once(text_input, echo))
                if len(samples) >= SAMPLES:
                    break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    print(f"  [skip {convo['id']}#{idx}] {str(exc)[:70]}", file=sys.stderr)
                time.sleep(2)
        if len(samples) < SAMPLES:
            continue

        # PAD: 两样本 18项各自转bin后取分布均值
        b1 = pad_schema.items_to_bins(samples[0]["items"])
        b2 = pad_schema.items_to_bins(samples[1]["items"])
        pad_bins, max_tv = {}, 0.0
        for dim in pad_schema.DIMS:
            mean = [(a + b_) / 2 for a, b_ in zip(b1[dim], b2[dim])]
            s = sum(mean)
            pad_bins[dim] = [round(v / s, 5) for v in mean]
            max_tv = max(max_tv, tv(b1[dim], b2[dim]))
        pad = pad_schema.bins_to_pad(pad_bins)
        prev_pad = [pad[d] for d in pad_schema.DIMS]

        fam_agree = samples[0]["family"] == samples[1]["family"]
        # 族分歧时无更优 tiebreak, 取样本1; 原始分歧已存 meta 供训练侧利用
        family = samples[0]["family"]
        noul_dis = sum(1 for q in battery.NOUL_IDS
                       if samples[0]["noul"][q] != samples[1]["noul"][q])
        low = (max_tv > TV_FLAG) or (noul_dis >= NOUL_DIS_FLAG)
        noul = {q: (samples[0]["noul"][q] and samples[1]["noul"][q])
                if samples[0]["noul"][q] != samples[1]["noul"][q]
                else samples[0]["noul"][q] for q in battery.NOUL_IDS}

        gold = {"pad_bins": pad_bins, "echo": echo, "family": family, "noul": noul}
        problems = battery.validate_battery_gold(gold)
        if problems:
            print(f"  [skip {convo['id']}#{idx}] gold非法: {problems}", file=sys.stderr)
            continue
        rows.append({"id": f"{convo['id']}-u{idx:02d}",
                     "text_input": text_input,
                     "echo": echo,
                     **gold,
                     "low_consistency": low,
                     "meta": {"scenario": convo.get("scenario", {}),
                              "tv": round(max_tv, 3), "fam_agree": fam_agree,
                              "noul_disagree": noul_dis}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/conversations.jsonl")
    ap.add_argument("--out", default="vibrato_dataset.jsonl")
    args = ap.parse_args()

    convos = [json.loads(l) for l in open(args.in_path, encoding="utf-8") if l.strip()]
    # 增量写盘 + 断点续跑: 已标注的 id 跳过(死机只丢正在标注的一条)
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
    print(f"题库指纹 {battery.battery_digest()[:16]} | {len(convos)} 段对话 | "
          f"已完成 {len(done_ids)} 行(续跑跳过)", flush=True)
    t0 = time.time()
    n_new = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for ci, convo in enumerate(convos, 1):
            rows = label_conversation(convo)
            for r in rows:
                if r["id"] in done_ids:
                    continue
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n_new += 1
            print(f"[{ci}/{len(convos)}] {convo['id']}: +{len(rows)} 行 "
                  f"(新增累计 {n_new}, {time.time()-t0:.0f}s)", flush=True)

    all_rows = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    n_low = sum(1 for r in all_rows if r["low_consistency"])
    fam_dist: Dict[str, int] = {}
    noul_true: Dict[str, int] = {q: 0 for q in battery.NOUL_IDS}
    for r in all_rows:
        fam_dist[r["family"]] = fam_dist.get(r["family"], 0) + 1
        for q in battery.NOUL_IDS:
            noul_true[q] += bool(r["noul"][q])
    print(f"\n完成: 文件现有 {len(all_rows)} 行(本次新增 {n_new}) -> {args.out} "
          f"(低一致性 {n_low} 行 {n_low/max(1,len(all_rows))*100:.0f}%)")
    print("情绪族分布:", dict(sorted(fam_dist.items(), key=lambda kv: -kv[1])))
    print("noul=true 比例:", {q: f"{n}/{len(all_rows)}" for q, n in noul_true.items()})


if __name__ == "__main__":
    main()
