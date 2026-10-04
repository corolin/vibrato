# -*- coding: utf-8 -*-
"""合成 用户↔AI伴侣 对话(chordia 分布) — Vibrato 训练数据上游。

场景矩阵: 关系阶段 × 用户情绪 × 话题。user=被分析的人类(情绪载体), ai=chordia。
输出 conversations.jsonl: {"id","scenario","messages":[{side,text},...]}

用法(服务器): cd /root/vibrato && python3 gen_convos.py --count 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from typing import Any, Dict, List

ENDPOINT = "http://localhost:11434/api/chat"
MODEL = os.environ.get("TEACHER_MODEL", "qwen3.5:4b")

STAGES = ["初次见面", "熟悉期", "亲密期", "冷淡疏远期"]
USER_MOODS = [
    "开心分享", "兴奋激动", "平静日常", "疲惫", "焦虑求安慰", "委屈",
    "生气怼AI", "敷衍冷淡", "言不由衷逞强", "深夜emo", "撒娇亲昵",
]
TOPICS = ["日常琐事", "工作学习压力", "感情话题", "游戏", "创作", "人际冲突", "睡前闲聊"]

PROMPT = """你是一名中文对话语料作家。写一段"用户与AI聊天伴侣"的微信风格中文对话。

设定:
- 关系阶段: {stage}; 用户当前情绪: {mood}; 话题围绕: {topic}
- "user"是人类用户(情绪载体), "ai"是陪伴型AI(chordia, 温暖但不卑躬)。
- 6-10 条消息轮流, 以 user 结尾。用户消息要真实体现[{mood}]的情绪({stage}阶段该有的语气)。
- 语言像真的打字: 口语、短句、语气词、标点随意。AI回复保持陪伴感, 1-2句为主。

只输出 JSON 对象: {{"messages": [{{"side": "user"|"ai", "text": "…"}}, …]}}, 不要解释。"""


def chat(prompt: str, temperature: float) -> str:
    """双通道: 设了 TEACHER_BASE_URL 走 OpenAI 兼容 API, 否则 ollama 原生。"""
    base = os.environ.get("TEACHER_BASE_URL")
    key = os.environ.get("TEACHER_API_KEY", "")
    if base:
        url = base.rstrip("/") + "/chat/completions"
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": 1600,
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
                   "options": {"temperature": temperature, "num_predict": 900}}
        headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    if base:
        return r["choices"][0]["message"]["content"].strip()
    return r["message"]["content"].strip()


def parse_messages(raw: str) -> List[Dict[str, str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    # format:json 可能包了一层数组字符串或对象
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        data = json.loads(text[start:end + 1])
    if isinstance(data, dict):
        data = data.get("messages", data.get("conversation", []))
    out = []
    for m in data:
        side, t = m.get("side"), str(m.get("text", "")).strip()
        if side in ("user", "ai") and t:
            out.append({"side": side, "text": t[:200]})
    return out


def validate(msgs: List[Dict[str, str]]) -> bool:
    if not (6 <= len(msgs) <= 11):
        return False
    users = [m for m in msgs if m["side"] == "user"]
    return len(users) >= 3 and msgs[-1]["side"] == "user"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", default="data/conversations.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 增量写盘 + 断点续跑: 已有行直接计入, 只补差额(死机最多丢正在生成的一段)
    done_ids: set = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
    existing = len(done_ids)
    remaining = max(0, args.count - existing)
    print(f"目标 {args.count} 段, 已有 {existing} 段, 本次生成 {remaining} 段")

    rng = random.Random(args.seed + existing)   # 续跑换种子, 避免场景重复
    results = 0
    fails = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        i = existing
        produced = 0
        while produced < remaining and fails < remaining * 2:
            stage, mood, topic = rng.choice(STAGES), rng.choice(USER_MOODS), rng.choice(TOPICS)
            prompt = PROMPT.format(stage=stage, mood=mood, topic=topic)
            ok = False
            for attempt in range(3):
                try:
                    msgs = parse_messages(chat(prompt, 1.0))
                    if validate(msgs):
                        cid = f"v{i:04d}"
                        fout.write(json.dumps({"id": cid,
                                               "scenario": {"stage": stage, "mood": mood, "topic": topic},
                                               "messages": msgs}, ensure_ascii=False) + "\n")
                        fout.flush()
                        i += 1
                        produced += 1
                        n_user = sum(1 for m in msgs if m["side"] == "user")
                        print(f"[{existing + produced}/{args.count}] {cid} {stage}/{mood}/{topic} "
                              f"-> {len(msgs)}条({n_user}条user) ✓", flush=True)
                        ok = True
                        break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f"[{i}] 失败: {str(exc)[:80]}", file=sys.stderr, flush=True)
                    else:
                        time.sleep(2)
            if not ok:
                fails += 1
    total_user = sum(sum(1 for m in json.loads(l)["messages"] if m["side"] == "user")
                     for l in open(args.out, encoding="utf-8") if l.strip())
    print(f"\n完成: 文件现有 {existing + produced} 段对话(本次新增 {produced}, 失败 {fails}) "
          f"共 {total_user} 条 user 消息 -> {args.out}")


if __name__ == "__main__":
    main()
