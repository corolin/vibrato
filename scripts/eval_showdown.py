# -*- coding: utf-8 -*-
"""终极对决: Vibrato(4B教出的SystemOne) vs Qwen-7B(生产答卷LLM) — 同题同卷。

裁判: 双教师共识(local-4B ∩ Qwen-7B, 两边都签的判项才算GT) + 分歧样本留人工。
赛事: 速度 / PAD MAE(vs共识) / 情绪词解码 top1&top2 一致率 / noul 一致率 / 分歧样本
用法(服务器):
  set -a; source /root/vibrato/.showdown_env; set +a
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-ort \
  python3 eval_showdown.py --ckpt ckpt_v3/vibrato.pt --convos test_convos.jsonl
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
from vibrato.decode import decode_top2  # noqa: E402
from vibrato.vib_state import build_context, echo_from_prev  # noqa: E402

CHALLENGER_MODEL = os.environ.get("CHALLENGER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CHALLENGER_BASE = os.environ.get("CHALLENGER_BASE_URL", "https://api.siliconflow.cn/v1")
REFEREE_LOCAL = "http://localhost:11434/api/chat"     # local 4B(学生师傅, 共识裁判之一)
REF_MODEL = os.environ.get("TEACHER_MODEL", "qwen3.5:4b-q8_0")


def api_chat(prompt: str, temperature: float = 0.3) -> str:
    payload = {"model": CHALLENGER_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": 2400,
               "response_format": {"type": "json_object"}}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ.get('TEACHER_API_KEY', '')}"}
    req = urllib.request.Request(CHALLENGER_BASE.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"]["content"].strip()


def local_chat(prompt: str) -> str:
    payload = {"model": REF_MODEL, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": False, "format": "json",
               "options": {"temperature": 0.3, "num_predict": 700}}
    req = urllib.request.Request(REFEREE_LOCAL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=240).read())["message"]["content"].strip()


def judge_to_pad(j: dict) -> list:
    bins = pad_schema.items_to_bins(j["items"])
    pad = pad_schema.bins_to_pad(bins)
    return [pad[d] for d in pad_schema.DIMS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--convos", required=True, help="测试对话 jsonl(不得在训练集)")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="showdown_report.json")
    args = ap.parse_args()

    import torch
    from net import VibratoNet

    ck = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]), max_len=ck["max_len"]).cuda().eval()
    net.load_state_dict(ck["model"])
    vocab = ck["vocab"]

    convos = [json.loads(l) for l in open(args.convos, encoding="utf-8") if l.strip()]
    rounds = []
    t_vib_all = t_7b_all = 0.0
    prev_pad = None
    n = 0
    for convo in convos:
        for idx, m in enumerate(convo["messages"]):
            if m["side"] != "user" or n >= args.limit:
                continue
            n += 1
            text_input = build_context(convo["messages"], idx)
            echo = echo_from_prev(prev_pad)
            prompt = battery.build_teacher_prompt(text_input, echo)

            # 选手A: Vibrato (本地)
            t0 = time.perf_counter()
            ids = torch.tensor([[vocab.get(ch, 1) for ch in text_input][-ck["max_len"] + 1:]]).cuda()
            e = torch.tensor([echo]).cuda()
            with torch.no_grad():
                out = net(ids, e)
            vib_pad = out["pad_norm"][0].tolist()
            vib_word, vib_word2, _ = decode_top2(vib_pad)
            vib_noul = {q: bool(out["noul_probs"][0, i, 1] > 0.5)
                        for i, q in enumerate(battery.NOUL_IDS)}
            t_vib_all += time.perf_counter() - t0

            # 选手B: Qwen-7B 答卷(生产模式) + 裁判: local-4B
            t0 = time.perf_counter()
            j7 = battery.parse_teacher_json(api_chat(prompt))
            t_7b_all += time.perf_counter() - t0
            j4 = battery.parse_teacher_json(local_chat(prompt))
            pad7, pad4 = judge_to_pad(j7), judge_to_pad(j4)

            w7, w7_2, _ = decode_top2(pad7)
            tv = max(sum(abs(a - b) for a, b in zip(pad7, pad4)) / 3, 0.0)
            consensus = tv < 0.30  # 两教师PAD均值差<0.3 才算共识GT

            rounds.append({
                "text": text_input[-60:], "echo": echo,
                "vib": {"pad": [round(v, 3) for v in vib_pad], "word": vib_word, "word2": vib_word2,
                        "noul": vib_noul, "ms": None},
                "q7": {"pad": [round(v, 3) for v in pad7], "word": w7, "word2": w7_2,
                       "noul": j7["noul"]},
                "ref_pad4": [round(v, 3) for v in pad4], "consensus": consensus,
                "vib_mae": (sum(abs(a - b) for a, b in zip(vib_pad, pad4)) / 3) if consensus else None,
                "q7_mae": (sum(abs(a - b) for a, b in zip(pad7, pad4)) / 3) if consensus else None,
            })
            prev_pad = pad4  # echo链用裁判PAD, 两选手同输入

    cons = [r for r in rounds if r["consensus"]]
    mae_v = [r["vib_mae"] for r in cons if r["vib_mae"] is not None]
    mae_q = [r["q7_mae"] for r in cons if r["q7_mae"] is not None]
    top1 = sum(r["vib"]["word"] == r["q7"]["word"] for r in rounds)
    top2 = sum(r["q7"]["word"] in (r["vib"]["word"], r["vib"]["word2"]) for r in rounds)
    noul_agree = sum(
        sum(r["vib"]["noul"][q] == r["q7"]["noul"][q] for q in battery.NOUL_IDS) / len(battery.NOUL_IDS)
        for r in rounds) / max(1, len(rounds))

    print(f"\n═══ 战报 (n={len(rounds)}, 共识判项 {len(cons)}) ═══")
    print(f"速度:   Vibrato {t_vib_all*1000/len(rounds):.2f} ms/条  vs  Qwen-7B {t_7b_all/len(rounds):.2f} s/条"
          f"  → {t_7b_all/max(t_vib_all,1e-9):.0f}倍")
    if mae_v and mae_q:
        print(f"PAD MAE(vs共识裁判4B): Vibrato {sum(mae_v)/len(mae_v):.4f}  vs  "
              f"Qwen-7B {sum(mae_q)/len(mae_q):.4f}  (注: 裁判=学生师傅, 此项偏向Vibrato, 仅参考)")
    print(f"情绪词: 双方 top1 一致 {top1}/{len(rounds)} = {top1/len(rounds):.2f} | "
          f"7B主词∈Vibrato top2 {top2}/{len(rounds)} = {top2/len(rounds):.2f}")
    print(f"noul:   五旗逐项一致率均值 {noul_agree:.2f}")
    print("\n分歧样本(前6, 人工裁决用):")
    for r in [r for r in rounds if r["vib"]["word"] != r["q7"]["word"]][:6]:
        print(f"  「…{r['text'][-40:]}」 vib={r['vib']['word']}/{r['vib']['word2']} "
              f"q7={r['q7']['word']}/{r['q7']['word2']}")
    json.dump({"rounds": rounds, "n": len(rounds)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"明细已存 {args.out}")


if __name__ == "__main__":
    main()
