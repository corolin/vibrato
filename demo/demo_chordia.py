# -*- coding: utf-8 -*-
"""demo_chordia — 核心演示: 颤音×弦音 v2 联动 vs LLM 自演绎。

本地联动模式(默认, 零API): 三幕 20 题 × 2 人格连续对话——
  颤音臂: Vibrato 逐轮实测用户 PAD(v1/affect 消息范式, 上一轮输出写回为状态快照)
          → 弦音引擎(chordia-v2) 以实测 PAD 驱动 agent PAD 动力学
  金标准臂: 手工预分析 user_pad(剧本自带) 驱动同一引擎
  对照: 两臂的用户 PAD 相关性 + agent 轨迹发散度 → "颤音可替代手工预分析"

LLM 自演绎对照(--llm, 需API): 同一人格只给系统提示词, LLM 自己演绎全程并
自报每轮 PAD —— 与联动臂的 agent 轨迹对照, 看"自我想象的动力学"与
"被测量的动力学"差多远。

用法: cd /root/vibrato && PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge \
      CUDA_VISIBLE_DEVICES=0 python3 demo/demo_chordia.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # vibrato 根
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # demo/

import numpy as np
import torch

import vib_messages
from battery import FAMILY_IDS
from demo_scenarios import PERSONALITIES, QUESTIONS
from eval_bench import make_batch
from feat_bge import BgeFeaturizer
from net import VibratoNet

CKPT = "ckpt_v6o/vibrato.pt"
ENGINE_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "checkpoints", "chordia_v0.0.1-alpha.pth")


def clamp3(v):
    return [max(-1.0, min(1.0, x)) for x in v]


@torch.no_grad()
def vib_read(net, ck, fe, messages):
    """v1/affect 冷读: 消息列表 → (pad_norm[3], family 名)。"""
    ids, sp, sv, sm, ft, Ls = make_batch([messages], ck, fe)
    echo = torch.tensor([[0.0, 0.0, 0.0]] if len(messages) == 1
                        else [next((m["pad"] for m in reversed(messages[:-1])
                                    if m["role"] == "user" and "pad" in m), [0.0] * 3)])
    out = net(ids.cuda(), sp.cuda(), sv.cuda(), sm.cuda(), echo.cuda(),
              torch.from_numpy(ft).cuda(), lengths=Ls.cuda())
    return out["pad_norm"][0].tolist(), FAMILY_IDS[out["family_probs"][0].argmax().item()]


def run_linked(net, ck, fe, engine):
    print("══ 联动模式: 颤音实测 → 弦音 v2 动力学 ══")
    summary = {}
    for pid, p in PERSONALITIES.items():
        agent_v = list(p["base_pad"])          # 颤音臂 agent PAD
        agent_g = list(p["base_pad"])          # 金标准臂 agent PAD
        msgs: list = []
        rows = []
        pad_v_hist, pad_g_hist = [], []
        for qid, text, ctx, gold in QUESTIONS:
            msgs.append({"role": "user", "message": text})
            pad_v, fam = vib_read(net, ck, fe, msgs)
            msgs[-1]["pad"] = pad_v            # write_back: 本轮输出=下轮状态快照
            pad_v_hist.append(pad_v)
            pad_g_hist.append(gold)
            d_v = engine.predict(user_pad=pad_v, vitality=p["vitality"],
                                 current_pad=agent_v,
                                 personality_scale=p["personality_scale"])["delta_pad"]
            agent_v = clamp3([a + d for a, d in zip(agent_v, d_v)])
            d_g = engine.predict(user_pad=gold, vitality=p["vitality"],
                                 current_pad=agent_g,
                                 personality_scale=p["personality_scale"])["delta_pad"]
            agent_g = clamp3([a + d for a, d in zip(agent_g, d_g)])
            rows.append((qid, fam, pad_v, gold, agent_v, agent_g))
        # 摘要
        pv = np.array(pad_v_hist)
        pg = np.array(pad_g_hist)
        corr = [float(np.corrcoef(pv[:, i], pg[:, i])[0, 1]) for i in range(3)]
        mae = [float(np.abs(pv[:, i] - pg[:, i]).mean()) for i in range(3)]
        traj_div = float(np.abs(np.array([r[4] for r in rows]) -
                                np.array([r[5] for r in rows])).max())
        summary[pid] = dict(corr=corr, mae=mae, traj_div=traj_div)
        print(f"\n── {p['name']} ({pid}) base_pad={p['base_pad']} "
              f"scale={p['personality_scale']} vitality={p['vitality']}")
        print(f"{'#':>2} {'颤音族':12s} {'颤音实测PAD':>22s} {'手工金标PAD':>22s} "
              f"{'agentPAD(颤音驱动)':>24s}")
        for qid, fam, pad_v, gold, av, ag in rows:
            fmt = lambda t: "[" + ",".join(f"{x:+.2f}" for x in t) + "]"
            print(f"{qid:>2} {fam:12s} {fmt(pad_v):>24s} {fmt(gold):>24s} {fmt(av):>26s}")
        print(f"   用户PAD 实测vs金标: r(P/A/D)={corr[0]:.2f}/{corr[1]:.2f}/{corr[2]:.2f} "
              f"MAE={mae[0]:.2f}/{mae[1]:.2f}/{mae[2]:.2f} | "
              f"agent轨迹最大发散={traj_div:.2f}")
    return summary


SYSTEM_PROMPT = """你是一个女性AI助手，{description}。

你的人格特征：
- 愉悦度基准: {pleasure:.2f} (-1到+1，越正越快乐)
- 唤醒度基准: {arousal:.2f} (-1到+1，越正越活跃)
- 支配度基准: {dominance:.2f} (-1到+1，越正越自信)

请始终保持这个人格特征与用户对话。每轮回复后，另起一行以
[SELF_PAD: P,A,D] 报告你此刻自己的真实情绪状态(三个-1到1的数)。"""


def run_llm_selfplay(net, ck, fe, engine, model, base, key):
    """LLM 自演绎对照: 无人格引擎, 只靠系统提示词自我想象。"""
    import json as _json
    import re
    import urllib.request
    print(f"\n══ LLM 自演绎对照 ({model}) ══")
    for pid, p in PERSONALITIES.items():
        agent_s = list(p["base_pad"])          # 自报轨迹
        msgs_api = [{"role": "system", "content": SYSTEM_PROMPT.format(
            description=p["description"], pleasure=p["base_pad"][0],
            arousal=p["base_pad"][1], dominance=p["base_pad"][2])}]
        msgs_vib: list = []
        for qid, text, ctx, gold in QUESTIONS:
            msgs_api.append({"role": "user", "content": text})
            payload = {"model": model, "messages": msgs_api, "max_tokens": 500,
                       "temperature": 0.8, "thinking": {"type": "disabled"}}
            req = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=_json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            r = _json.loads(urllib.request.urlopen(req, timeout=240).read())
            reply = r["choices"][0]["message"]["content"].strip()
            msgs_api.append({"role": "assistant", "content": reply})
            m = re.search(r"\[SELF_PAD:\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\]",
                          reply)
            if m:
                agent_s = clamp3([float(x) for x in m.groups()])
            # 联动臂同题: vib 实测 + 引擎
            msgs_vib.append({"role": "user", "message": text})
            pad_v, _ = vib_read(net, ck, fe, msgs_vib)
            msgs_vib[-1]["pad"] = pad_v
            print(f"  q{qid:02d} 自报PAD={[round(x,2) for x in agent_s]} "
                  f"联动实测用户PAD={[round(x,2) for x in pad_v]}")
        print(f"   ← {p['name']} 完整自演绎轨迹已收集(与联动臂对照用)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="追加 LLM 自演绎对照(需API)")
    ap.add_argument("--model", default="deepseek/deepseek-flash")
    ap.add_argument("--base", default="https://open.cherryin.net/v1")
    ap.add_argument("--key", default=os.environ.get("OCC_API_KEY", ""))
    args = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    fe = BgeFeaturizer()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chordia_engine import ChordiaEngine
    engine = ChordiaEngine(ENGINE_CKPT, device="cpu")   # MLP 小, 钉CPU免设备错位

    summary = run_linked(net, ck, fe, engine)
    print("\n== 摘要 ==")
    for pid, s in summary.items():
        print(f"{pid}: 实测vs金标 r={s['corr']} | agent轨迹发散={s['traj_div']:.2f}")
    if args.llm:
        if not args.key:
            sys.exit("--llm 需要 --key 或 OCC_API_KEY")
        run_llm_selfplay(net, ck, fe, engine, args.model, args.base, args.key)


if __name__ == "__main__":
    main()
