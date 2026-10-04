# -*- coding: utf-8 -*-
"""demo_html — 生成 HTML 版核心演示: 微信式气泡流 + 颤音逐条侧车评价。

三臂数据全部内嵌(离线单文件):
  颤音实测: 每条用户消息的 PAD/族/noul 侧车卡(WechatVibe 式展示)
  弦音 v2 联动: 引擎驱动的 agent PAD 轨迹(颤音臂 + 金标臂)
  LLM 自演绎: --llm 收集的自报轨迹与回复文本(有则展示为左侧气泡)

用法: cd /root/vibrato && PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 \
      python3 demo/demo_html.py [--llm --key sk-..]
输出: demo/vibrato_demo.html (数据内嵌, 浏览器直接打开)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import vib_messages
from battery import FAMILY_IDS, NOUL_IDS
from demo_scenarios import PERSONALITIES, QUESTIONS, SCENARIOS
from eval_bench import make_batch
from feat_bge import BgeFeaturizer
from net import VibratoNet

CKPT = "ckpt_v6o/vibrato.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_CKPT = os.path.join(HERE, "checkpoints", "chordia_v0.0.1-alpha.pth")
LLM_JSON = os.path.join(HERE, "demo_llm.json")
OUT_HTML = os.path.join(HERE, "vibrato_demo.html")

SCENE_OF = {qid: s for s, ids in SCENARIOS.items() for qid in ids}
SCENE_NAME = {"anger": "第一幕 · 愤怒质问", "breakdown": "第二幕 · 崩溃恢复",
              "growth": "第三幕 · 从自大到友谊"}


def clamp3(v):
    return [max(-1.0, min(1.0, x)) for x in v]


OCC_ZH = {"happy_for": "为你高兴", "gloating": "幸灾乐祸", "resentment": "怨恨",
          "pity": "心疼", "hope": "期待", "fear": "担忧", "satisfaction": "满意",
          "fears_confirmed": "应验", "relief": "释然", "disappointment": "失望",
          "joy": "喜悦", "distress": "苦恼", "pride": "骄傲", "admiration": "钦佩",
          "shame": "羞愧", "reproach": "责备", "gratification": "欣慰",
          "remorse": "懊悔", "love": "喜爱", "hate": "厌恶", "liking": "好感",
          "disliking": "轻度反感"}


def occ_centroids():
    """OCC-22 实测质心(自家 occ_cases.jsonl 2106 条, [-4,4]→[-1,1])——侧车情绪
    解码表: score 头的 PAD → 距离最近 top2(Nexus 生产口径)。"""
    from collections import defaultdict
    acc = defaultdict(list)
    with open("occ_cases.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                acc[r["occ"]].append(r["pad"])
    return {k: dict(zh=OCC_ZH.get(k, k),
                    pad=[round(sum(p[i] for p in ps) / len(ps) / 4.0, 3)
                         for i in range(3)])
            for k, ps in acc.items()}


@torch.no_grad()
def vib_read(net, ck, fe, messages):
    ids, sp, sv, sm, ft, Ls = make_batch([messages], ck, fe)
    echo = torch.tensor([[0.0, 0.0, 0.0]] if len(messages) == 1
                        else [next((m["pad"] for m in reversed(messages[:-1])
                                    if m["role"] == "user" and "pad" in m), [0.0] * 3)])
    out = net(ids.cuda(), sp.cuda(), sv.cuda(), sm.cuda(), echo.cuda(),
              torch.from_numpy(ft).cuda(), lengths=Ls.cuda())
    return (out["pad_norm"][0].tolist(), FAMILY_IDS[out["family_probs"][0].argmax().item()],
            {q: bool(out["noul_probs"][0, j, 1] > 0.5) for j, q in enumerate(NOUL_IDS)})


def collect_linked(net, ck, fe, engine):
    personas = {}
    msgs: list = []
    turns = []
    for qid, text, ctx, gold in QUESTIONS:
        msgs.append({"role": "user", "message": text})
        pad_v, fam, noul = vib_read(net, ck, fe, msgs)
        msgs[-1]["pad"] = pad_v
        turns.append(dict(qid=qid, text=text, ctx=ctx, gold=gold,
                          vib_pad=[round(x, 3) for x in pad_v],
                          vib_family=fam, noul=noul))
    for pid, p in PERSONALITIES.items():
        agent_v, agent_g = list(p["base_pad"]), list(p["base_pad"])
        traj_v, traj_g = [], []
        for t in turns:
            d = engine.predict(user_pad=t["vib_pad"], vitality=p["vitality"],
                               current_pad=agent_v,
                               personality_scale=p["personality_scale"])["delta_pad"]
            agent_v = clamp3([a + x for a, x in zip(agent_v, d)])
            traj_v.append([round(x, 3) for x in agent_v])
            d = engine.predict(user_pad=t["gold"], vitality=p["vitality"],
                               current_pad=agent_g,
                               personality_scale=p["personality_scale"])["delta_pad"]
            agent_g = clamp3([a + x for a, x in zip(agent_g, d)])
            traj_g.append([round(x, 3) for x in agent_g])
        pv = np.array([t["vib_pad"] for t in turns])
        pg = np.array([t["gold"] for t in turns])
        corr = [round(float(np.corrcoef(pv[:, i], pg[:, i])[0, 1]), 3) for i in range(3)]
        personas[pid] = dict(meta=p, traj_engine=traj_v, traj_gold=traj_g,
                             corr=corr, vitality=p["vitality"])
    return turns, personas


SYSTEM_PROMPT = """你是一个女性AI助手，{description}。

你的人格特征：
- 愉悦度基准: {pleasure:.2f} (-1到+1，越正越快乐)
- 唤醒度基准: {arousal:.2f} (-1到+1，越正越活跃)
- 支配度基准: {dominance:.2f} (-1到+1，越正越自信)

请始终保持这个人格特征与用户对话。每轮回复后，另起一行以
[SELF_PAD: P,A,D] 报告你此刻自己的真实情绪状态(三个-1到1的数)。"""


SYSTEM_LINKED = """你是一个女性AI助手，{description}。

你的底层情绪系统（弦音）实时维护你的状态。以下是【此刻】的真实数值——请让回复的语气、节奏、用词自然对齐这些状态（低落时语气放轻放缓、激动时急切、活力低时显得疲惫），但不要直接提及或报出任何数值：
- 愉悦度 P: {p:+.2f}（-1 低落 ~ +1 愉悦）
- 唤醒度 A: {a:+.2f}（-1 平静 ~ +1 激动）
- 支配度 D: {d:+.2f}（-1 顺从无措 ~ +1 自信主导）
- 活力: {vitality:.0f}/100

同时，情绪传感系统（颤音）实测用户此刻情绪为: 愉悦 {up:+.2f} / 唤醒 {ua:+.2f} / 支配 {ud:+.2f}——请据此调整共情与安抚的深度。回复 60 字以内。"""


def _chat(model, base, key, messages, max_tokens=400):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.8, "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    return r["choices"][0]["message"]["content"].strip()


def collect_linked_story(model, base, key, net, ck, fe, engine, personas, turns):
    """联动剧情臂: 颤音实测用户 → 弦音引擎更新 agent PAD → agent 在该状态下回复。
    每轮回复由 [引擎 agent PAD + 活力 + 颤音实测用户 PAD] 实时条件化——剧情即联动。"""
    out = {}
    for pid, p in personas.items():
        meta = p["meta"]
        agent = list(meta["base_pad"])
        api, replies, cond = [], [], []
        msgs: list = []
        for t in turns:
            msgs.append({"role": "user", "message": t["text"]})
            pad_v, fam, noul = vib_read(net, ck, fe, msgs)
            msgs[-1]["pad"] = pad_v
            d = engine.predict(user_pad=pad_v, vitality=meta["vitality"],
                               current_pad=agent,
                               personality_scale=meta["personality_scale"])["delta_pad"]
            agent = clamp3([a + x for a, x in zip(agent, d)])
            cond.append({"agent_pad": [round(x, 3) for x in agent],
                         "user_pad": [round(x, 3) for x in pad_v]})
            sysmsg = SYSTEM_LINKED.format(
                description=meta["description"], p=agent[0], a=agent[1], d=agent[2],
                vitality=meta["vitality"], up=pad_v[0], ua=pad_v[1], ud=pad_v[2])
            api = [{"role": "system", "content": sysmsg}] + \
                  [m for m in api if m["role"] != "system"]
            api.append({"role": "user", "content": t["text"]})
            reply = _chat(model, base, key, api)
            api.append({"role": "assistant", "content": reply})
            replies.append(reply)
            print(f"  {pid} linked q{t['qid']:02d} agentPAD={agent} ok", flush=True)
        out[pid] = dict(replies=replies, cond=cond)
    return out


def collect_llm(model, base, key):
    out = {}
    for pid, p in PERSONALITIES.items():
        api = [{"role": "system", "content": SYSTEM_PROMPT.format(
            description=p["description"], pleasure=p["base_pad"][0],
            arousal=p["base_pad"][1], dominance=p["base_pad"][2])}]
        traj, replies = [], []
        for qid, text, ctx, gold in QUESTIONS:
            api.append({"role": "user", "content": text})
            reply = _chat(model, base, key, api, max_tokens=500)
            api.append({"role": "assistant", "content": reply})
            m = re.search(r"\[SELF_PAD:\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\]",
                          reply)
            traj.append(clamp3([float(x) for x in m.groups()]) if m else (traj[-1] if traj
                                                                           else p["base_pad"]))
            replies.append(re.sub(r"\[SELF_PAD:[^\]]*\]", "", reply).strip())
            print(f"  {pid} q{qid:02d} ok", flush=True)
        out[pid] = dict(traj_self=[[round(x, 3) for x in t] for t in traj],
                        replies=replies)
    return out


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Vibrato × Chordia v2 — 联动演示</title>
<style>
:root{--bg:#ededed;--me:#95ec69;--ink:#111;--sub:#888;--card:#fff;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);}
header{background:linear-gradient(120deg,#2b3a55,#4a5d8a);color:#fff;padding:18px 24px;}
header h1{font-size:20px;font-weight:600;}
header p{font-size:12px;opacity:.85;margin-top:4px;}
.wrap{max-width:1200px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;}
@media(max-width:980px){.wrap{grid-template-columns:1fr;}}
.tabs{display:flex;gap:8px;margin:14px 24px 0;}
.tab{padding:8px 16px;border-radius:18px;background:#d7d7d7;font-size:13px;cursor:pointer;user-select:none;}
.tab.on{background:#07c160;color:#fff;font-weight:600;}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:12px 24px 0;}
.stat{background:#fff;border-radius:10px;padding:8px 14px;font-size:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);}
.stat b{display:block;font-size:15px;margin-top:2px;}
.chat{background:#f5f5f5;border-radius:12px;padding:14px;overflow-y:auto;max-height:82vh;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.scene{ text-align:center;margin:12px auto 6px;font-size:12px;color:#fff;background:#b9b9b9;display:inline-block;padding:3px 12px;border-radius:10px;}
.row{display:flex;margin:10px 0;align-items:flex-start;}
.row.user{flex-direction:row-reverse;}
.avatar{width:38px;height:38px;border-radius:6px;flex:none;display:flex;align-items:center;justify-content:center;font-size:18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.1);}
.col{max-width:78%;display:flex;flex-direction:column;gap:5px;margin:0 9px;}
.row.user .col{align-items:flex-end;}
.bubble{padding:9px 12px;border-radius:8px;font-size:14px;line-height:1.55;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.06);white-space:pre-wrap;text-align:left;}
.row.user .bubble{background:var(--me);}
.side{background:var(--card);border-radius:9px;padding:8px 11px;font-size:11px;box-shadow:0 1px 3px rgba(0,0,0,.08);width:330px;}
.side .cap{color:var(--sub);margin-bottom:5px;display:flex;justify-content:space-between;align-items:center;}
.fam{color:#fff;border-radius:9px;padding:1px 9px;font-weight:600;}
.padbars{display:grid;grid-template-columns:14px 1fr;gap:3px 6px;align-items:center;margin:4px 0;}
.dim{font-size:10px;color:var(--sub);text-align:right;}
.bar{height:8px;background:#eee;border-radius:4px;position:relative;overflow:hidden;}
.bar i{position:absolute;top:0;bottom:0;border-radius:4px;}
.gold{height:4px;position:relative;margin-top:2px;}
.gold i{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,#bbb,#bb 3px,#999 3px,#999 6px);border-radius:2px;}
.chips{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px;}
.chip{padding:1px 7px;border-radius:8px;background:#eee;color:#999;font-size:10px;}
.chip.on{background:#ffe2e2;color:#d33;font-weight:600;}
.mtabs{display:flex;gap:8px;margin:0 0 10px;}
.mtab{padding:7px 14px;border-radius:16px;background:#d7d7d7;font-size:13px;cursor:pointer;user-select:none;}
.mtab.on{background:#d6385e;color:#fff;font-weight:600;}
.side.chordia{border-left:3px solid #d6385e;}
.vit{margin-top:6px;font-size:10px;color:#999;}
.vitbar{height:6px;background:#eee;border-radius:3px;overflow:hidden;margin-top:2px;}
.vitbar i{display:block;height:100%;background:linear-gradient(90deg,#f5a623,#7bb735);}
.ochip{color:#fff;border-radius:9px;padding:2px 10px;font-weight:600;font-size:12px;}
.ochip2{border:1.5px solid;border-radius:9px;padding:1px 8px;font-size:11px;background:#fff;}
.dim2{opacity:.35;}
.upad{margin-top:5px;font-size:10px;color:#888;}
.panel{position:sticky;top:16px;display:flex;flex-direction:column;gap:12px;}
.card{background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.card h3{font-size:13px;margin-bottom:8px;color:#444;}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;margin-bottom:6px;}
.lg{display:flex;align-items:center;gap:4px;color:#555;}
.lg i{width:16px;height:3px;border-radius:2px;display:inline-block;}
.dims{display:flex;gap:6px;}
.dimtab{font-size:12px;padding:3px 12px;border-radius:12px;background:#eee;cursor:pointer;}
.dimtab.on{background:#2b3a55;color:#fff;}
footer{padding:14px 24px;font-size:11px;color:#999;}
</style>
</head>
<body>
<header>
  <h1>颤音 Vibrato × 弦音 Chordia v2 — 情绪联动演示</h1>
  <p>🎭 联动剧情：颤音逐轮实测用户情绪 → 弦音引擎更新 agent 状态 → agent 在该状态下回复（每句台词被实时驱动）· 🤖 LLM 自演绎对照：仅人格提示词自我想象 · 卡片斜纹小条 = 剧本手工金标</p>
</header>
<div class="tabs" id="tabs"></div>
<div class="stats" id="stats"></div>
<div class="wrap">
  <div>
    <div class="mtabs" id="mtabs"></div>
    <div class="chat" id="chat"></div>
  </div>
  <div class="panel">
    <div class="card">
      <h3>轨迹对照（切换维度）</h3>
      <div class="legend">
        <span class="lg"><i style="background:#07c160"></i>颤音实测·用户</span>
        <span class="lg"><i style="background:repeating-linear-gradient(45deg,#777,#777 3px,#aaa 3px,#aaa 6px)"></i>手工金标·用户</span>
        <span class="lg"><i style="background:#d6385e"></i>弦音·agent(颤音驱动)</span>
        <span class="lg"><i style="background:#8e6dd8"></i>LLM 自演绎·自报</span>
      </div>
      <div class="dims" id="dims"></div>
      <svg id="chart" viewBox="0 0 380 240" style="width:100%;margin-top:8px;"></svg>
    </div>
    <div class="card" id="agentcard"></div>
  </div>
</div>
<footer>零 API 本地计算：颤音 v6.3 (1.86M) 实测用户情绪 → 弦音 v2 引擎(7维MLP)驱动 agent 动力学。LLM 自演绎臂为 deepseek-flash 对照。</footer>
<script>
const DATA = @DATA@;
const FAM_COLOR = {happy:"#f5a623",affectionate:"#ec5f8f",calm:"#4a90d9",amused:"#7bb735",sad:"#5d6d9e",anxious:"#9a6fd0",angry:"#d6385e",wronged:"#c97b3d"};
let curP = Object.keys(DATA.personas)[0], curD = 0, MODE = "linked", DIMS = ["P 愉悦","A 唤醒","D 支配"];

function occTop2(p){
  return Object.entries(DATA.occ).map(([k,o])=>{
    const q=[p[0]*4,p[1]*4,p[2]*4], c=o.pad.map(x=>x*4);
    return {zh:o.zh, d:Math.hypot(q[0]-c[0],q[1]-c[1],q[2]-c[2]), v:o.pad[0]};
  }).sort((a,b)=>a.d-b.d).slice(0,2);
}
function occColor(v){ return v>0.3 ? "#2e9e5b" : (v<-0.3 ? "#d6385e" : "#4a6fa5"); }
function ochChips(p){
  const [a,b] = occTop2(p);
  const wall = (b.d/a.d) < 1.3;                    // 骑墙规则(Nexus 同款)
  return `<span class="ochip" style="background:${occColor(a.v)}">${a.zh}</span>` +
    `<span class="ochip2 ${wall?"":"dim2"}" style="border-color:${occColor(b.v)};color:${occColor(b.v)}">${wall?"≈ ":""}${b.zh}</span>`;
}
function padBar(v, color){
  const w = Math.abs(v)/2*50, left = v<0 ? 50-w : 50;
  return `<div class="bar"><i style="left:${left}%;width:${w}%;background:${color}"></i></div>`;
}
function goldBar(v){
  const w = Math.abs(v)/2*50, left = v<0 ? 50-w : 50;
  return `<div class="gold"><i style="left:${left}%;width:${w}%"></i></div>`;
}
function sideCard(t, i){
  const p = t.vib_pad, g = t.gold;
  const chips = Object.entries(t.noul||{}).map(([k,v])=>{
    const zh = {negative:"负面",needs_comfort:"需安慰",directed_at_me:"指向我",escalating:"升级",suppressed:"压抑"}[k];
    return `<span class="chip ${v?"on":""}">${zh}</span>`;}).join("");
  return `<div class="side"><div class="cap"><span>颤音侧车评价 #${t.qid} · score→OCC</span></div>
    <div class="chips" style="margin:0 0 5px">${ochChips(p)}</div>
    <div class="padbars">
      <span class="dim">P</span><div>${padBar(p[0],"#07c160")}${goldBar(g[0])}</div>
      <span class="dim">A</span><div>${padBar(p[1],"#f5a623")}${goldBar(g[1])}</div>
      <span class="dim">D</span><div>${padBar(p[2],"#4a90d9")}${goldBar(g[2])}</div>
    </div><div class="chips">${chips}</div></div>`;
}
function chordiaCard(story, i, vit){
  const c = story.cond[i], ap = c.agent_pad, up = c.user_pad;
  return `<div class="side chordia" style="width:300px">
    <div class="cap"><span>弦音 v2 驱动 · 第${i+1}轮回复时的状态</span></div>
    <div class="chips" style="margin:0 0 5px">${ochChips(ap)}</div>
    <div class="padbars">
      <span class="dim">P</span><div>${padBar(ap[0],"#d6385e")}</div>
      <span class="dim">A</span><div>${padBar(ap[1],"#f5a623")}</div>
      <span class="dim">D</span><div>${padBar(ap[2],"#4a90d9")}</div>
    </div>
    <div class="vit">活力 ${vit}/100<div class="vitbar"><i style="width:${(vit+30)/130*100}%"></i></div></div>
    <div class="upad">↳ 本轮颤音实测用户: P ${up[0].toFixed(2)} / A ${up[1].toFixed(2)} / D ${up[2].toFixed(2)}</div>
  </div>`;
}
function renderChat(){
  const ps = DATA.personas[curP], llm = (DATA.llm||{})[curP], story = ps.story;
  let html = "", lastScene = "";
  DATA.turns.forEach((t,i)=>{
    const sc = DATA.scene_of[String(t.qid)];
    if(sc!==lastScene){html+=`<div style="text-align:center;margin:14px 0 4px"><span class="scene">${DATA.scene_name[sc]}</span></div>`;lastScene=sc;}
    html+=`<div class="row user"><div class="avatar">👤</div><div class="col"><div class="bubble">${t.text}</div>${sideCard(t,i)}</div></div>`;
    if(MODE==="linked" && story){
      html+=`<div class="row"><div class="avatar">🎻</div><div class="col"><div class="bubble" style="font-size:13px;color:#333">${story.replies[i]||""}</div>${chordiaCard(story,i,ps.meta.vitality)}</div></div>`;
    } else if(llm){
      html+=`<div class="row"><div class="avatar">🤖</div><div class="col"><div class="bubble" style="font-size:13px;color:#333">${llm.replies[i]||""}</div>
      <div class="side" style="width:260px"><div class="cap"><span>LLM 自演绎·自报</span></div>
      <div class="chips" style="margin:0 0 5px">${ochChips(llm.traj_self[i])}</div>
      <div class="padbars"><span class="dim">P</span><div>${padBar(llm.traj_self[i][0],"#8e6dd8")}</div>
      <span class="dim">A</span><div>${padBar(llm.traj_self[i][1],"#8e6dd8")}</div>
      <span class="dim">D</span><div>${padBar(llm.traj_self[i][2],"#8e6dd8")}</div></div>
      <div class="upad">↳ 本轮颤音实测用户: P ${t.vib_pad[0].toFixed(2)} / A ${t.vib_pad[1].toFixed(2)} / D ${t.vib_pad[2].toFixed(2)}</div></div></div></div>`;
    }
  });
  document.getElementById("chat").innerHTML = html;
  const meta = ps.meta;
  const endTxt = story ? `联动剧情末态 ${JSON.stringify(story.cond[story.cond.length-1].agent_pad)}`
                       : `agent 末态(颤音驱动) ${JSON.stringify(ps.traj_engine[ps.traj_engine.length-1])}`;
  document.getElementById("agentcard").innerHTML = `<h3>弦音 v2 · ${meta.name}</h3>
    <div style="font-size:12px;color:#666;line-height:1.8">base_pad ${JSON.stringify(meta.base_pad)} · 人格缩放 ×${meta.personality_scale} · 活力 ${meta.vitality}<br>
    颤音实测 vs 手工金标 r(P/A/D) = ${ps.corr.join(" / ")}<br>
    ${endTxt}<br>
    agent 末态(金标驱动) ${JSON.stringify(ps.traj_gold[ps.traj_gold.length-1])}</div>`;
}
function renderChart(){
  const ps = DATA.personas[curP], llm = (DATA.llm||{})[curP];
  const W=380,H=240,L=34,R=10,T=12,B=26,n=DATA.turns.length;
  const X=i=>L+(W-L-R)*i/(n-1), Y=v=>T+(H-T-B)*(1-(v+1)/2);
  let s=`<line x1="${L}" y1="${Y(0)}" x2="${W-R}" y2="${Y(0)}" stroke="#ddd"/>`;
  [1,-1].forEach(v=>s+=`<line x1="${L}" y1="${Y(v)}" x2="${W-R}" y2="${Y(v)}" stroke="#eee"/>
    <text x="4" y="${Y(v)+3}" font-size="9" fill="#aaa">${v}</text>`);
  for(let i=1;i<n;i+=3)s+=`<text x="${X(i)-3}" y="${H-8}" font-size="8" fill="#aaa">${i+1}</text>`;
  const line=(arr,c,dash)=>{let d="";arr.forEach((v,i)=>d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" ");
    return `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" ${dash?'stroke-dasharray="5 3"':""}/>`;};
  s+=line(DATA.turns.map(t=>t.gold[curD]),"#999",true);
  if(llm)s+=line(llm.traj_self.map(t=>t[curD]),"#8e6dd8");
  const eng = ps.story ? ps.story.cond.map(c=>c.agent_pad[curD])
                       : ps.traj_engine.map(t=>t[curD]);
  s+=line(eng,"#d6385e");
  s+=line(DATA.turns.map(t=>t.vib_pad[curD]),"#07c160");
  document.getElementById("chart").innerHTML=s;
}
function renderTabs(){
  document.getElementById("tabs").innerHTML = Object.entries(DATA.personas).map(([pid,p])=>
    `<div class="tab ${pid===curP?"on":""}" onclick="curP='${pid}';renderAll()">${p.meta.name}</div>`).join("");
  document.getElementById("mtabs").innerHTML =
    `<div class="mtab ${MODE==="linked"?"on":""}" onclick="MODE='linked';renderAll()">🎭 联动剧情（颤音×弦音 v2）</div>
     <div class="mtab ${MODE==="self"?"on":""}" onclick="MODE='self';renderAll()">🤖 LLM 自演绎对照</div>`;
  document.getElementById("dims").innerHTML = DIMS.map((d,i)=>
    `<div class="dimtab ${i===curD?"on":""}" onclick="curD=${i};renderAll()">${d}</div>`).join("");
}
function renderStats(){
  const ps = DATA.personas[curP];
  const engEnd = ps.story ? ps.story.cond.at(-1).agent_pad
                          : ps.traj_engine.at(-1);   // 有剧情臂用实时轨迹(审查#6口径统一)
  document.getElementById("stats").innerHTML =
    `<div class="stat">颤音 vs 金标 r(P/A/D)<b>${ps.corr.join(" / ")}</b></div>
     <div class="stat">联动 agent 末态<b>${JSON.stringify(engEnd)}</b></div>
     <div class="stat">金标驱动末态<b>${JSON.stringify(ps.traj_gold.at(-1))}</b></div>` +
    (DATA.llm?`<div class="stat">LLM 自演绎末态<b>${JSON.stringify(DATA.llm[curP].traj_self.at(-1))}</b></div>`:"");
}
function renderAll(){renderTabs();renderStats();renderChat();renderChart();}
renderAll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="重收 LLM 自演绎臂(需 --key)")
    ap.add_argument("--model", default="deepseek/deepseek-flash")
    ap.add_argument("--base", default="https://open.cherryin.net/v1")
    ap.add_argument("--key", default=os.environ.get("OCC_API_KEY", ""))
    args = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"]).cuda().eval()
    net.load_state_dict(ck["model"])
    fe = BgeFeaturizer()
    from chordia_engine import ChordiaEngine
    engine = ChordiaEngine(ENGINE_CKPT, device="cpu")

    print("联动臂…", flush=True)
    turns, personas = collect_linked(net, ck, fe, engine)
    llm = None
    if args.llm:
        if not args.key:
            sys.exit("--llm 需要 --key")
        print("联动剧情臂（agent 台词由引擎状态实时驱动）…", flush=True)
        linked = collect_linked_story(args.model, args.base, args.key, net, ck, fe,
                                      engine, personas, turns)
        for pid, st in linked.items():
            personas[pid]["story"] = st
        json.dump(linked, open(os.path.join(HERE, "demo_linked.json"), "w",
                               encoding="utf-8"), ensure_ascii=False)
        print("LLM 自演绎对照臂…", flush=True)
        llm = collect_llm(args.model, args.base, args.key)
        json.dump(llm, open(LLM_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    else:
        lj = os.path.join(HERE, "demo_linked.json")
        if os.path.exists(lj):
            for pid, st in json.load(open(lj, encoding="utf-8")).items():
                personas[pid]["story"] = st
            print("剧情臂: 复用 demo_linked.json")
        if os.path.exists(LLM_JSON):
            llm = json.load(open(LLM_JSON, encoding="utf-8"))
            print("LLM 臂: 复用 demo_llm.json")

    data = {"turns": turns, "personas": personas, "llm": llm,
            "occ": occ_centroids(), "scene_of": SCENE_OF, "scene_name": SCENE_NAME}
    html = HTML.replace("@DATA@", json.dumps(data, ensure_ascii=False))
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"→ {OUT_HTML} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
