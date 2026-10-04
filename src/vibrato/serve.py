# -*- coding: utf-8 -*-
"""serve.py — Vibrato HTTP 服务（零 torch，ONNX 推理）

端点:
  POST /v1/affect/score    v1/affect 原生接口（消息数组 → PAD + OCC-22 + noul×2 + conf）
  GET  /health             健康检查

认证: X-API-KEY 请求头，值须以 sk-vibrato- 开头且与 VIBRATO_API_KEY 环境变量一致。
启动: uvicorn serve:app --host 0.0.0.0 --port ${VIBRATO_PORT:-18973}
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from . import vib_messages
from .occ import occ_rank

# ── API Key 校验 ───────────────────────────────────────────
KEY_PREFIX = "sk-vibrato-"
_HEX = set("0123456789abcdefABCDEF")

def validate_api_key(provided: str, expected: str) -> bool:
    """格式校验（前缀+hex字符集）+ 值比对。"""
    if not provided or not provided.startswith(KEY_PREFIX):
        return False
    hex_part = provided[len(KEY_PREFIX):]
    if not hex_part or not all(c in _HEX for c in hex_part):
        return False
    return provided == expected

# ── 模型加载（启动时一次） ────────────────────────────────
VIBRATO_API_KEY = os.environ.get("VIBRATO_API_KEY", "")
if not VIBRATO_API_KEY:
    raise RuntimeError("缺少 VIBRATO_API_KEY 环境变量")

HEAD_PATH = os.environ.get("VIBRATO_HEAD", os.path.join(HERE, "models/e5s/vibrato_e5s_int8.onnx"))
BGE_PATH = os.environ.get("VIBRATO_BGE", os.path.join(HERE, "models/e5s/e5s_int8.onnx"))
VOCAB_PATH = os.environ.get("VIBRATO_VOCAB", os.path.join(HERE, "models/e5s/vocab.json"))
TOK_PATH = os.environ.get("VIBRATO_TOK", os.path.join(HERE, "models/e5s/tokenizer/tokenizer.json"))

t0 = time.time()
_vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
_unk = len(_vocab)
_tok = Tokenizer.from_file(TOK_PATH)
_so = ort.SessionOptions()
_so.intra_op_num_threads = int(os.environ.get("VIBRATO_THREADS", "2"))
_head = ort.InferenceSession(HEAD_PATH, _so, providers=["CPUExecutionProvider"])
_bge = ort.InferenceSession(BGE_PATH, _so, providers=["CPUExecutionProvider"])
LOAD_MS = (time.time() - t0) * 1000

# 从 battery 获取 noul ID（只取 2 个核心）
NOUL_CORE = ["directed_at_me", "suppressed"]
# 网络内部 5 个 noul 头的索引
NOUL_ALL = ["negative", "needs_comfort", "directed_at_me", "escalating", "suppressed"]
NOUL_IDX = {qid: NOUL_ALL.index(qid) for qid in NOUL_CORE}

print(f"[vibrato] 模型加载 {LOAD_MS:.0f}ms | head={os.path.basename(HEAD_PATH)} | bge={os.path.basename(BGE_PATH)}")

# ── 推理 ───────────────────────────────────────────────────
def _score(messages: List[Dict]) -> Dict:
    vib_messages.validate_messages(messages)
    echo = next((m["pad"] for m in reversed(messages[:-1])
                 if m["role"] == "user" and "pad" in m), [0.0] * 3)
    units = vib_messages.sequence_units(messages, max_window=512)
    ids, spos, svals, smask, segs = [], [], [], [], []
    char_seg, char_tok = [], []
    for u in units:
        if u["kind"] == "text":
            for ch in u["text"]:
                ids.append(_vocab.get(ch, _unk)); spos.append(False)
            svals.extend([None] * len(u["text"])); smask.extend([None] * len(u["text"]))
            if u["text"].strip():
                enc = _tok.encode(u["text"], add_special_tokens=False)
                s = len(segs); segs.append(enc)
                cs = [-1] * len(u["text"]); ct = [-1] * len(u["text"])
                for t, (a, b) in enumerate(enc.offsets):
                    for k in range(a, min(b, len(u["text"]))):
                        cs[k], ct[k] = s, t
                char_seg.extend(cs); char_tok.extend(ct)
            else:
                char_seg.extend([-1] * len(u["text"])); char_tok.extend([-1] * len(u["text"]))
        else:
            ids.append(0); spos.append(True); svals.append(u["vals"]); smask.append(u["mask"])
            char_seg.append(-1); char_tok.append(-1)
    L = len(ids)
    T = max((len(e.ids) for e in segs), default=1)
    bge_ids = np.zeros((max(len(segs), 1), T), dtype=np.int64)
    bge_mask = np.zeros((max(len(segs), 1), T), dtype=np.int64)
    for i, e in enumerate(segs):
        bge_ids[i, :len(e.ids)] = e.ids
        bge_mask[i, :len(e.ids)] = 1
    z = [0.0] * 5
    hidden = _bge.run(None, {"input_ids": bge_ids, "attention_mask": bge_mask})[0]
    flat = hidden.reshape(-1, hidden.shape[-1])
    idx = np.clip(np.array(char_seg), 0, None) * T + np.clip(np.array(char_tok), 0, None)
    f = flat[idx]
    feats = np.where((np.array(char_seg) < 0)[:, None], 0.0, f).astype(np.float32)
    o = _head.run(None, {
        "ids": np.array([ids], dtype=np.int64),
        "state_pos": np.array([spos], dtype=np.float32),
        "state_vals": np.array([[v if v is not None else z for v in svals]], dtype=np.float32),
        "state_mask": np.array([[v if v is not None else z for v in smask]], dtype=np.float32),
        "echo_prev": np.array([echo], dtype=np.float32),
        "feats": feats[None]})
    pad = o[2][0].tolist()          # pad_norm
    pad_bins = o[0][0].tolist()     # (3,9)
    conf = float(o[5][0])
    noul_probs = o[4][0]            # (5,2) → 取 [*, 1] 为真概率

    ranked = occ_rank(pad)

    # write_back: 本轮输出钉回最后一条 user 消息
    d_p = 0.0
    if len(messages) >= 2:
        prev = next((m["pad"] for m in reversed(messages[:-1])
                     if m["role"] == "user" and "pad" in m), None)
        if prev:
            import pad_schema as _ps
            d_p = _ps.pad_to_pressure_delta(pad[0] - prev[0], pad[1] - prev[1], pad[2] - prev[2])

    return {
        "pad": {"p": round(pad[0], 4), "a": round(pad[1], 4), "d": round(pad[2], 4)},
        "pad_bins": {"p": [round(v, 5) for v in pad_bins[0]],
                     "a": [round(v, 5) for v in pad_bins[1]],
                     "d": [round(v, 5) for v in pad_bins[2]]},
        "occ": ranked,
        "noul": {qid: round(float(noul_probs[NOUL_IDX[qid]][1]), 4) for qid in NOUL_CORE},
        "confidence": round(conf, 4),
        "write_back": {"pad": [round(v, 4) for v in pad],
                       "d_pressure_raw": round(d_p, 4)},
    }


# ── FastAPI ────────────────────────────────────────────────
app = FastAPI(title="Vibrato", version="0.1.1",
              description="System One emotion model — v1/affect")

class ScoreRequest(BaseModel):
    messages: List[Dict]

@app.post("/v1/affect/score")
async def score(req: ScoreRequest, x_api_key: str = Header(...)):
    if not validate_api_key(x_api_key, VIBRATO_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        result = _score(req.messages)
        return {"schema": "v1/affect", "version": "0.1.1", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.1", "model_load_ms": round(LOAD_MS)}
