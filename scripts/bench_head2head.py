# -*- coding: utf-8 -*-
"""bench_head2head — e5-small vs bge-m3 底座全面对比: 精度+延迟(p50/p90/p99)+内存。

同一 40 题真实输入, 各底座满血路径(CPU 纯 ONNX int8) + GPU 参照。
精度: 对独立裁判 zcode 的 PAD 逐维 Pearson r。
内存: /proc/self/status VmRSS 分阶段读数。
用法: PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 python3 bench_head2head.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from vibrato import vib_messages
from answer_sheet import questions, echo_chain_from_ref
from vibrato.feat_bge import BgeFeaturizer
from vibrato.net import VibratoNet

import torch

def rss_mb():
    with open("/proc/self/status") as fh:
        for ln in fh:
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) / 1024
    return 0.0

def gpu_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1

def pstats(ts):
    ts = sorted(ts)[max(0, len(ts) // 10):]
    n = len(ts)
    q = lambda p: ts[min(n - 1, int(n * p))] * 1000
    return q(0.5), q(0.9), q(0.99)

CONFIGS = {
    "bge-m3": {
        "ckpt": "checkpoints/ckpt_v6o/vibrato.pt",
        "head_fp32": "models/v6/vibrato_v6_fp32.onnx",
        "head_int8": "models/v6/vibrato_v6_int8.onnx",
        "bge_int8": "models/v6/bge_m3_int8.onnx",
        "tok": "models/v6/tokenizer/tokenizer.json",
        "vocab": "models/v6/vocab.json",
        "model_dir": "/root/models/bge-m3",
        "dim": 1024,
    },
    "e5-small": {
        "ckpt": "checkpoints/ckpt_e5s_s2/vibrato.pt",
        "head_fp32": "models/e5s/vibrato_e5s_fp32.onnx",
        "head_int8": "models/e5s/vibrato_e5s_int8.onnx",
        "bge_int8": "models/e5s/e5s_int8.onnx",
        "tok": "models/e5s/tokenizer/tokenizer.json",
        "vocab": "models/e5s/vocab.json",
        "model_dir": ("/root/.cache/huggingface/hub/models--intfloat--"
                      "multilingual-e5-small/snapshots/"
                      "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        "dim": 384,
    },
}

def build_cases(vocab_path, tok_path, model_dir, dim):
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    unk = len(vocab)
    tok = Tokenizer.from_file(tok_path)
    convos = {c["id"]: c for c in (json.loads(l) for l
                                   in open("data/test_convos.jsonl", encoding="utf-8")
                                   if l.strip())}
    qs, echo = questions(), echo_chain_from_ref()
    cases = []
    for i, (cid, idx, _t) in enumerate(qs):
        msgs = convos[cid]["messages"][max(0, idx - 6):idx + 1]
        vmsgs, pin = [], None
        for j, m in enumerate(msgs):
            vm = {"role": "user" if m["side"] == "user" else "assistant",
                  "message": str(m.get("text", ""))}
            if j < len(msgs) - 1 and vm["role"] == "user":
                pin = len(vmsgs)
            vmsgs.append(vm)
        if pin is not None and i > 0:
            vmsgs[pin]["pad"] = echo[i]
        units = vib_messages.sequence_units(vmsgs, max_window=512)
        ids, spos, svals, smask, segs = [], [], [], [], []
        char_seg, char_tok = [], []
        for u in units:
            if u["kind"] == "text":
                for ch in u["text"]:
                    ids.append(vocab.get(ch, unk)); spos.append(False)
                svals.extend([None] * len(u["text"])); smask.extend([None] * len(u["text"]))
                if u["text"].strip():
                    enc = tok.encode(u["text"], add_special_tokens=False)
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
        for si, e in enumerate(segs):
            bge_ids[si, :len(e.ids)] = e.ids
            bge_mask[si, :len(e.ids)] = 1
        z = [0.0] * 5
        cases.append(dict(
            ids=ids, spos=spos,
            svals=[v if v is not None else z for v in svals],
            smask=[v if v is not None else z for v in smask],
            echo=echo[i], bge_ids=bge_ids, bge_mask=bge_mask,
            char_seg=char_seg, char_tok=char_tok, L=L, dim=dim))
    return cases


def run_full_cpu(head_path, bge_path, cases):
    """分离式两会话 CPU 满血。返回 pads + 延迟列表。"""
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    hs = ort.InferenceSession(head_path, so, providers=["CPUExecutionProvider"])
    bs = ort.InferenceSession(bge_path, so, providers=["CPUExecutionProvider"])
    pads, ts = [], []
    for c in cases:
        t0 = time.perf_counter()
        hidden = bs.run(None, {"input_ids": c["bge_ids"],
                               "attention_mask": c["bge_mask"]})[0]  # (Nseg,T,dim)
        T = c["bge_ids"].shape[1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        idx = np.clip(np.array(c["char_seg"]), 0, None) * T + np.clip(np.array(c["char_tok"]), 0, None)
        f = flat[idx]
        feats = np.where((np.array(c["char_seg"]) < 0)[:, None], 0.0, f).astype(np.float32)
        o = hs.run(None, {
            "ids": np.array([c["ids"]], dtype=np.int64),
            "state_pos": np.array([c["spos"]], dtype=np.float32),
            "state_vals": np.array([c["svals"]], dtype=np.float32),
            "state_mask": np.array([c["smask"]], dtype=np.float32),
            "echo_prev": np.array([c["echo"]], dtype=np.float32),
            "feats": feats[None]})
        ts.append(time.perf_counter() - t0)
        pads.append(o[2][0])
    return np.array(pads), ts


def main():
    zc = [json.loads(l) for l in open("data/answers_zcode.jsonl", encoding="utf-8")]
    R = np.array([r["pad"] for r in zc])
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n{'═' * 20} {name} {'═' * 20}")
        r0 = rss_mb()

        # 精度 + CPU 延迟
        cases = build_cases(cfg["vocab"], cfg["tok"], cfg["model_dir"], cfg["dim"])
        pads, ts = run_full_cpu(cfg["head_int8"], cfg["bge_int8"], cases)
        r_bge = rss_mb() - r0
        p50, p90, p99 = pstats(ts)
        corr = [float(np.corrcoef(pads[:, d], R[:, d])[0, 1]) for d in range(3)]
        mae = float(np.abs(pads - R).mean())

        # 判头单独延迟
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        hs = ort.InferenceSession(cfg["head_int8"], so, providers=["CPUExecutionProvider"])
        ht = []
        for c in cases:
            t0 = time.perf_counter()
            hs.run(None, {
                "ids": np.array([c["ids"]], dtype=np.int64),
                "state_pos": np.array([c["spos"]], dtype=np.float32),
                "state_vals": np.array([c["svals"]], dtype=np.float32),
                "state_mask": np.array([c["smask"]], dtype=np.float32),
                "echo_prev": np.array([c["echo"]], dtype=np.float32),
                "feats": np.zeros((1, c["L"], cfg["dim"]), dtype=np.float32)})
            ht.append(time.perf_counter() - t0)
        hp50, hp90, hp99 = pstats(ht)
        r_head = rss_mb() - r0

        # GPU 参照
        ck = torch.load(cfg["ckpt"], map_location="cuda:0", weights_only=False)
        net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                         feat_dim=ck["feat_dim"]).cuda().eval()
        net.load_state_dict(ck["model"])
        if name == "bge-m3":
            fe = BgeFeaturizer()
        else:
            from feat_any import AnyFeaturizer
            fe = AnyFeaturizer(cfg["model_dir"])
        gt = []
        with torch.no_grad():
            for c in cases:
                t0 = time.perf_counter()
                o = net(torch.tensor([c["ids"]]).cuda(),
                        torch.tensor([c["spos"]], dtype=torch.bool).cuda(),
                        torch.tensor([c["svals"]]).cuda(),
                        torch.tensor([c["smask"]]).cuda(),
                        torch.tensor([c["echo"]]).cuda(),
                        torch.zeros(1, c["L"], cfg["dim"]).cuda(),
                        lengths=torch.tensor([c["L"]]).cuda())
                gt.append(time.perf_counter() - t0)
        gp50, gp90, gp99 = pstats(gt)
        vram = gpu_mb()
        r_total = rss_mb() - r0

        sz_head = os.path.getsize(cfg["head_int8"]) / 1e6
        sz_bge = os.path.getsize(cfg["bge_int8"]) / 1e6
        del net, fe
        torch.cuda.empty_cache()

        results[name] = dict(
            corr=corr, r_mean=float(np.mean(corr)), mae=mae,
            cpu_e2e=(p50, p90, p99), cpu_head=(hp50, hp90, hp99),
            gpu_head=(gp50, gp90, gp99),
            rss_head=r_head, rss_e2e=r_bge, rss_total=r_total, vram=vram,
            sz_head=sz_head, sz_bge=sz_bge)

        print(f"  精度: r̄={np.mean(corr):.3f} (P {corr[0]:.3f} / A {corr[1]:.3f} / D {corr[2]:.3f})  MAE={mae:.3f}")
        print(f"  CPU 满血 e2e:  p50={p50:.0f}ms  p90={p90:.0f}ms  p99={p99:.0f}ms")
        print(f"  CPU 判头 only: p50={hp50:.2f}ms  p90={hp90:.2f}ms  p99={hp99:.2f}ms")
        print(f"  GPU 判头:      p50={gp50:.2f}ms  p90={gp90:.2f}ms  p99={gp99:.2f}ms")
        print(f"  内存: 判头={r_head:.0f}MB  e2e={r_bge:.0f}MB  总进程={r_total:.0f}MB  VRAM={vram}MB")
        print(f"  磁盘: 判头={sz_head:.1f}MB  底座={sz_bge:.0f}MB")

    # 汇总表
    print(f"\n{'═' * 20} 汇总对比 {'═' * 20}")
    print(f"{'指标':<25} {'bge-m3 (568M)':>15} {'e5-small (118M)':>15} {'差':>8}")
    print("-" * 68)
    b, e = results["bge-m3"], results["e5-small"]
    rows = [
        ("r̄ (独立裁判)", f"{b['r_mean']:.3f}", f"{e['r_mean']:.3f}",
         f"{(e['r_mean'] - b['r_mean']) * 100:+.1f}%"),
        ("  P 愉悦", f"{b['corr'][0]:.3f}", f"{e['corr'][0]:.3f}", ""),
        ("  A 唤醒", f"{b['corr'][1]:.3f}", f"{e['corr'][1]:.3f}", ""),
        ("  D 支配", f"{b['corr'][2]:.3f}", f"{e['corr'][2]:.3f}", ""),
        ("CPU 满血 p50 (ms)", f"{b['cpu_e2e'][0]:.0f}", f"{e['cpu_e2e'][0]:.0f}",
         f"{(e['cpu_e2e'][0] / b['cpu_e2e'][0] - 1) * 100:+.0f}%"),
        ("CPU 满血 p99 (ms)", f"{b['cpu_e2e'][2]:.0f}", f"{e['cpu_e2e'][2]:.0f}", ""),
        ("CPU 判头 p50 (ms)", f"{b['cpu_head'][0]:.2f}", f"{e['cpu_head'][0]:.2f}", ""),
        ("GPU 判头 p50 (ms)", f"{b['gpu_head'][0]:.2f}", f"{e['gpu_head'][0]:.2f}", ""),
        ("RSS 判头 (MB)", f"{b['rss_head']:.0f}", f"{e['rss_head']:.0f}", ""),
        ("RSS e2e 常驻 (MB)", f"{b['rss_e2e']:.0f}", f"{e['rss_e2e']:.0f}",
         f"{(e['rss_e2e'] / b['rss_e2e'] - 1) * 100:+.0f}%"),
        ("VRAM (MB)", f"{b['vram']}", f"{e['vram']}", ""),
        ("磁盘 判头 (MB)", f"{b['sz_head']:.1f}", f"{e['sz_head']:.1f}", ""),
        ("磁盘 底座 (MB)", f"{b['sz_bge']:.0f}", f"{e['sz_bge']:.0f}",
         f"{(e['sz_bge'] / b['sz_bge'] - 1) * 100:+.0f}%"),
    ]
    for label, bv, ev, delta in rows:
        print(f"{label:<25} {bv:>15} {ev:>15} {delta:>8}")

    json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                   for kk, vv in v.items()}
               for k, v in results.items()},
              open("bench_head2head.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n→ bench_head2head.json")


if __name__ == "__main__":
    main()
