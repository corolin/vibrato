# -*- coding: utf-8 -*-
"""Vibrato 推理延迟实测 — 单条消息前向, GPU 与 CPU 各测, 与教师(4B)对照。

用法: python bench_latency.py --ckpt ckpt_v2/vibrato.pt [--device cuda:0|cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEXT = ("TARGET(需判断): [用户]: 哈哈哈哈今天也太顺利了吧，中午还抽中了奶茶！\n"
        "[AI]: 听起来今天是幸运日呀～\n"
        "TARGET(需判断): [用户]: 其实最近上班有点累，不过想到周六就开心了")


def bench(device_str: str, ckpt: str, n_warm: int = 10, n: int = 200) -> None:
    import torch
    from net import VibratoNet

    device = torch.device(device_str)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]), max_len=ck["max_len"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    vocab = ck["vocab"]
    ids = torch.tensor([[vocab.get(ch, 1) for ch in TEXT][-ck["max_len"] + 1:]]).to(device)
    echo = torch.zeros(1, 3).to(device)

    for _ in range(n_warm):
        with torch.no_grad():
            net(ids, echo)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            out = net(ids, echo)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / n * 1000
    pad = [round(v, 3) for v in out["pad_norm"][0].tolist()]
    conf = round(out["conf"][0].item(), 3)
    print(f"[{device_str}] 单条前向: 平均 {dt_ms:.2f} ms × {n} 次 "
          f"(seq={ids.shape[1]}) | PAD={pad} conf={conf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2/vibrato.pt")
    ap.add_argument("--device", default=None, help="不填则 GPU+CPU 都测")
    args = ap.parse_args()
    if args.device:
        bench(args.device, args.ckpt)
    else:
        bench("cuda:0", args.ckpt)
        bench("cpu", args.ckpt)
        print("[教师对照] qwen3.5:4b-q8_0 @1080Ti 单次标注调用 ≈ 3000-5000 ms (think:false)")
