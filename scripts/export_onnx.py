# -*- coding: utf-8 -*-
"""ONNX 导出 + INT8 动态量化 + CPU 实测 — Vibrato 部署链第一段。

  python export_onnx.py --ckpt ckpt_v2/vibrato.pt          # 导出 fp32 + int8
  python export_onnx.py --ckpt ckpt_v2/vibrato.pt --bench  # 加跑 CPU 基准
依赖: onnxruntime(pip install --target /root/vibrato-ort onnxruntime)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def export(ckpt_path: str, out_fp32: str, out_int8: str) -> None:
    import torch
    from net import VibratoNet
    from torch import nn

    class OnnxWrap(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, ids, echo, lengths):
            out = self.net(ids, echo, lengths=lengths)
            return (out["pad_bins"], out["pad_norm"],
                    out["family_probs"], out["noul_probs"], out["conf"])

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = VibratoNet(vocab_size=len(ck["vocab"]), max_len=ck["max_len"])
    net.load_state_dict(ck["model"])
    net.eval()
    wrap = OnnxWrap(net)

    vocab = ck["vocab"]
    text = ("TARGET(需判断): [用户]: 其实最近上班有点累，不过想到周六就开心了"
            "[AI]: 听起来今天是幸运日呀～ TARGET(需判断): [用户]: 哈哈是吧")
    ids = torch.tensor([[vocab.get(ch, 1) for ch in text][-ck["max_len"] + 1:]],
                       dtype=torch.long)
    echo = torch.zeros(1, 3)
    lengths = torch.tensor([ids.shape[1]], dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrap, (ids, echo, lengths), out_fp32,
            input_names=["ids", "echo", "lengths"],
            output_names=["pad_bins", "pad_norm", "family_probs", "noul_probs", "conf"],
            dynamic_axes={"ids": {0: "batch", 1: "seq"},
                          "echo": {0: "batch"}, "lengths": {0: "batch"},
                          "pad_bins": {0: "batch"}, "pad_norm": {0: "batch"},
                          "family_probs": {0: "batch"}, "noul_probs": {0: "batch"},
                          "conf": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"fp32 导出: {out_fp32} ({os.path.getsize(out_fp32)//1024} KB)")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(out_fp32, out_int8, weight_type=QuantType.QInt8)
    print(f"int8 量化: {out_int8} ({os.path.getsize(out_int8)//1024} KB)")


def bench(model_path: str, ckpt_path: str, n_warm: int = 10, n: int = 300) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    text = ("TARGET(需判断): [用户]: 其实最近上班有点累，不过想到周六就开心了"
            "[AI]: 听起来今天是幸运日呀～ TARGET(需判断): [用户]: 哈哈是吧")
    ids = np.array([[vocab.get(ch, 1) for ch in text][-ck["max_len"] + 1:]], dtype=np.int64)
    echo = np.zeros((1, 3), dtype=np.float32)
    lengths = np.array([ids.shape[1]], dtype=np.int64)

    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        feed = {"ids": ids, "echo": echo, "lengths": lengths}
        for _ in range(n_warm):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(n):
            out = sess.run(None, feed)
        dt = (time.perf_counter() - t0) / n * 1000
        tag = os.path.basename(model_path)
        print(f"[CPU {threads}线程] {tag}: 平均 {dt:.2f} ms × {n} "
              f"| PAD={[round(v,3) for v in out[1][0].tolist()]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2/vibrato.pt")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    fp32 = args.ckpt.rsplit("/", 1)[0] + "/vibrato_fp32.onnx"
    int8 = args.ckpt.rsplit("/", 1)[0] + "/vibrato_int8.onnx"
    export(args.ckpt, fp32, int8)
    if args.bench:
        bench(fp32, args.ckpt)
        bench(int8, args.ckpt)
        print("[基线] PyTorch eager CPU = 124.18 ms")
