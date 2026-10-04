# -*- coding: utf-8 -*-
"""export_onnx_v6 — v6 判头 ONNX 导出 + int8 动态量化 + 三档延迟实测。

全输入签名(ids/state_pos/state_vals/state_mask/echo/feats, L 动态)——
无 bge 的部署喂零 feats(bge-dropout 15% 训练保出的在分布路径)。
用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-ort \
      python3 export_onnx_v6.py --ckpt ckpt_v6o/vibrato.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from vibrato import pad_schema
from vibrato import vib_messages
from vibrato.net import VibratoNet


class Wrap(torch.nn.Module):
    """ONNX 包装: 展平 forward, 元组输出, bool→float 适配。"""

    def __init__(self, net: VibratoNet):
        super().__init__()
        self.net = net

    def forward(self, ids, state_pos_f, state_vals, state_mask, echo_prev, feats):
        out = self.net(ids, state_pos_f.to(torch.bool), state_vals, state_mask,
                       echo_prev, feats, lengths=None)
        return (out["pad_bins"], out["pad_expected"], out["pad_norm"],
                out["family_probs"], out["noul_probs"], out["conf"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_v6o/vibrato.pt")
    ap.add_argument("--out", default="model_v6")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pad_schema.verify_ckpt(ck, hard=True)
    net = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                     feat_dim=ck["feat_dim"])
    net.load_state_dict(ck["model"])
    net.eval()
    w = Wrap(net)

    os.makedirs(args.out, exist_ok=True)
    B, L = 1, 96
    demo = (torch.randint(2, 3000, (B, L)),
            torch.zeros(B, L),                      # state_pos(float 0/1)
            torch.zeros(B, L, vib_messages.STATE_DIM),
            torch.zeros(B, L, vib_messages.STATE_DIM),
            torch.zeros(B, 3),
            torch.zeros(B, L, ck["feat_dim"]))
    fp32 = os.path.join(args.out, "models/v6/vibrato_v6_fp32.onnx")
    torch.onnx.export(
        w, demo, fp32, dynamo=False,
        input_names=["ids", "state_pos", "state_vals", "state_mask", "echo_prev", "feats"],
        output_names=["pad_bins", "pad_expected", "pad_norm", "family_probs",
                      "noul_probs", "conf"],
        dynamic_axes={n: {0: "batch", 1: "seq"} for n in
                      ["ids", "state_pos", "state_vals", "state_mask", "feats"]})
    print(f"fp32 导出: {fp32} {os.path.getsize(fp32)//1024} KB")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = os.path.join(args.out, "models/v6/vibrato_v6_int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"int8 量化: {int8} {os.path.getsize(int8)//1024} KB")

    # 等价性冒烟 + 三档延迟
    import onnxruntime as ort
    with torch.no_grad():
        ref = net(demo[0], demo[1].to(torch.bool), demo[2], demo[3], demo[4], demo[5])
    for path, so in ((fp32, ort.SessionOptions()), (int8, ort.SessionOptions())):
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {n: v.numpy() for n, v in
                               zip(["ids", "state_pos", "state_vals", "state_mask",
                                    "echo_prev", "feats"], demo)})
        err = float(np.abs(outs[2] - ref["pad_norm"].numpy()).max())
        print(f"{os.path.basename(path)}: PAD 最大偏差 {err:.4f}")
        ts = []
        for _ in range(220):
            t0 = time.perf_counter()
            sess.run(None, {n: v.numpy() for n, v in
                           zip(["ids", "state_pos", "state_vals", "state_mask",
                                "echo_prev", "feats"], demo)})
            ts.append(time.perf_counter() - t0)
        ts = ts[20:]
        print(f"  CPU 单线程 {len(ts)} 次: 中位 {sorted(ts)[len(ts)//2]*1000:.2f} ms "
              f"均值 {sum(ts)/len(ts)*1000:.2f} ms  (L={L})")


if __name__ == "__main__":
    main()
