# -*- coding: utf-8 -*-
"""export_fused — 融合单文件 ONNX: bge 前向 + 字符对齐 + 判头, 一张图一次 run。

输入契约: 原六项(ids/状态三件套/echo) + bge_ids/bge_mask [Nseg,T] + 
         char_seg/char_tok [B,L](每字符位归属的段号/token号, -1=零特征位)。
分词(tokenizer.json)与字符词表留在图外——它们是数据不是模型。
产出: fused_onnx/vibrato_fused_fp32.onnx(>2GB→主图+外部张量) 与
     vibrato_fused_int8.onnx(单文件 ~571MB, 整图量化)。
用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge:/root/vibrato-ort \
      python3 export_fused.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from vibrato import pad_schema
from vibrato import vib_messages
from vibrato.net import VibratoNet

BGE_DIR = "/root/models/bge-m3"
OUT = "fused_onnx"


class VibratoFused(torch.nn.Module):
    """bge(Nseg 批) → 展平 → 按字符映射 gather → 判头。一次前向出全部头。"""

    def __init__(self, head: VibratoNet, bge):
        super().__init__()
        self.head = head
        self.bge = bge

    def forward(self, ids, state_pos_f, state_vals, state_mask, echo_prev,
                bge_ids, bge_mask, char_seg, char_tok):
        h = self.bge(input_ids=bge_ids,
                     attention_mask=bge_mask).last_hidden_state      # [Nseg,T,1024]
        T = bge_ids.shape[1]
        flat = h.reshape(-1, h.shape[-1])                            # [Nseg*T,1024]
        idx = torch.clamp(char_seg, min=0) * T + torch.clamp(char_tok, min=0)
        f = flat[idx]                                                # [B,L,1024]
        feats = torch.where((char_seg < 0).unsqueeze(-1),
                            torch.zeros_like(f), f)
        out = self.head(ids, state_pos_f.to(torch.bool), state_vals,
                        state_mask, echo_prev, feats, lengths=None)
        return (out["pad_bins"], out["pad_expected"], out["pad_norm"],
                out["family_probs"], out["noul_probs"], out["conf"])


def main():
    from transformers import AutoModel

    ck = torch.load("checkpoints/ckpt_v6o/vibrato.pt", map_location="cpu", weights_only=False)
    pad_schema.verify_ckpt(ck, hard=True)
    head = VibratoNet(vocab_size=len(ck["vocab"]) + 1, max_len=ck["max_len"],
                      feat_dim=ck["feat_dim"])
    head.load_state_dict(ck["model"])
    head.eval()
    bge = AutoModel.from_pretrained(BGE_DIR).eval()
    fused = VibratoFused(head, bge)

    os.makedirs(OUT, exist_ok=True)
    B, L, Nseg, T = 1, 48, 2, 16
    seg_row = [-1] * 8 + [0] * 20 + [-1] * 4 + [1] * 16
    tok_row = [0] * 8 + [3] * 20 + [0] * 4 + [5] * 16
    demo = (torch.randint(2, 3000, (B, L)),
            torch.zeros(B, L), torch.zeros(B, L, vib_messages.STATE_DIM),
            torch.zeros(B, L, vib_messages.STATE_DIM), torch.zeros(B, 3),
            torch.randint(0, 1000, (Nseg, T)), torch.ones(Nseg, T, dtype=torch.long),
            torch.tensor([seg_row]), torch.tensor([tok_row]))
    fp32 = os.path.join(OUT, "models/v6/vibrato_fused_fp32.onnx")
    if not os.path.exists(fp32):
        torch.onnx.export(
            fused, demo, fp32, dynamo=False, opset_version=17,
            input_names=["ids", "state_pos", "state_vals", "state_mask", "echo_prev",
                         "bge_ids", "bge_mask", "char_seg", "char_tok"],
            output_names=["pad_bins", "pad_expected", "pad_norm", "family_probs",
                          "noul_probs", "conf"],
            dynamic_axes={"ids": {0: "b", 1: "l"}, "state_pos": {0: "b", 1: "l"},
                          "state_vals": {0: "b", 1: "l"}, "state_mask": {0: "b", 1: "l"},
                          "bge_ids": {0: "n", 1: "t"}, "bge_mask": {0: "n", 1: "t"},
                          "char_seg": {0: "b", 1: "l"}, "char_tok": {0: "b", 1: "l"}})
    print(f"fp32 融合: {fp32} 主图 {os.path.getsize(fp32)/1e6:.1f} MB (外部张量另计)")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = os.path.join(OUT, "models/v6/vibrato_fused_int8.onnx")
    if not os.path.exists(int8):
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"int8 融合: {int8} {os.path.getsize(int8)/1e6:.0f} MB 单文件")


if __name__ == "__main__":
    main()
