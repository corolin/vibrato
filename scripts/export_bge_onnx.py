# -*- coding: utf-8 -*-
"""export_bge_onnx — bge-m3 ONNX 导出(fp32) + int8 动态量化(向量层进 ONNX 考场)。

用法: PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge \
      python3 export_bge_onnx.py            # 导出+量化(~几分钟, 盘上需 ~5GB)
"""
from __future__ import annotations

import os
import sys

OUT = "bge_onnx"
MODEL_DIR = "/root/models/bge-m3"


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    class BgeWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state

    os.makedirs(OUT, exist_ok=True)
    fp32 = os.path.join(OUT, "bge_m3_fp32.onnx")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BgeWrap(AutoModel.from_pretrained(MODEL_DIR).eval())
    enc = tok("你好，世界。这是一个导出用例。", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    if not os.path.exists(fp32):
        torch.onnx.export(
            model, (ids, mask), fp32, dynamo=False, opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"},
                          "attention_mask": {0: "b", 1: "s"},
                          "last_hidden_state": {0: "b", 1: "s"}})
        print(f"fp32 导出: {fp32} {os.path.getsize(fp32)/1e6:.0f} MB")
    else:
        print(f"fp32 已存在: {fp32}")
    del model
    torch.cuda.empty_cache()

    int8 = os.path.join(OUT, "models/v6/bge_m3_int8.onnx")
    if not os.path.exists(int8):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        print(f"int8 量化: {int8} {os.path.getsize(int8)/1e6:.0f} MB")
    else:
        print(f"int8 已存在: {int8}")


if __name__ == "__main__":
    main()
