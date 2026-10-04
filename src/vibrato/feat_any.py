# -*- coding: utf-8 -*-
"""feat_any — 任意 HF 嵌入模型的逐字符特征( feat_bge 的泛化版)。

与 feat_bge.BgeFeaturizer 同接口: char_feats(text) → (len(text), dim)。
offset 对齐、未覆盖字符零向量、状态位置零特征——口径与 bge 管线一致,
保证换底座实验里唯一变量是嵌入模型本身。
"""
from __future__ import annotations

import os
from typing import Optional

import torch


class AnyFeaturizer:
    """逐字符特征提取器(冻结)。dim 由模型决定(384/768/1024)。"""

    def __init__(self, model_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dim = self.model.config.hidden_size
        self.last_coverage = 1.0

    @torch.no_grad()
    def char_feats(self, text: str, max_chars: int = 512) -> torch.Tensor:
        text = text[:max_chars]
        if not text:
            return torch.zeros(0, self.dim)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False,
                       truncation=True, max_length=1024, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state[0].float()  # (T, dim)
        feats = torch.zeros(len(text), self.dim,
                            dtype=hidden.dtype, device=hidden.device)
        span = torch.zeros(len(text), dtype=torch.bool, device=hidden.device)
        for t, (s, e) in enumerate(offsets):
            if e > s and s < len(text):
                feats[s:min(e, len(text))] = hidden[t]
                span[s:min(e, len(text))] = True
        self.last_coverage = span.float().mean().item()
        return feats
