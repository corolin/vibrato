# -*- coding: utf-8 -*-
"""feat_bge — bge-m3 冻结逐token知识注入(v6 piece ②, 单文件自包含)

v6 的知识答案: 61k legacy 教不会"撒娇=亲昵正向"这类语义(词崩塌/极性反转的根因),
外挂一个冻结的 bge-m3(BAAI, XLM-R 底座, 1024维, 中文强)做逐字符语义特征,
与 char_emb 并联进 in_proj —— Vibrato 出判读语义, bge 出知识, 各司其职。

纪律:
- 冻结: requires_grad 全 False, eval 模式, no_grad 前向(它不是被训练对象)
- 对齐: bge tokenizer 的 offset_mapping(字符span)把 token 向量摊到字符位;
  一个 span 内所有字符继承该 token 向量, 未覆盖字符(罕见)置零
- 位置: 状态伪token位由调用方置零(它们没有文字); 本模块只管纯文本
- 容量: feats 不落盘(1024×512×fp16≈0.5MB/行, 全量存不起), 训练侧 on-the-fly
  + RAM LRU(见 train.py v6); 逐位对齐拼装与 char id 构建在同一循环完成
  (text单元按字符数推进, state伪token占1位且特征置零)

用法:
    PYTHONPATH=/root/vibrato-torch-lib:/root/vibrato-bge python3 feat_bge.py
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

BGE_DIR = os.environ.get("VIB_BGE_DIR", "/root/models/bge-m3")
FEAT_DIM = 1024
MAX_CHARS = 512            # 与 vib_messages.MAX_WINDOW 对齐(截断由调用方做过)


class BgeFeaturizer:
    """冻结 bge-m3 → 逐字符 1024 维特征。线程不安全(单进程数据准备/训练用)。"""

    def __init__(self, model_dir: str = BGE_DIR, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModel, AutoTokenizer   # 延迟导入(仅本模块需要)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def char_feats(self, text: str, max_chars: int = MAX_CHARS) -> torch.Tensor:
        """纯文本 → (min(len(text),max_chars), 1024) fp32。超长截尾(窗口预算已在上游做过)。"""
        text = text[:max_chars]
        if not text:
            return torch.zeros(0, FEAT_DIM)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False,
                       truncation=True, max_length=1024, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()      # [(char_start, char_end), ...]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state[0]      # (T, 1024)
        feats = torch.zeros(len(text), FEAT_DIM,
                            dtype=hidden.dtype, device=hidden.device)
        span = torch.zeros(len(text), dtype=torch.bool, device=hidden.device)
        for t, (s, e) in enumerate(offsets):
            if e > s and s < len(text):                      # 跳过零宽/越界 span
                feats[s:min(e, len(text))] = hidden[t]
                span[s:min(e, len(text))] = True
        # 未覆盖字符保留零向量(覆盖率自查见 self.coverage)
        self.last_coverage = span.float().mean().item()
        return feats

    @torch.no_grad()
    def batch_char_feats(self, texts: List[str], max_chars: int = MAX_CHARS
                         ) -> List[torch.Tensor]:
        """批量便利封装(逐条前向; bge-m3 560M, 批内 padding 收益留给调用方自己拼)。"""
        return [self.char_feats(t, max_chars) for t in texts]


# ── 自检 ──
if __name__ == "__main__":
    import time

    fe = BgeFeaturizer()
    print(f"device={fe.device}, bge params={sum(p.numel() for p in fe.model.parameters()):,}")

    t1 = "嘿嘿，那你要当那个最暖和的抱枕嘛？🥺"
    t2 = "算了，没什么好说的，随便吧。"
    f1, f2 = fe.char_feats(t1), fe.char_feats(t2)
    assert f1.shape == (len(t1), FEAT_DIM) and f2.shape == (len(t2), FEAT_DIM)
    assert fe.last_coverage > 0.95, f"字符覆盖率过低: {fe.last_coverage:.3f}"
    print(f"形状对齐 OK (覆盖率 {fe.last_coverage:.3f}, 含emoji位)")

    # 语义健全性: 同文本相邻句向量相似度 > 跨文本(亲昵句 vs 冷漠句应可分)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
    same = cos(f1[3:9].mean(0), f1[10:16].mean(0)).item()
    cross = cos(f1[3:9].mean(0), f2[4:10].mean(0)).item()
    print(f"语义健全性: 同文相似 {same:.3f} vs 跨文 {cross:.3f} (同文应占优)")

    # 吞吐: 训练侧 on-the-fly 可行性
    batch_texts = [t1 + t2] * 32
    t0 = time.time()
    for t in batch_texts:
        fe.char_feats(t)
    dt = (time.time() - t0) / len(batch_texts) * 1000
    print(f"吞吐: {dt:.1f} ms/条(约{len(t1+t2)}字符) — batch64 一步约 {dt*64/1000:.2f}s")
    print("✓ feat_bge 自检通过")
