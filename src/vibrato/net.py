# -*- coding: utf-8 -*-
"""Vibrato(颤音) 网络正身 v6 — PAD 特化 System One(单文件自包含)

  输入:  字id序列(B,L) + 状态伪token(state_pos/state_vals/state_mask, v1/affect)
         [+ 可选逐位特征 feats(B,L,feat_dim): bge-m3 冻结注入]
  正身:  CEncoder风格小编码器 — 字emb64(+状态swap) → proj → 3层transformer d192(手写MHA)
  头:    P/A/D 三维各 9档序数bin头(score题) + 情绪族(choice题,8路) + noul(是非题,5×2路)
         + 全局conf头(sigmoid)
  输出:  pad_bins(B,3,9) / pad_expected(B,3)∈[1,9] / pad_norm(B,3)∈[-1,1] / conf(B,)

v6 变更(对 v5/net_v5.py):
- 双路状态: echo 回归 cls 直连(echo_in, 锚定先验——实测同信息 cls 直连+0.28 vs
  序列内+0.06) + 消息范式状态伪token(vib_messages, 细粒度带时序), 两路并存
- 状态伪token通路: state_pos 位上 char_emb 被 state_in(cat(值×遮罩,遮罩)) 替换,
  缺席字段结构性为零(非哨兵); 窗口 192→512; vitality 收而不读(v6 未消化)
- 状态值由 vib_messages.state_vector 归一化(单位纪律: 归一化是模型私事)

血统(借自 pad-jev/prisma/prisma_v016_networks.py, 2026-09-28版):
- CEncoder 正身与手写MHA(规避 nn.MultiheadAttention reshape 导出成常量的坑)
- conf 纪律(EchoTagger conf_head): 低置信样本路由到 LLM teacher(Arbiter纪律)
- bin分布输出刻度来自 Laya qtype=score 思想 + chordia 18项李克特量表

参数预算: ~1.9M(实测 char 词表 ~3k: emb 3k×64 ≈0.2M; bge 通路扩 in_proj 至
1088×192≈0.21M; 512窗 pos_emb 0.1M) — CPU毫秒级, 1080Ti fp32 从容训练
(Pascal 无fp16算力, 勿开autocast)。知识容量住在冻结的 bge-m3(567M, 不计此处)。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pad_schema
from . import vib_messages
from .battery import FAMILY_IDS, N_FAMILY, N_NOUL, NOUL_IDS, battery_digest

STATE_DIM = vib_messages.STATE_DIM          # 5: [p,a,d,pressure,vitality]


class VibratoNet(nn.Module):
    """PAD 特化 System One 网络。刻度契约见 pad_schema.py, 消息范式见 vib_messages.py。"""

    def __init__(self, vocab_size: int = 21000, d_model: int = 192, nhead: int = 6,
                 enc_layers: int = 3, max_len: int = 512, feat_dim: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.char_emb = nn.Embedding(vocab_size, 64)
        self.feat_dim = feat_dim
        in_dim = 64 + feat_dim
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        # 状态伪token: [值×遮罩(5), 遮罩(5)] → 64, swap 进 char_emb 位
        self.state_in = nn.Linear(STATE_DIM * 2, 64)
        # echo 锚定: 上轮用户PAD直连cls(快通路)。序列内状态token是细通路(带时序),
        # 但1.4k battery行教不动"从序列里挖状态"——实测同信息cls直连+0.28 vs
        # 序列内+0.06。两路并存: echo管锚定先验, 状态token管逐轮上下文。
        self.echo_in = nn.Linear(3, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        # 头: 三维各9档(score题) + 情绪族(choice题) + 是非(noul题, 2路softmax: [假,真])
        self.bin_heads = nn.ModuleList(
            [nn.Linear(d_model, pad_schema.N_BINS) for _ in pad_schema.DIMS])
        self.family_head = nn.Linear(d_model, N_FAMILY)
        self.noul_heads = nn.ModuleList([nn.Linear(d_model, 2) for _ in NOUL_IDS])
        self.conf_head = nn.Linear(d_model, 1)

    # ── prisma CEncoder._mha 同式(ONNX导出安全), 扩了padding mask ──
    @staticmethod
    def _mha(m: nn.MultiheadAttention, x: torch.Tensor,
             key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, E = x.shape
        H, hd = m.num_heads, E // m.num_heads
        qkv = F.linear(x, m.in_proj_weight, m.in_proj_bias)
        q, k, v = qkv.chunk(3, -1)
        q = q.reshape(B, L, H, hd).transpose(1, 2)
        k = k.reshape(B, L, H, hd).transpose(1, 2)
        v = v.reshape(B, L, H, hd).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / (hd ** 0.5)
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = torch.softmax(att, -1)
        return m.out_proj((att @ v).transpose(1, 2).reshape(B, L, E))

    def _encode(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 手写层循环(Prisma 同式): Post-Norm 语义 Norm(x+Sublayer(x)), 不走 layer()。
        # __init__ 的 norm_first=False 与此对齐——旗标本身不被读取, 只为语义诚实。
        for layer in self.encoder.layers:
            x = layer.norm1(x + self._mha(layer.self_attn, x, key_padding_mask))
            x = layer.norm2(x + layer.linear2(F.relu(layer.linear1(x))))
        if getattr(self.encoder, "norm", None) is not None:
            x = self.encoder.norm(x)
        return x

    def forward(self, ids: torch.Tensor,
                state_pos: Optional[torch.Tensor] = None,
                state_vals: Optional[torch.Tensor] = None,
                state_mask: Optional[torch.Tensor] = None,
                echo_prev: Optional[torch.Tensor] = None,
                feats: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """ids(B,L) 字id(状态位可为任意值, 会被替换); state_pos(B,L)布尔标记伪token位;
        state_vals/state_mask(B,L,5) 归一化值+字段遮罩(vib_messages.state_vector);
        echo_prev(B,3) 上轮用户归一化PAD(cls锚定, 缺省零); feats(B,L,feat_dim) 可选
        逐位特征(bge, 状态位置零); lengths(B,) 真实长度。"""
        B, L = ids.shape
        if L > self.max_len:
            raise ValueError(f"序列长 {L} 超过 max_len={self.max_len}, 调用方应先截断")
        emb = self.char_emb(ids)                                   # (B,L,64)
        if state_pos is not None:
            if state_vals is None or state_mask is None:
                raise ValueError("state_pos 需要同时提供 state_vals/state_mask")
            if state_vals.shape != (B, L, STATE_DIM):
                raise ValueError(f"state_vals 形状应为 ({B},{L},{STATE_DIM})")
            st = self.state_in(torch.cat([state_vals * state_mask, state_mask], -1))
            emb = torch.where(state_pos.unsqueeze(-1), st, emb)    # 伪token位整体swap
        parts = [emb]
        if self.feat_dim > 0:
            if feats is None:
                raise ValueError("feat_dim>0 需要传 feats")
            parts.append(feats)
        x = self.in_proj(torch.cat(parts, -1))
        x = x + self.pos_emb.weight[:L].unsqueeze(0)
        # cls = 可学习查询 + echo锚定(无echo时零投影, 与旧调用兼容)
        if echo_prev is None:
            echo_prev = torch.zeros(B, 3, dtype=x.dtype, device=x.device)
        elif echo_prev.shape != (B, 3):
            raise ValueError(f"echo_prev 形状应为 ({B}, 3)")
        cls_tok = self.cls + self.echo_in(echo_prev).unsqueeze(1)
        x = torch.cat([cls_tok.expand(B, 1, -1), x], dim=1)
        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = (torch.arange(L, device=ids.device)[None, :]
                                >= lengths[:, None])
            key_padding_mask = F.pad(key_padding_mask, (1, 0), value=False)  # cls 恒有效
        enc = self._encode(x, key_padding_mask)
        h = enc[:, 0]                                              # (B, d) cls 状态
        bins = torch.stack([F.softmax(head(h), -1) for head in self.bin_heads], dim=1)
        centers = torch.tensor(pad_schema.BIN_CENTERS, dtype=h.dtype, device=h.device)
        expected = (bins * centers).sum(-1)                        # (B,3) ∈[1,9]
        pad_norm = ((expected - pad_schema.LIKERT_MID) / pad_schema.LIKERT_HALF).clamp(-1, 1)
        family = F.softmax(self.family_head(h), -1)                # (B,8)
        nouls = torch.stack([F.softmax(nh(h), -1) for nh in self.noul_heads], dim=1)  # (B,5,2)
        conf = torch.sigmoid(self.conf_head(h)).squeeze(-1)
        return {"pad_bins": bins, "pad_expected": expected,
                "pad_norm": pad_norm, "family_probs": family,
                "noul_probs": nouls, "conf": conf}


def collate_pad(rows, pad_id: int = 0):
    """训练批组装: [{ids, state_pos?, state_vals?, state_mask?, pad_bins, family?, noul?}, ...]
    → 张量批。状态三件套可选(整批统一: 有任一行带状态则全体产出张量, 无状态行全零)。

    混合数据可选键: 全批都有才产出对应监督(legacy 60k 行有 family 无 noul/conf;
    battery 行三样全有)。family → (B,) 索引; noul → (B,5) 布尔; 缺失为 None。
    """
    B = len(rows)
    L = max(len(r["ids"]) for r in rows)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    has_state = any(r.get("state_pos") and any(r["state_pos"]) for r in rows)
    state_pos = torch.zeros(B, L, dtype=torch.bool) if has_state else None
    state_vals = torch.zeros(B, L, STATE_DIM) if has_state else None
    state_mask = torch.zeros(B, L, STATE_DIM) if has_state else None
    target = torch.zeros(B, len(pad_schema.DIMS), pad_schema.N_BINS)
    echo = torch.zeros(B, 3)
    has_family = all("family" in r for r in rows)
    has_noul = all("noul" in r for r in rows)
    family = torch.zeros(B, dtype=torch.long) if has_family else None
    noul = torch.zeros(B, N_NOUL, dtype=torch.float32) if has_noul else None
    for i, r in enumerate(rows):
        n = len(r["ids"])
        ids[i, :n] = torch.as_tensor(r["ids"], dtype=torch.long)
        lengths[i] = n
        if r.get("echo"):
            echo[i] = torch.as_tensor(r["echo"][:3], dtype=torch.float32)
        if has_state and r.get("state_pos") and any(r["state_pos"]):
            sp = torch.as_tensor(r["state_pos"], dtype=torch.bool)
            state_pos[i, :n] = sp
            state_vals[i, :n] = torch.as_tensor(r["state_vals"], dtype=torch.float32)
            state_mask[i, :n] = torch.as_tensor(r["state_mask"], dtype=torch.float32)
        target[i] = torch.as_tensor([r["pad_bins"][d] for d in pad_schema.DIMS],
                                    dtype=torch.float32)
        if has_family:
            family[i] = FAMILY_IDS.index(r["family"])
        if has_noul:
            noul[i] = torch.as_tensor([1.0 if r["noul"][q] else 0.0 for q in NOUL_IDS])
    return {"ids": ids, "lengths": lengths, "target": target, "echo": echo,
            "state_pos": state_pos, "state_vals": state_vals, "state_mask": state_mask,
            "family": family, "noul": noul}


if __name__ == "__main__":
    torch.manual_seed(0)
    V, L, B = 21000, 300, 4

    net = VibratoNet(vocab_size=V)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"VibratoNet v6: {n_params:,} params (d{net.d_model}, 3层, 词表{V}, 窗口{net.max_len})")

    ids = torch.randint(0, V, (B, L))
    lengths = torch.tensor([L, L - 60, L - 120, 40])
    # 状态伪token: 行0 双token(行17位满血含vitality位), 行2 单token, 行1/3 无状态
    state_pos = torch.zeros(B, L, dtype=torch.bool)
    state_pos[0, 5] = True
    state_pos[0, 17] = True
    state_pos[2, 5] = True
    state_vals = torch.zeros(B, L, STATE_DIM)
    state_mask = torch.zeros(B, L, STATE_DIM)
    for b in range(B):
        state_vals[b, 5] = torch.tensor([0.4, -0.2, 0.1, 0.12, 0.0])
        state_mask[b, 5] = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    state_vals[0, 17] = torch.tensor([-0.1, 0.2, 0.0, 0.185, 0.0])
    state_mask[0, 17] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])   # 满血(含vitality位)

    out = net(ids, state_pos, state_vals, state_mask, lengths=lengths)
    assert out["pad_bins"].shape == (B, 3, pad_schema.N_BINS)
    assert torch.allclose(out["pad_bins"].sum(-1), torch.ones(B, 3), atol=1e-5)
    assert (out["pad_expected"] >= 1).all() and (out["pad_expected"] <= 9).all()
    assert (out["pad_norm"] >= -1).all() and (out["pad_norm"] <= 1).all()
    assert out["family_probs"].shape == (B, N_FAMILY)
    assert torch.allclose(out["family_probs"].sum(-1), torch.ones(B), atol=1e-5)
    assert out["noul_probs"].shape == (B, N_NOUL, 2)
    assert torch.allclose(out["noul_probs"].sum(-1), torch.ones(B, N_NOUL), atol=1e-5)
    assert (out["conf"] >= 0).all() and (out["conf"] <= 1).all()
    print(f"前向: bins{tuple(out['pad_bins'].shape)} family{tuple(out['family_probs'].shape)} "
          f"noul{tuple(out['noul_probs'].shape)} conf{tuple(out['conf'].shape)} "
          f"PAD0={out['pad_norm'][0].tolist()} conf0={out['conf'][0].item():.3f}")

    # 状态伪token确实影响输出(旧 echo 自检的消息范式版)
    out_no_state = net(ids, lengths=lengths)
    delta = (out["pad_norm"] - out_no_state["pad_norm"]).abs().max().item()
    assert delta > 1e-4, "状态伪token输入未生效"
    # echo 锚定(cls直连)生效检查
    echo = torch.tensor([[0.8, -0.7, 0.5]] * B)
    out_echo = net(ids, state_pos, state_vals, state_mask, echo_prev=echo, lengths=lengths)
    delta_echo = (out_echo["pad_norm"] - out["pad_norm"]).abs().max().item()
    assert delta_echo > 1e-4, "echo cls锚定未生效"
    print(f"状态伪token生效 OK (PAD 最大差 {delta:.4f}) | "
          f"echo锚定生效 OK (差 {delta_echo:.4f})")

    # 训练批组装 + 一步反向: score软标签CE + choice CE + noul CE + 占位conf BCE
    rows = []
    for b in range(B):
        row = {"ids": ids[b, :lengths[b]].tolist(),
               "echo": [0.8, -0.7, 0.5],
               "pad_bins": {d: out["pad_bins"][b, i].detach().tolist()
                            for i, d in enumerate(pad_schema.DIMS)},
               "family": FAMILY_IDS[out["family_probs"][b].argmax().item()],
               "noul": {q: bool(out["noul_probs"][b, j, 1] > 0.5)
                        for j, q in enumerate(NOUL_IDS)}}
        if b < 3:   # 混合批: 2 行带状态, 1 行全False列表(无状态但键在), 1 行纯文字
            row["state_pos"] = state_pos[b, :lengths[b]].tolist()
            if any(row["state_pos"]):
                row["state_vals"] = state_vals[b, :lengths[b]].tolist()
                row["state_mask"] = state_mask[b, :lengths[b]].tolist()
        rows.append(row)
    batch = collate_pad(rows)
    assert batch["state_pos"] is not None and not batch["state_pos"][3].any()
    assert not batch["state_pos"][1].any()          # 全False列表行(键在, 无状态)不炸
    assert torch.allclose(batch["echo"], torch.tensor([0.8, -0.7, 0.5]).expand(B, 3))
    pred = net(batch["ids"], batch["state_pos"], batch["state_vals"],
               batch["state_mask"], batch["echo"], lengths=batch["lengths"])
    # conf 的真实监督信号(escalation 标签)由 train.py 定义, 冒烟用占位
    loss_ce = -(batch["target"] * pred["pad_bins"].clamp_min(1e-9).log()).sum(-1).mean()
    loss_family = F.nll_loss(pred["family_probs"].clamp_min(1e-9).log(), batch["family"])
    noul_target = torch.stack([1 - batch["noul"], batch["noul"]], -1)   # (B,5,2)
    loss_noul = -(noul_target * pred["noul_probs"].clamp_min(1e-9).log()).sum(-1).mean()
    conf_target = torch.full_like(pred["conf"], 0.5)
    loss_conf = F.binary_cross_entropy(pred["conf"], conf_target)
    loss = loss_ce + loss_family + loss_noul + loss_conf
    loss.backward()
    # 未用到的 embedding 行天然无梯度, 点名检查关键路径(含 状态/choice/noul/conf 通路)
    keys = ["state_in.weight", "echo_in.weight", "in_proj.weight", "bin_heads.0.weight",
            "bin_heads.2.weight", "family_head.weight", "noul_heads.0.weight",
            "noul_heads.4.weight", "conf_head.weight", "char_emb.weight"]
    named = dict(net.named_parameters())
    grad_ok = all(named[k].grad is not None and named[k].grad.abs().sum() > 0 for k in keys)
    print(f"collate+反向 OK(混合批 3状态+1纯文字), ce={loss_ce.item():.4f} "
          f"family={loss_family.item():.4f} noul={loss_noul.item():.4f} "
          f"conf={loss_conf.item():.4f}, 关键参数梯度流通={grad_ok}")
    print("✓ VibratoNet v6 自检全部通过 | 刻度指纹", pad_schema.contract_digest()[:16],
          "| 题库指纹", battery_digest()[:16],
          "| affect格式", vib_messages.schema_digest()[:16])
