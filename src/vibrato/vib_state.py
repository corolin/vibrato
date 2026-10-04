# -*- coding: utf-8 -*-
"""Vibrato 上下文文本构建 — 训练输入与 teacher 输入同源(逐字符一致)。

parity 纪律: net 看到的 char id 序列来自这个文本, teacher 判断的也是这个文本,
二者必须是同一个字符串(含 TARGET 标记), 由构造保证。
角色: [用户]=被分析的 them(情绪载体), [AI]=chordia。
"""

from __future__ import annotations

import re
from typing import Dict, List

MAX_CONTEXT_MSGS = 6      # 目标之前最多带 6 条
_CRLF = re.compile(r"\r\n?")
_BLANK = re.compile(r"\n{3,}")


def sanitize(text: str) -> str:
    return _BLANK.sub("\n\n", _CRLF.sub("\n", text)).strip()


def role_of(side: str) -> str:
    """user→[用户](情绪载体/them), ai→[AI](chordia/self)。"""
    return "[用户]" if side == "user" else "[AI]"


def build_context(messages: List[Dict[str, str]], target_idx: int,
                  max_context: int = MAX_CONTEXT_MSGS) -> str:
    """产出带 TARGET 标记的上下文文本(旧→新, 目标收尾)。

    messages: [{side: "user"|"ai", text}], target_idx 指向被标注的 user 消息。
    """
    if messages[target_idx]["side"] != "user":
        raise ValueError("target 必须是 user 侧消息(情绪载体)")
    start = max(0, target_idx - max_context)
    lines = []
    for i in range(start, target_idx + 1):
        m = messages[i]
        tag = "TARGET(需判断): " if i == target_idx else ""
        lines.append(f"{tag}{role_of(m['side'])}: {sanitize(m['text'])}")
    return "\n".join(lines)


def echo_from_prev(pad_norm_3: List[float]) -> List[float]:
    """上一条 user 消息的归一化 PAD → echo(截到[-1,1]; 无历史返回零向量)。"""
    if not pad_norm_3:
        return [0.0, 0.0, 0.0]
    return [max(-1.0, min(1.0, float(v))) for v in pad_norm_3[:3]]


if __name__ == "__main__":
    demo = [
        {"side": "ai", "text": "今天过得怎么样呀？"},
        {"side": "user", "text": "还行吧，就那样\r\n\n\n\n一堆破事"},
        {"side": "ai", "text": "听起来有点烦，想说说吗？"},
        {"side": "user", "text": "算了不说了，你陪我聊点别的"},
        {"side": "ai", "text": "好呀，那就聊点开心的～"},
        {"side": "user", "text": "嗯"},
    ]
    text = build_context(demo, 5)
    print(text)
    assert "TARGET(需判断): [用户]: 嗯" in text
    assert "一堆破事" in text and "\r" not in text and "\n\n\n" not in text
    assert echo_from_prev(None) == [0.0, 0.0, 0.0]
    assert echo_from_prev([0.5, -2, 1]) == [0.5, -1.0, 1.0]
    # 窗口裁剪: 目标(偶数idx=user)前只带 6 条
    long = [{"side": "user" if i % 2 == 0 else "ai", "text": f"m{i}"} for i in range(20)]
    ctx = build_context(long, 18)
    assert "m11" not in ctx and "m12" in ctx
    print("✓ vib_state 自检通过")
