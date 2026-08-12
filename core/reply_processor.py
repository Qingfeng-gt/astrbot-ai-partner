"""LLM 回复解析：分条发送 + 表情包标记 + 去句号。

纯函数模块，可独立测试：输入 LLM 原始文本，输出消息片段列表。
"""

import re
from typing import Optional

# 表情包占位标记：[表情包] 或 [表情包:情绪词]（支持全角冒号）
STICKER_RE = re.compile(r"\[表情包\s*[:：]?\s*([^\]]*)\]")


def strip_period(line: str) -> str:
    """去掉结尾的单句号；。。/。。。/.. 等省略式保留。"""
    if line.endswith("。") and not line.endswith("。。"):
        return line[:-1]
    if line.endswith(".") and not line.endswith(".."):
        return line[:-1]
    return line


def parse_reply(text: str) -> list[tuple[str, Optional[str]]]:
    """把 LLM 回复解析为消息片段列表：("text", 内容) 或 ("img", 情绪词|None)。

    一行 = 一条消息；行内出现 [表情包] / [表情包:情绪词] 时拆成先后两条。
    文本片段自动去除结尾句号。
    """
    parts: list[tuple[str, Optional[str]]] = []
    for line in text.split("\n"):
        segs = re.split(STICKER_RE, line)
        i = 0
        while i < len(segs):
            if segs[i].strip():
                parts.append(("text", strip_period(segs[i].strip())))
            if i + 1 < len(segs):
                keyword = segs[i + 1].strip()
                parts.append(("img", keyword or None))
            i += 2
    return parts
