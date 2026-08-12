"""
憨憨 — AstrBot 人格插件（适配 AstrBot v4.x）

把 ex-skill 生成的前任人格（exes/hanhan）注入 AstrBot 的 LLM 请求，
使 bot 以憨憨的身份、记忆和说话方式回复。

- 人格提示词在同目录 persona_prompt.md，可自行编辑
- 默认对所有会话生效，可用 /憨憨开关 按会话切换
- @on_llm_request：把人格写入 req.system_prompt
- @on_decorating_result：把 LLM 结果按行拆成多条消息依次发送（一行 = 一条，
  模拟真实微信分条习惯）
- 表情包按情绪匹配：LLM 输出 [表情包:情绪词]（如 [表情包:开心]），插件按
  文件名关键词在 stickers/ 里匹配图片；无情绪词的 [表情包] 随机抽取
- 消息结尾的句号自动去除（硬保证，防止模型习惯性加句号）
- 注意：v4.x 钩子为装饰器注册制（@on_llm_request/@on_decorating_result），
  不再是旧版的类方法覆写；流式输出时装饰阶段不生效，建议关闭流式输出
"""

import random
import re
from pathlib import Path
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.event.filter import command, on_decorating_result, on_llm_request
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

# 注入标记：用于防止同一会话内重复注入
_PERSONA_MARK = "<!-- hanhan-persona -->"
# 表情包占位标记：[表情包] 或 [表情包:情绪词]（支持全角冒号）
_STICKER_RE = re.compile(r"\[表情包\s*[:：]?\s*([^\]]*)\]")
_STICKER_DIR = Path(__file__).parent / "stickers"
_STICKER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _strip_period(line: str) -> str:
    """去掉结尾的单句号；。。/。。。/.. 等省略式保留。"""
    if line.endswith("。") and not line.endswith("。。"):
        return line[:-1]
    if line.endswith(".") and not line.endswith(".."):
        return line[:-1]
    return line


class HanhanPersonaPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.config = config or {}
        self.persona_prompt = self._load_persona()
        # 会话粒度开关：session_id -> bool（True 为注入人格）
        self.enabled_sessions: dict[str, bool] = {}

    def _load_persona(self) -> str:
        prompt_file = Path(__file__).parent / "persona_prompt.md"
        try:
            text = prompt_file.read_text(encoding="utf-8")
            if text.strip():
                logger.info(f"[hanhan] 已加载人格提示词（{len(text)} 字符）")
                return text
        except FileNotFoundError:
            pass
        logger.warning(f"[hanhan] 未找到人格文件 {prompt_file}，插件将不注入人格")
        return ""

    def _session_id(self, event: AstrMessageEvent) -> str:
        """获取会话标识，兼容不同 AstrBot 版本。"""
        unified = getattr(event, "unified_msg_origin", None)
        if unified:
            return unified
        sender = event.message_obj.sender
        return f"{getattr(sender, 'platform', '')}:{getattr(sender, 'user_id', '')}"

    def _persona_enabled(self, event: AstrMessageEvent) -> bool:
        return self.enabled_sessions.get(self._session_id(event), True)

    # ---------- LLM 请求钩子：注入人格 ----------

    @on_llm_request()
    async def inject_persona(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """LLM 请求前注入人格（只修改 req，不能 yield）。"""
        if not self.persona_prompt or not self._persona_enabled(event):
            return
        if _PERSONA_MARK in req.system_prompt:
            return  # 已注入过，跳过

        req.system_prompt = f"{_PERSONA_MARK}\n{self.persona_prompt}\n\n{req.system_prompt}"

    # ---------- 结果装饰钩子：分条发送 + 情绪表情包 ----------

    @on_decorating_result()
    async def handle_result(self, event: AstrMessageEvent) -> None:
        """发送前处理 LLM 结果：按行拆消息、去句号、[表情包:情绪词] 换图片。"""
        if not self._persona_enabled(event):
            return
        result = event.get_result()
        if (
            result is None
            or not result.chain
            or not result.is_llm_result()
            or not all(isinstance(comp, Plain) for comp in result.chain)
        ):
            return  # 非 LLM 结果、流式结果或含富媒体组件时，交给默认流程

        text = "".join(comp.text for comp in result.chain)
        if not text.strip():
            return
        parts = self._parse_reply(text)
        if len(parts) <= 1 and parts and parts[0][0] == "text":
            # 单条纯文本：仅当去掉了结尾句号时才重写，否则交给默认流程
            payload = parts[0][1]
            if payload == text.strip():
                return
            event.clear_result()
            await event.send(MessageChain().message(payload))
            return

        # 清空默认结果，所有消息按序主动发送，保证分条顺序
        event.clear_result()
        for kind, payload in parts:
            if kind == "img":
                img = self._pick_sticker(payload)
                if img is None:
                    logger.warning("[hanhan] stickers/ 文件夹为空或不存在，跳过表情包")
                    continue
                await event.send(MessageChain().file_image(str(img)))
            else:
                await event.send(MessageChain().message(payload))

    def _parse_reply(self, text: str) -> list[tuple[str, Optional[str]]]:
        """把 LLM 回复解析为消息片段列表：("text", 内容) 或 ("img", 情绪词|None)。

        一行 = 一条消息；行内出现 [表情包] / [表情包:情绪词] 时拆成先后两条。
        文本片段自动去除结尾句号。
        """
        parts: list[tuple[str, Optional[str]]] = []
        for line in text.split("\n"):
            segs = re.split(_STICKER_RE, line)
            i = 0
            while i < len(segs):
                if segs[i].strip():
                    parts.append(("text", _strip_period(segs[i].strip())))
                if i + 1 < len(segs):
                    keyword = segs[i + 1].strip()
                    parts.append(("img", keyword or None))
                i += 2
        return parts

    def _pick_sticker(self, keyword: Optional[str] = None) -> Optional[Path]:
        """按情绪词匹配 stickers/ 里的图片（文件名包含关键词），匹配不到随机抽。

        例如 keyword="开心" 会优先选中 catbug-开心.webp；文件夹为空返回 None。
        """
        try:
            files = [
                p for p in _STICKER_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in _STICKER_EXTS
            ]
        except OSError:
            return None
        if not files:
            return None
        if keyword:
            matched = [p for p in files if keyword.lower() in p.stem.lower()]
            if matched:
                return random.choice(matched)
        return random.choice(files)

    # ---------- 命令 ----------

    @command("憨憨开关")
    async def toggle_persona(self, event: AstrMessageEvent):
        """/憨憨开关 —— 切换本会话是否启用憨憨人格。"""
        session_id = self._session_id(event)
        current = self.enabled_sessions.get(session_id, True)
        self.enabled_sessions[session_id] = not current
        state = "已开启" if not current else "已关闭"
        yield event.plain_result(f"憨憨人格{state}")

    @command("憨憨状态")
    async def persona_status(self, event: AstrMessageEvent):
        """/憨憨状态 —— 查看本会话人格开关状态。"""
        session_id = self._session_id(event)
        current = self.enabled_sessions.get(session_id, True)
        yield event.plain_result(f"憨憨人格：{'开启' if current else '关闭'}")
