"""
憨憨 — AstrBot 人格插件（适配 AstrBot v4.x）

把 ex-skill 生成的前任人格（exes/hanhan）注入 AstrBot 的 LLM 请求，
使 bot 以憨憨的身份、记忆和说话方式回复。

模块结构：
- main.py            编排层：Star 类、钩子、命令（本文件）
- persona_loader.py  人格提示词加载
- reply_processor.py LLM 回复解析（分条/去句号/表情包标记）
- sticker_bot.py     表情包选择（情绪匹配/防重复/限频）
- memory_engine.py   情景感知（时间间隔/话题突变/遗忘/忙碌）

行为特性：
- @on_llm_request：注入人格 + 情景感知（间隔、话题突变、遗忘提示）+ 上下文截断
- @on_decorating_result：按行拆成多条消息依次发送；[表情包:情绪词] 换图片；
  消息结尾单句号强制去除；每轮最多 1 张表情包且有限频
- 注意：v4.x 钩子为装饰器注册制；流式输出时装饰阶段不生效，建议关闭流式输出
"""

from pathlib import Path
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.event.filter import command, on_decorating_result, on_llm_request
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .memory_engine import MemoryEngine
from .persona_loader import PersonaLoader
from .reply_processor import parse_reply
from .sticker_bot import StickerBot

# 注入标记：用于防止重复注入
_PERSONA_MARK = "<!-- hanhan-persona -->"
_SITUATION_MARK = "<!-- hanhan-situation -->"
_PLUGIN_DIR = Path(__file__).parent


class HanhanPersonaPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.config = config or {}
        self.persona = PersonaLoader(_PLUGIN_DIR / "persona_prompt.md")
        self.persona_text = self.persona.load()
        self.memory = MemoryEngine()
        self.stickers = StickerBot(_PLUGIN_DIR / "stickers")
        # 会话粒度开关：session_id -> bool（True 为注入人格）
        self.enabled_sessions: dict[str, bool] = {}

    def _session_id(self, event: AstrMessageEvent) -> str:
        """获取会话标识，兼容不同 AstrBot 版本。"""
        unified = getattr(event, "unified_msg_origin", None)
        if unified:
            return unified
        sender = event.message_obj.sender
        return f"{getattr(sender, 'platform', '')}:{getattr(sender, 'user_id', '')}"

    def _persona_enabled(self, event: AstrMessageEvent) -> bool:
        return self.enabled_sessions.get(self._session_id(event), True)

    # ---------- LLM 请求钩子：人格 + 情景感知 ----------

    @on_llm_request()
    async def inject_persona(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """LLM 请求前：注入人格与情景感知，截断过长上下文（只修改 req）。"""
        if not self.persona_text or not self._persona_enabled(event):
            return
        sid = self._session_id(event)

        # 记录用户消息（供间隔/话题突变判断）
        self.memory.on_user_message(sid, req.prompt or "")

        # 1) 注入人格
        if _PERSONA_MARK not in req.system_prompt:
            req.system_prompt = f"{_PERSONA_MARK}\n{self.persona_text}\n\n{req.system_prompt}"

        # 2) 上下文截断（遗忘机制）
        contexts = getattr(req, "contexts", None) or []
        req.contexts, trimmed = self.memory.trim_contexts(contexts)

        # 3) 情景感知
        situation = self.memory.build_situation(sid, req.prompt or "", trimmed)
        if situation:
            req.system_prompt = f"{req.system_prompt}\n{_SITUATION_MARK}\n{situation}"

    # ---------- 结果装饰钩子：分条发送 + 表情包 ----------

    @on_decorating_result()
    async def handle_result(self, event: AstrMessageEvent) -> None:
        """发送前处理 LLM 结果：分条、去句号、[表情包:情绪词] 换图片。"""
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
        parts = parse_reply(text)
        sid = self._session_id(event)
        self.memory.on_reply(sid, text)  # 检测"要去忙/睡了"等意图

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
        sent_sticker = False  # 每轮回复最多 1 张表情包
        for kind, payload in parts:
            if kind == "img":
                if sent_sticker or self.stickers.is_rate_limited(sid):
                    continue
                img = self.stickers.pick(sid, payload)
                if img is None:
                    logger.warning("[hanhan] stickers/ 文件夹为空或不存在，跳过表情包")
                    continue
                await event.send(MessageChain().file_image(str(img)))
                self.stickers.record_used(sid, img.name)
                self.stickers.record_sent(sid)
                sent_sticker = True
            else:
                await event.send(MessageChain().message(payload))

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
