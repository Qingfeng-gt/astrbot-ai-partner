"""
憨憨 — AstrBot 人格插件（适配 AstrBot v4.x）

把 /create-ex skill 生成的前任人格（浓缩为单一 md 提示词）注入
AstrBot 的 LLM 请求，使 bot 以憨憨的身份、记忆和说话方式回复。

模块结构：
- main.py            编排层：Star 类、钩子、命令（本文件，必须位于根目录）
- metadata.yaml      插件元数据（必须位于根目录）
- core/              业务逻辑包：persona_loader / reply_processor /
                     sticker_bot / memory_engine
- persona/           人格提示词（单一 md 提示词，可自行编辑）
- stickers/          表情包文件夹（用户直接管理，位于根目录）

行为特性：
- 人格仅在私聊会话生效（前任人格不适合群聊）；群聊消息走默认回复
- @on_llm_request：注入人格 + 情景感知（间隔、话题突变、遗忘提示）+ 上下文截断
- @on_decorating_result：按行拆成多条消息依次发送；[表情包:情绪词] 换图片；
  消息结尾单句号强制去除；每轮最多 1 张表情包且有限频
- 注意：v4.x 钩子为装饰器注册制，且钩子调度不检查事件过滤器，
  私聊判断需在函数内部完成；流式输出时装饰阶段不生效
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.event.filter import (
    command,
    llm_tool,
    on_astrbot_loaded,
    on_decorating_result,
    on_llm_request,
    regex,
)
from astrbot.api.message_components import Image, Plain
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request

from .core.memory_engine import MemoryEngine
from .core.persona_loader import PersonaLoader
from .core.proactive_sender import DAY_PROB, DAY_WINDOW, NIGHT_PROB, NIGHT_WINDOW, ProactiveSender
from .core.reply_processor import parse_reply
from .core.sticker_bot import StickerBot, ensure_sendable, infer_emotion
from .core.vision import VisionEngine

# 注入标记：用于防止重复注入
_PERSONA_MARK = "<!-- hanhan-persona -->"
_SITUATION_MARK = "<!-- hanhan-situation -->"
_PLUGIN_DIR = Path(__file__).parent
# 插件名/版本（Web API 路由前缀与页面展示用，版本与 metadata.yaml 同步）
_PLUGIN_NAME = "astrbot_plugin_hanhan"
_PLUGIN_VERSION = "1.0.24"


def _fmt_window(window: tuple[int, int]) -> str:
    """把跨天绝对分钟窗口格式化为 '23:00 ~ 01:30' 样式。"""
    lo, hi = window
    return f"{lo % (24 * 60) // 60:02d}:{lo % 60:02d} ~ {hi % (24 * 60) // 60:02d}:{hi % 60:02d}"
# 人格是否仅在私聊会话生效（前任人格在群里不合适，默认 True）
_PERSONA_ONLY_PRIVATE = True


class HanhanPersonaPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.config = config or {}
        self.persona = PersonaLoader(_PLUGIN_DIR / "persona" / "persona_prompt.md")
        self.persona_text = self.persona.load()
        # 传入人格文本：其中的【时间线】表是身份阶段推算的依据（时间不写死）
        self.memory = MemoryEngine(persona_text=self.persona_text)
        # 表情包配置（sticker_config.json：多标签 + 补发概率 + 限频，页面可改）
        self.sticker_cfg_file = _PLUGIN_DIR / "sticker_config.json"
        scfg = self._load_sticker_config()
        self.stickers = StickerBot(
            _PLUGIN_DIR / "stickers",
            rate_window=float(scfg.get("rate_window", 600)),
            rate_max=int(scfg.get("rate_max", 4)),
            boost_prob=float(scfg.get("boost_prob", 0.35)),
            tags=scfg.get("tags") or {},
        )
        # 主动消息：深夜语音/白天分享，随机时段（状态持久化在插件目录）
        self.proactive = ProactiveSender(
            context=self.context,
            persona_text=self.persona_text,
            sticker_bot=self.stickers,
            state_file=_PLUGIN_DIR / "proactive_state.json",
        )
        # 图片识别：百炼多模态（配置了 API Key 才启用），让她"看到"用户发的图
        # key 来源：环境变量 HANHAN_BAILIAN_API_KEY > WebUI 插件配置 > 插件目录 bailian.key
        self.vision = VisionEngine(
            api_key=(self.config or {}).get("bailian_api_key", ""),
            endpoint=(self.config or {}).get("bailian_endpoint", ""),
            model=(self.config or {}).get("bailian_model", ""),
            key_file=str(_PLUGIN_DIR / "bailian.key"),
        )
        # 诊断：加载时确认识别是否启用（环境变量需在 AstrBot 进程启动前设置）
        logger.info(
            "[hanhan] 图片识别"
            + ("已启用（环境变量 HANHAN_BAILIAN_API_KEY）" if self.vision.enabled() else "未启用：环境变量 HANHAN_BAILIAN_API_KEY 未读到")
            + f"，端点={self.vision.endpoint}"
        )
        # 会话粒度开关：session_id -> bool（True 为注入人格）
        self.enabled_sessions: dict[str, bool] = {}
        # 插件页面（WebUI pages/ 目录）调用的接口：路由必须带插件名前缀
        self.context.register_web_api(
            f"/{_PLUGIN_NAME}/status",
            self.page_status,
            ["GET"],
            "憨憨插件实时状态（供插件页面调用）",
        )
        self.context.register_web_api(
            f"/{_PLUGIN_NAME}/config",
            self.page_update_config,
            ["POST"],
            "更新憨憨表情包频率参数（供插件页面调用）",
        )

    def _load_sticker_config(self) -> dict:
        """读 sticker_config.json；缺失/损坏时写默认并返回。"""
        default = {
            "boost_prob": 0.35,  # LLM 没标表情包时补发一张的概率
            "rate_max": 4,  # 限频窗口内最多表情包数
            "rate_window": 600,  # 限频窗口（秒）
            "tags": {},  # 多标签：文件名 -> 标签列表（可覆盖自动推断）
        }
        try:
            if self.sticker_cfg_file.exists():
                data = json.loads(self.sticker_cfg_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError):
            logger.warning(f"[hanhan] sticker_config.json 解析失败，使用默认配置: {self.sticker_cfg_file}")
        try:
            self.sticker_cfg_file.write_text(
                json.dumps(default, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"[hanhan] sticker_config.json 写入失败: {e}")
        return default

    def _save_sticker_config(self, cfg: dict) -> None:
        try:
            self.sticker_cfg_file.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"[hanhan] sticker_config.json 保存失败: {e}")

    def _session_id(self, event: AstrMessageEvent) -> str:
        """获取会话标识（unified_msg_origin），兼容不同 AstrBot 版本。

        优先用 event.unified_msg_origin；旧版没有该属性时，兜底拼出三段式
        platform:message_type:session_id —— 主动消息 send_message 依赖此格式解析
        （MessageSession.from_str 需要恰好三段，两段串会抛 ValueError）。
        """
        unified = getattr(event, "unified_msg_origin", None)
        if unified:
            return unified
        sender = event.message_obj.sender
        try:
            mtype = event.get_message_type()
            mtype_str = getattr(mtype, "value", None) or str(mtype)
        except Exception:
            mtype_str = "FriendMessage"  # 兜底：拼不出消息类型时按私聊处理
        return (
            f"{getattr(sender, 'platform', '')}:{mtype_str}:"
            f"{getattr(sender, 'user_id', '')}"
        )

    def _persona_enabled(self, event: AstrMessageEvent) -> bool:
        return self.enabled_sessions.get(self._session_id(event), True)

    def _is_private(self, event: AstrMessageEvent) -> bool:
        """是否私聊会话（LLM 钩子调度不检查事件过滤器，需内部判断）。"""
        try:
            return event.get_message_type() == MessageType.FRIEND_MESSAGE
        except Exception:
            return False

    async def _human_pause(self, sid: str) -> None:
        """模拟真人回复节奏：看完消息 + 打字 + 情境延迟，避免秒回露出 AI 味。

        - 基础：随机 1.5~4s（看消息、打字）
        - 她说过要去忙/睡：下一轮"丢失"20~60s 再回（忙完才看到），只慢一次
        - 隔得越久回得越慢：30min+ 附加 2~5s；2h+ 附加 4~9s；12h+ 附加 6~15s
        - 总延迟封顶 120s，避免像卡死
        """
        delay = 0.0
        if self.memory.take_busy_pause(sid):
            delay = random.uniform(20, 60)  # 她说去忙了，这一轮"忙完回来"
        else:
            delay = random.uniform(1.5, 4.0)
            gap = self.memory.gap_seconds(sid) or 0.0
            if gap > 12 * 3600:
                delay += random.uniform(6, 15)  # 隔了很久，像刚想起来回
            elif gap > 2 * 3600:
                delay += random.uniform(4, 9)
            elif gap > 30 * 60:
                delay += random.uniform(2, 5)
        await asyncio.sleep(min(delay, 120.0))

    # ---------- LLM 请求钩子：人格 + 情景感知 ----------

    @on_llm_request()
    async def inject_persona(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """LLM 请求前：注入人格与情景感知，截断过长上下文（只修改 req）。"""
        if not self.persona_text or not self._persona_enabled(event):
            return
        if _PERSONA_ONLY_PRIVATE and not self._is_private(event):
            return  # 群聊不注入人格，走默认回复
        sid = self._session_id(event)

        # 私聊会话自动绑定为主动消息目标（不可更改）
        self.proactive.record_session(sid)
        # 懒启动调度循环：插件重载时 on_astrbot_loaded 不会再次触发，这里兜底
        self.proactive.start()
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

    @regex(r"[\s\S]*")
    async def capture_vision(self, event: AstrMessageEvent) -> None:
        """消息过滤器阶段（所有消息都经过，包括 agent 的 follow-up 消息）：
        检测用户消息里的图片，调百炼识别后把描述写回消息链。

        v4.2x 的 agent 模式下图片消息可能作为 follow-up 并入进行中的 agent run，
        不触发 on_llm_request——所以在更早的 filter 阶段处理，保证 agent 无论哪轮都能看到描述。
        """
        if not self.vision.enabled():
            return
        if _PERSONA_ONLY_PRIVATE and not self._is_private(event):
            return
        imgs = [
            c for c in event.message_obj.message
            if isinstance(c, Image) and (c.file or c.url or c.path)
        ]
        if not imgs:
            return
        logger.info(
            f"[hanhan] 消息过滤器捕获图片消息: 组件 {len(event.message_obj.message)} 个, "
            f"图片 {len(imgs)} 个, 来源 file={getattr(imgs[0], 'file', '')!r:.60}"
        )
        desc = await self.vision.describe(imgs)
        # 把 Image 组件替换成含描述的文本：v4.2x agent 对图片消息只认 Image 组件
        # （会尝试 file_read_tool 读图，deepseek 无视觉→空回复），不会读附加文本；
        # 换成文本后 agent 的消息里只有描述，必能正常回答。
        # 识别失败也替换为 "[图片]" 占位，避免 agent 空回复循环。
        text = f"[图片]（图片内容：{desc}）" if desc else "[图片]"
        replaced = False
        for i, comp in enumerate(event.message_obj.message):
            if isinstance(comp, Image):
                event.message_obj.message[i] = Plain(text)
                replaced = True
        if not replaced:
            event.message_obj.message.append(Plain(text))
        # 关键：只改消息链不够。AstrBot 的 agent 构建 LLM prompt 用的是
        # event.message_str（事件创建时由适配器冻结的字符串，plugin 过滤器改不到它），
        # 而不是改后的消息链——所以这里必须把描述同步写回 message_str，否则
        # LLM 只会看到 "[图片]" 占位，就会像之前那样回一句空文本 + 表情包。
        # 这与 AstrBot 自身 STT 的写法一致（preprocess_stage 替换组件后同样回写
        # message_str），标准消息和 agent follow-up 消息两条路径都走 message_str。
        event.message_str = event.message_str.replace("[图片]", text, len(imgs))
        event.message_obj.message_str = event.message_obj.message_str.replace(
            "[图片]", text, len(imgs)
        )
        logger.info(
            f"[hanhan] 图片消息已替换为文本描述: {text[:60]}{'…' if len(text) > 60 else ''}"
        )

    # ---------- 结果装饰钩子：分条发送 + 表情包 ----------

    @on_decorating_result()
    async def handle_result(self, event: AstrMessageEvent) -> None:
        """发送前处理 LLM 结果：分条、去句号、[表情包:情绪词] 换图片。"""
        if not self._persona_enabled(event):
            return
        if _PERSONA_ONLY_PRIVATE and not self._is_private(event):
            return  # 群聊不处理人格回复格式
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

        # 补发表情包：LLM 这轮没标表情包时，按概率从回复文本推断情绪补一张
        # （提升使用频率；限频时跳过，避免刷屏）
        if (
            not any(kind == "img" for kind, _ in parts)
            and not self.stickers.is_rate_limited(sid)
            and random.random() < self.stickers.boost_prob
        ):
            emotion = infer_emotion(text)
            if emotion:
                parts.append(("img", emotion))
                logger.info(f"[hanhan] 补发表情包（文本情绪推断: {emotion}）")

        # 模拟真人节奏再回复：基础打字延迟 + 间隔越久越慢 + 说过要去忙则"忙完回来"
        await self._human_pause(sid)

        if len(parts) <= 1 and parts and parts[0][0] == "text":
            # 单条纯文本且未补发：仅当去掉了结尾句号时才重写，否则交给默认流程
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
                await event.send(MessageChain().file_image(str(ensure_sendable(img))))
                self.stickers.record_used(sid, img.name)
                self.stickers.record_sent(sid)
                sent_sticker = True
            else:
                await event.send(MessageChain().message(payload))

    # ---------- 插件页面（WebUI）接口 ----------

    async def page_status(self) -> dict:
        """插件页面状态接口：返回憨憨的实时运行状态（只读）。

        路由 /astrbot_plugin_hanhan/status 在 __init__ 中注册，由
        pages/overview/index.html 通过 bridge.apiGet("status") 调用。
        """
        try:
            sticker_count = len(self.stickers._all_files())
        except Exception:
            sticker_count = 0
        enabled_cnt = sum(1 for v in self.enabled_sessions.values() if v)
        disabled_cnt = sum(1 for v in self.enabled_sessions.values() if not v)
        return json_response(
            {
                "name": _PLUGIN_NAME,
                "version": _PLUGIN_VERSION,
                "persona": {
                    "default_enabled": True,
                    "enabled_sessions": enabled_cnt,
                    "disabled_sessions": disabled_cnt,
                },
                "proactive": {
                    "active": self.proactive.is_active(),
                    "describe": self.proactive.describe(),
                    "night_prob": NIGHT_PROB,
                    "day_prob": DAY_PROB,
                    "night_window": _fmt_window(NIGHT_WINDOW),
                    "day_window": _fmt_window(DAY_WINDOW),
                    "plan": self.proactive.plan_info(),
                },
                "vision": {
                    "enabled": self.vision.enabled(),
                    "model": self.vision.model or "qwen-vl-plus（默认）",
                    "endpoint": self.vision.endpoint,
                },
                "stickers": {
                    "count": sticker_count,
                    "tagged": self.stickers._tagged_count(),
                    "boost_prob": self.stickers.boost_prob,
                    "rate_max": self.stickers.rate_max,
                    "rate_window": self.stickers.rate_window,
                    "top_tags": sorted(
                        self.stickers._tag_stats().items(), key=lambda x: -x[1]
                    )[:10],
                },
                "repo": "https://github.com/Qingfeng-gt/astrbot-ai-partner",
            }
        )

    async def page_update_config(self) -> dict:
        """插件页面配置接口：修改表情包频率参数（白名单字段，热生效并持久化）。"""
        try:
            data = await request.json(default={})
        except Exception as e:
            return error_response(f"请求体解析失败: {e}", status_code=400)
        if not isinstance(data, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        cfg = self._load_sticker_config()
        updated = {}
        try:
            if "boost_prob" in data:
                v = float(data["boost_prob"])
                if not 0 <= v <= 1:
                    raise ValueError
                cfg["boost_prob"], updated["boost_prob"] = v, v
            if "rate_max" in data:
                v = int(data["rate_max"])
                if not 1 <= v <= 20:
                    raise ValueError
                cfg["rate_max"], updated["rate_max"] = v, v
            if "rate_window" in data:
                v = float(data["rate_window"])
                if not 30 <= v <= 86400:
                    raise ValueError
                cfg["rate_window"], updated["rate_window"] = v, v
        except (TypeError, ValueError):
            return error_response("参数不合法（boost_prob 0~1，rate_max 1~20，rate_window 30~86400）", status_code=400)
        # 热生效 + 持久化
        self.stickers.boost_prob = float(cfg["boost_prob"])
        self.stickers.rate_max = int(cfg["rate_max"])
        self.stickers.rate_window = float(cfg["rate_window"])
        self._save_sticker_config(cfg)
        logger.info(f"[hanhan] 插件页面更新表情包参数: {updated}")
        return json_response({"ok": True, **updated})

    # ---------- LLM 工具：表情包 ----------

    @llm_tool(name="send_hanhan_sticker")
    async def send_hanhan_sticker(self, event: AstrMessageEvent, emotion: str):
        """发送一张憨憨表情包（按情绪匹配，如开心、无语、生气时用）。

        表情包只起配合作用：调用时仍必须同时用文字正常回复用户（例如回应图片
        内容、接住话题），表情包不能单独作为一条回复——只有表情包没有文字会
        显得像没看到消息。

        Args:
            emotion(string): 情绪词，1-2 个字描述当前情绪，如：开心、无语、生气、害羞、难过、敷衍、疑问、亲亲、思考中。
        """
        sid = self._session_id(event)
        img = self.stickers.pick(sid, (emotion or "").strip() or None)
        if img is None:
            yield event.plain_result("（stickers/ 文件夹没有可用表情包）")
            return
        yield event.image_result(str(ensure_sendable(img)))  # webp 转 png

    # ---------- 生命周期：主动调度 ----------

    @on_astrbot_loaded()
    async def on_loaded(self) -> None:
        """AstrBot 加载完成后启动主动消息调度（深夜语音/白天分享）。"""
        self.proactive.start()

    async def terminate(self) -> None:
        """插件卸载/重载时停止主动调度。"""
        await self.proactive.stop()

    # ---------- 命令 ----------

    @command("憨憨主动")
    async def toggle_proactive(self, event: AstrMessageEvent):
        """/憨憨主动 —— 切换主动消息开关（私聊自动绑定目标，不可更改）。"""
        active = self.proactive.set_active(not self.proactive.is_active())
        yield event.plain_result(f"憨憨主动消息：{'已开启' if active else '已关闭'}")
        if active:
            yield event.plain_result(self.proactive.describe())

    @command("憨憨主动状态")
    async def proactive_status(self, event: AstrMessageEvent):
        """/憨憨主动状态 —— 查看主动消息的调度与已发状态。"""
        yield event.plain_result(self.proactive.describe())

    @command("憨憨主动测试")
    async def proactive_test(self, event: AstrMessageEvent):
        """/憨憨主动测试 [night|day] —— 预览她此刻会主动说什么（不发送）。"""
        args = (event.message_str or "").split()
        kind = args[1].strip().lower() if len(args) > 1 else "night"
        if kind not in ("night", "day"):
            yield event.plain_result("用法：/憨憨主动测试 night 或 /憨憨主动测试 day")
            return
        text = await self.proactive.preview(kind)
        label = "深夜" if kind == "night" else "白天"
        yield event.plain_result(f"【{label}主动预览】她此刻会说：")
        sid = self._session_id(event)
        for piece in parse_reply(text):
            if piece[0] == "text":
                yield event.plain_result(piece[1])
            else:
                # 预览时直接发出真实表情包图片（与被动回复同一套选图/转换逻辑）
                img = self.stickers.pick(sid, piece[1])
                if img is None:
                    yield event.plain_result("[表情包]（stickers/ 文件夹为空）")
                else:
                    yield event.image_result(str(ensure_sendable(img)))

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
