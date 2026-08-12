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
import time
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
_PLUGIN_VERSION = "1.0.32"


def _fmt_window(window: tuple[int, int]) -> str:
    """把跨天绝对分钟窗口格式化为 '23:00 ~ 01:30' 样式。"""
    lo, hi = window
    return f"{lo % (24 * 60) // 60:02d}:{lo % 60:02d} ~ {hi % (24 * 60) // 60:02d}:{hi % 60:02d}"


def _deep_merge(base: dict, *overrides: dict) -> dict:
    """深合并多个 dict（后者覆盖前者，嵌套 dict 递归合并，其余类型直接覆盖）。"""
    result = json.loads(json.dumps(base))
    for override in overrides:
        if not isinstance(override, dict):
            continue
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
    return result


def _diff_keys(builtin: dict, file_cfg: dict, prefix: str = "") -> set:
    """递归找出 file_cfg 中与内置默认不同的叶子键（点分路径），视为用户微调。"""
    keys: set = set()
    for k, v in file_cfg.items():
        if k == "_user_override":
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict) and isinstance(builtin.get(k), dict):
            keys |= _diff_keys(builtin[k], v, path + ".")
        elif v != builtin.get(k):
            keys.add(path)
    return keys


def _apply_override_keys(target: dict, source: dict, keys: set) -> dict:
    """把 source 中指定点分路径的键覆盖到 target（深拷贝避免污染）。"""
    result = json.loads(json.dumps(target))
    for path in keys:
        parts = path.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        parent = source
        for part in parts[:-1]:
            parent = parent.get(part, {})
        if parts[-1] in parent:
            node[parts[-1]] = parent[parts[-1]]
    return result


# 内置默认表现参数（与 behavior_config.json 全量一致；文件缺失时兜底）
_DEFAULT_BEHAVIOR: dict = {
      "persona": "persona_prompt",
      "private_only": True,
      "reply": {
        "base_delay": [
          1.0,
          2.5
        ],
        "busy_delay": [
          20,
          60
        ],
        "gap_extra_30m": [
          2,
          5
        ],
        "gap_extra_2h": [
          4,
          9
        ],
        "gap_extra_12h": [
          6,
          15
        ],
        "max_delay": 60,
        "max_stickers_per_reply": 1,
        "boost_prob": 0.35,
        "merge_window": 3.5,
        "merge_max_wait": 2.5,
        "merge_only_private": True,
        "max_reply_parts": 6,
        "merge_max_total": 15.0,
        "sticker_only_prob": 0.4
      },
      "sticker": {
        "avoid_repeat": 2,
        "rate_max": 4,
        "rate_window": 600,
        "tags": {}
      },
      "sticker_emotion": {
        "开心": [
          "开心",
          "扭一扭",
          "扭屁股",
          "摇摇晃晃",
          "wave",
          "melt"
        ],
        "得意": [
          "棒",
          "点赞",
          "牛",
          "膜拜",
          "triumph"
        ],
        "夸赞": [
          "棒",
          "点赞",
          "牛",
          "膜拜"
        ],
        "爱你": [
          "爱你",
          "爱你2",
          "贴贴",
          "love",
          "kissheart"
        ],
        "亲亲": [
          "亲亲",
          "亲脸",
          "偷偷给心",
          "kissheart"
        ],
        "贴贴": [
          "贴贴",
          "爱你",
          "扭屁股",
          "snuggle"
        ],
        "撒娇": [
          "扭屁股",
          "贴贴",
          "摇摇晃晃",
          "melt"
        ],
        "无语": [
          "呕吼",
          "不要",
          "无奈",
          "吹风",
          "dead",
          "facepalm"
        ],
        "无奈": [
          "无奈",
          "呕吼",
          "摇摇晃晃",
          "dead",
          "disturbed"
        ],
        "尴尬": [
          "脸红",
          "facepalm",
          "无奈"
        ],
        "生气": [
          "咬你",
          "吃掉你",
          "不准色色",
          "生气"
        ],
        "害羞": [
          "脸红",
          "偷偷看",
          "亲亲",
          "亲脸"
        ],
        "疑问": [
          "问号",
          "思考中",
          "umm"
        ],
        "思考中": [
          "思考中",
          "问号",
          "think"
        ],
        "难过": [
          "cry",
          "sadreach",
          "heartbroken",
          "再见",
          "呕吼"
        ],
        "委屈": [
          "cry",
          "sadreach",
          "不要"
        ],
        "害怕": [
          "scared",
          "shocked"
        ],
        "震惊": [
          "shocked",
          "openmouth",
          "呕吼"
        ],
        "惊讶": [
          "openmouth",
          "shocked",
          "问号"
        ],
        "敷衍": [
          "不要",
          "呕吼",
          "无奈",
          "dead"
        ],
        "紧张": [
          "sipsweat",
          "脸红",
          "偷偷看"
        ],
        "困": [
          "睡觉",
          "dead"
        ],
        "心碎": [
          "heartbroken",
          "cry"
        ],
        "安慰": [
          "snuggle",
          "贴贴",
          "爱你"
        ],
        "色色": [
          "和我色色",
          "舔屏",
          "不准色色"
        ]
      },
      "text_emotion_rules": [
        [
          "哈哈|笑死|好耶|嘿嘿|乐死|太好啦|开心|高兴|爽死",
          "开心"
        ],
        [
          "好看|漂亮|可爱|好美|好帅|喜欢死|太好看",
          "夸赞"
        ],
        [
          "厉害|太强|牛|优秀|膜拜|佩服|好棒",
          "得意"
        ],
        [
          "无语|救命|绝了|什么鬼|服了|受不了|麻了|离大谱",
          "无语"
        ],
        [
          "唉|哎|难过|伤心|委屈|想哭|难受|破防|emo",
          "难过"
        ],
        [
          "气死|生气|烦死|可恶|气人",
          "生气"
        ],
        [
          "害羞|不好意思|脸红|羞死",
          "害羞"
        ],
        [
          "哇|震惊|竟然|居然|我的天|天哪|惊了",
          "震惊"
        ],
        [
          "想想|让我想想|思考|琢磨|纠结",
          "思考中"
        ],
        [
          "困|睡觉|晚安|熬不住|好累",
          "困"
        ],
        [
          "想你了|想你|亲亲|么么|抱抱|爱你",
          "爱你"
        ],
        [
          "？|啥|什么|为什么|真的吗|是这样吗",
          "疑问"
        ]
      ],
      "proactive": {
        "night_prob": 0.3,
        "day_prob": 0.1,
        "night_window": [
          1380,
          1530
        ],
        "day_window": [
          720,
          1320
        ],
        "loop_interval": 60,
        "max_retries": 3
      },
      "memory": {
        "max_contexts": 30,
        "topic_shift_min_overlap": 0.2,
        "topic_shift_max_gap": 600,
        "busy_window": 7200
      },
      "vision": {
        "max_images": 3,
        "timeout": 30,
        "describe_prompt": "你是憨憨的眼睛。用户刚刚给憨憨发了一张图片，请客观、简短地描述图片内容：画面主体、场景、图片里的文字（如果有）。200 字以内，只描述，不评价，不要揣测发图人的意图。"
      }
    }





class HanhanPersonaPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.config = config or {}
        # 表现参数：内置默认 → 人格模板推荐 → 全局 behavior_config.json（用户微调）
        self.behavior_file = _PLUGIN_DIR / "behavior_config.json"
        self.bcfg = self._load_behavior_config()
        # 人格模板：persona/<名>.md，激活名存配置（模板可带同名 .behavior.json 推荐参数）
        self.persona_name = str(self.bcfg.get("persona", "persona_prompt"))
        self.persona_file = _PLUGIN_DIR / "persona" / f"{self.persona_name}.md"
        if not self.persona_file.exists():
            logger.warning(f"[hanhan] 人格模板 {self.persona_name} 不存在，回退 persona_prompt")
            self.persona_name = "persona_prompt"
            self.persona_file = _PLUGIN_DIR / "persona" / f"{self.persona_name}.md"
        self.persona = PersonaLoader(self.persona_file)
        self.persona_text = self.persona.load()
        # 传入人格文本：其中的【时间线】表是身份阶段推算的依据（时间不写死）
        self.memory = MemoryEngine(persona_text=self.persona_text, **self.bcfg["memory"])
        st = self.bcfg["sticker"]
        self.stickers = StickerBot(
            _PLUGIN_DIR / "stickers",
            avoid_repeat=int(st.get("avoid_repeat", 2)),
            rate_window=float(st.get("rate_window", 600)),
            rate_max=int(st.get("rate_max", 4)),
            boost_prob=float(st.get("boost_prob", 0.35)),
            tags=st.get("tags") or {},
            emotion_keywords=self.bcfg.get("sticker_emotion"),
            text_rules=self.bcfg.get("text_emotion_rules"),
        )
        # 主动消息：深夜语音/白天分享，随机时段（状态持久化在插件目录）
        self.proactive = ProactiveSender(
            context=self.context,
            persona_text=self.persona_text,
            sticker_bot=self.stickers,
            state_file=_PLUGIN_DIR / "proactive_state.json",
            **self.bcfg["proactive"],
        )
        # 图片识别：百炼多模态（配置了 API Key 才启用），让她"看到"用户发的图
        # key 来源：环境变量 HANHAN_BAILIAN_API_KEY > WebUI 插件配置 > 插件目录 bailian.key
        self.vision = VisionEngine(
            api_key=(self.config or {}).get("bailian_api_key", ""),
            endpoint=(self.config or {}).get("bailian_endpoint", ""),
            model=(self.config or {}).get("bailian_model", ""),
            key_file=str(_PLUGIN_DIR / "bailian.key"),
            **self.bcfg["vision"],
        )
        # 诊断：加载时确认识别是否启用（环境变量需在 AstrBot 进程启动前设置）
        logger.info(
            "[hanhan] 图片识别"
            + ("已启用（环境变量 HANHAN_BAILIAN_API_KEY）" if self.vision.enabled() else "未启用：环境变量 HANHAN_BAILIAN_API_KEY 未读到")
            + f"，端点={self.vision.endpoint}"
        )
        # 会话粒度开关：session_id -> bool（True 为注入人格）
        self.enabled_sessions: dict[str, bool] = {}
        # 消息合并缓冲区：session_id -> {components, last_arrival, count}
        self._merge_buffers: dict[str, dict] = {}
        # 图片识别期间的消息吸收缓冲：session_id -> {components, last_arrival}
        self._vision_buffers: dict[str, dict] = {}
        # 插件页面（WebUI pages/ 目录）调用的接口：路由必须带插件名前缀
        self.context.register_web_api(
            f"/{_PLUGIN_NAME}/status",
            self.page_status,
            ["GET"],
            "插件实时状态（供插件页面调用）",
        )
        self.context.register_web_api(
            f"/{_PLUGIN_NAME}/config",
            self.page_update_config,
            ["POST"],
            "更新人格/表现参数（供插件页面调用）",
        )

    def _load_behavior_config(self) -> dict:
        """加载合并后的表现参数：内置默认 → 人格模板推荐 → 全局配置（用户微调）。

        - 全局 behavior_config.json 是唯一存储（全量文件，缺键自动补全）
        - persona/<名>.behavior.json 为该人格的推荐参数（若存在，覆盖内置默认）
        - 旧版 sticker_config.json 若存在，其 sticker 段自动迁移合并
        """
        # 1) 全局配置（用户微调层）
        file_cfg: dict = {}
        try:
            if self.behavior_file.exists():
                data = json.loads(self.behavior_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    file_cfg = data
        except (OSError, ValueError):
            logger.warning(f"[hanhan] behavior_config.json 解析失败，使用默认配置: {self.behavior_file}")
        # 2) 旧 sticker_config.json 迁移（v1.0.24 及之前）
        legacy = _PLUGIN_DIR / "sticker_config.json"
        if legacy.exists() and not file_cfg.get("sticker"):
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    file_cfg["sticker"] = data
                    logger.info("[hanhan] 已从 sticker_config.json 迁移 sticker 段到 behavior_config.json")
            except (OSError, ValueError):
                pass
        # 3) 模板推荐参数
        persona_name = str(file_cfg.get("persona", _DEFAULT_BEHAVIOR.get("persona", "persona_prompt")))
        tpl_cfg: dict = {}
        tpl_file = _PLUGIN_DIR / "persona" / f"{persona_name}.behavior.json"
        if tpl_file.exists():
            try:
                data = json.loads(tpl_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    tpl_cfg = data
            except (OSError, ValueError):
                logger.warning(f"[hanhan] 人格模板参数解析失败: {tpl_file}")
        # 4) 合并：内置默认 → 模板推荐 → 用户微调（仅用户改过的键，细粒度）
        merged = _deep_merge(_DEFAULT_BEHAVIOR, tpl_cfg)
        user_keys = set(file_cfg.get("_user_override", []) or [])
        user_keys |= _diff_keys(_DEFAULT_BEHAVIOR, file_cfg)  # 与内置默认不同的键视为用户微调
        if user_keys:
            merged = _apply_override_keys(merged, file_cfg, user_keys)
        merged.pop("_user_override", None)
        merged["persona"] = persona_name
        return merged

    def _save_behavior_config(self, cfg: dict, updated_keys: Optional[list] = None) -> None:
        """增量持久化全局配置：只写 persona 与用户本次改过的键（微调层，不写模板参数）。"""
        try:
            data = (
                json.loads(self.behavior_file.read_text(encoding="utf-8"))
                if self.behavior_file.exists()
                else {}
            )
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["persona"] = cfg.get("persona", self.bcfg.get("persona", "persona_prompt"))
        if updated_keys:
            for path in updated_keys:
                parts = path.split(".")
                node = data
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                parent = cfg
                for part in parts[:-1]:
                    parent = parent.get(part, {})
                if parts[-1] in parent:
                    node[parts[-1]] = parent[parts[-1]]
            seen = set(data.get("_user_override", []) or []) | set(updated_keys)
            data["_user_override"] = sorted(seen)
        try:
            self.behavior_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"[hanhan] behavior_config.json 保存失败: {e}")

    def list_persona_templates(self) -> list[str]:
        """列出 persona/ 目录下的人格模板名（*.md，不含 .behavior.json）。"""
        try:
            return sorted(
                p.stem for p in (_PLUGIN_DIR / "persona").glob("*.md")
            )
        except OSError:
            return []

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

    def _private_only(self) -> bool:
        """人格是否仅私聊生效（behavior_config.json 的 private_only，可调）。"""
        return bool(self.bcfg.get("private_only", True))

    async def _human_pause(self, sid: str) -> None:
        """模拟真人回复节奏：看完消息 + 打字 + 情境延迟，避免秒回露出 AI 味。

        - 基础：随机 1.5~4s（看消息、打字）
        - 她说过要去忙/睡：下一轮"丢失"20~60s 再回（忙完才看到），只慢一次
        - 隔得越久回得越慢：30min+ 附加 2~5s；2h+ 附加 4~9s；12h+ 附加 6~15s
        - 总延迟封顶 120s，避免像卡死
        """
        rp = self.bcfg.get("reply", {})
        base = rp.get("base_delay", [1.5, 4.0])
        busy = rp.get("busy_delay", [20, 60])
        gap_30m = rp.get("gap_extra_30m", [2, 5])
        gap_2h = rp.get("gap_extra_2h", [4, 9])
        gap_12h = rp.get("gap_extra_12h", [6, 15])
        max_delay = float(rp.get("max_delay", 120))
        delay = 0.0
        if self.memory.take_busy_pause(sid):
            delay = random.uniform(*busy)  # 她说去忙了，这一轮"忙完回来"
        else:
            delay = random.uniform(*base)
            gap = self.memory.gap_seconds(sid) or 0.0
            if gap > 12 * 3600:
                delay += random.uniform(*gap_12h)  # 隔了很久，像刚想起来回
            elif gap > 2 * 3600:
                delay += random.uniform(*gap_2h)
            elif gap > 30 * 60:
                delay += random.uniform(*gap_30m)
        await asyncio.sleep(min(delay, max_delay))

    # ---------- LLM 请求钩子：人格 + 情景感知 ----------

    @on_llm_request()
    async def inject_persona(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """LLM 请求前：注入人格与情景感知，截断过长上下文（只修改 req）。"""
        if not self.persona_text or not self._persona_enabled(event):
            return
        if self._private_only() and not self._is_private(event):
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

    async def _absorb_sticker_only(self, event: AstrMessageEvent) -> bool:
        """纯表情/无实质文本消息（图片表情、误发）：不触发 LLM 回复。

        - 有活跃合并/识别缓冲：并入缓冲，等后续文本消息一起处理
        - 无缓冲：吞掉本消息，按 sticker_only_prob 概率回一张随机表情包
          （模拟真实回应），否则静默——表情包只是情绪助词，不值得单独回文字
        """
        if not self._is_private(event):
            return False
        has_text = any(
            isinstance(c, Plain) and c.text.strip()
            for c in event.message_obj.message
        )
        if has_text:
            return False
        sid = self._session_id(event)
        if sid in self._merge_buffers or sid in self._vision_buffers:
            return await self._merge_messages(event)  # 并入现有缓冲
        prob = float(self.bcfg.get("reply", {}).get("sticker_only_prob", 0.4))
        if random.random() < prob:
            img = self.stickers.pick(sid, None)
            if img is not None:
                try:
                    await event.send(MessageChain().file_image(str(ensure_sendable(img))))
                    self.stickers.record_used(sid, img.name)
                    self.stickers.record_sent(sid)
                    logger.info(f"[hanhan] 纯表情消息回应表情包: {img.name}")
                except Exception as e:
                    logger.warning(f"[hanhan] 纯表情回应发送失败: {e}")
        else:
            logger.info("[hanhan] 纯表情消息已静默（情绪助词，不单独回复）")
        return True

    async def _merge_messages(self, event: AstrMessageEvent) -> bool:
        """短时间多条消息合并（去抖）：同一私聊会话在窗口内连发的多条消息，
        吞掉后续消息并入缓冲区，等消息流稳定（对方输入完）后放行第一条，
        让 LLM 整合回复；窗口超时仍未发完则先回复已收到的部分。

        返回 True 表示本消息已被吞掉/正在等待合并（不再走 LLM 管道）。
        参数：reply.merge_window（窗口秒，0=关闭）、merge_max_wait（额外等待上限）、
        merge_only_private（仅私聊合并）。
        """
        if not self._is_private(event):
            return False  # 群聊不合并（多人消息不能混淆成一条）
        if not self.bcfg.get("reply", {}).get("merge_only_private", True):
            return False
        window = float(self.bcfg.get("reply", {}).get("merge_window", 2.5))
        if window <= 0:
            return False
        sid = self._session_id(event)
        # 图片识别期间：后续消息（如"我刚发的图是xxx"）直接吸收，识别完成后拼接
        vbuf = self._vision_buffers.get(sid)
        if vbuf is not None:
            vbuf["components"].append(list(event.message_obj.message))
            vbuf["last_arrival"] = time.time()
            logger.info(f"[hanhan] 图片识别期间消息已吸收（{sid}）")
            return True
        buf = self._merge_buffers.get(sid)
        if buf is not None:
            # 后续消息：并入缓冲区，吞掉本消息（不触发独立回复）
            buf["components"].append(list(event.message_obj.message))
            buf["last_arrival"] = time.time()
            buf["count"] += 1
            logger.info(f"[hanhan] 消息合并：{sid} 第 {buf['count']} 条并入，等待消息流稳定")
            return True
        # 第一条消息：建立缓冲区，等待窗口（消息流持续则顺延，最多窗口+额外等待）
        self._merge_buffers[sid] = {
            "components": [list(event.message_obj.message)],
            "last_arrival": time.time(),
            "count": 1,
        }
        extra_wait = float(self.bcfg.get("reply", {}).get("merge_max_wait", window))
        max_total = float(self.bcfg.get("reply", {}).get("merge_max_total", 15.0))
        deadline = time.time() + window + extra_wait
        hard = time.time() + max_total  # 绝对上限：连发消息再多也最多等这么久
        try:
            while time.time() < deadline and time.time() < hard:
                await asyncio.sleep(0.3)
                buf = self._merge_buffers.get(sid)
                if buf is None:
                    break
                # 有新消息：顺延稳定期（最多到绝对上限），覆盖"连发十几条"场景
                deadline = max(deadline, buf["last_arrival"] + 1.5)
                # 消息流稳定判定只在窗口期之后生效（至少等满 window+extra）：
                # 否则第一条创建 1 秒后就被放行，连发消息根本合并不上
                if time.time() >= deadline and time.time() - buf["last_arrival"] >= 1.2:
                    break  # 窗口已满且 1.2 秒无新消息，认为对方输入完了
        finally:
            buf = self._merge_buffers.pop(sid, None)
        if not buf or buf["count"] <= 1:
            return False  # 只有一条消息：正常放行
        # 合并：后续消息的文本并入第一条（Plain 拼接），富媒体组件追加到消息链
        extra_texts = []
        for comps in buf["components"][1:]:
            for comp in comps:
                if isinstance(comp, Plain):
                    extra_texts.append(comp.text)
                else:
                    event.message_obj.message.append(comp)
        if extra_texts:
            joined = "\n".join(t for t in extra_texts if t)
            if joined:
                event.message_obj.message.append(Plain(joined))
                event.message_str = f"{event.message_str}\n{joined}"
                event.message_obj.message_str = f"{event.message_obj.message_str}\n{joined}"
        logger.info(f"[hanhan] 消息合并完成：{sid} 合并 {buf['count']} 条为一条，整合回复")
        return False

    @regex(r"[\s\S]*")
    async def capture_vision(self, event: AstrMessageEvent) -> None:
        """消息过滤器阶段（所有消息都经过，包括 agent 的 follow-up 消息）：
        检测用户消息里的图片，调百炼识别后把描述写回消息链。

        v4.2x 的 agent 模式下图片消息可能作为 follow-up 并入进行中的 agent run，
        不触发 on_llm_request——所以在更早的 filter 阶段处理，保证 agent 无论哪轮都能看到描述。
        同时做短时间多条消息合并（详见 _merge_messages）。
        """
        if await self._absorb_sticker_only(event):
            return  # 纯表情消息：已回应表情包或静默，不触发 LLM
        if await self._merge_messages(event):
            return  # 已被合并等待/吞掉，不继续走 LLM 管道
        if not self.vision.enabled():
            return
        if self._private_only() and not self._is_private(event):
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
        # 识别期间（约 2~10s）同会话的后续消息会被吸收，识别完成后一起拼接，
        # 避免"发图后又补了文字说明"导致 follow-up 乱序、回复上下文错乱
        sid = self._session_id(event)
        self._vision_buffers[sid] = {"components": [], "last_arrival": time.time()}
        try:
            desc = await self.vision.describe(imgs)
        finally:
            vbuf = self._vision_buffers.pop(sid, None)
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
        # 拼接识别期间吸收的后续消息（图片 + 文字说明 → 一条完整消息进 LLM）
        if vbuf:
            extra = [c.text for comps in vbuf["components"] for c in comps if isinstance(c, Plain)]
            extra = [t for t in extra if t]
            if extra:
                joined = "\n".join(extra)
                event.message_obj.message.append(Plain(joined))
                event.message_str = f"{event.message_str}\n{joined}"
                event.message_obj.message_str = f"{event.message_obj.message_str}\n{joined}"
                logger.info(f"[hanhan] 图片识别期间吸收 {len(extra)} 条文本已拼接")
        logger.info(
            f"[hanhan] 图片消息已替换为文本描述: {text[:60]}{'…' if len(text) > 60 else ''}"
        )

    # ---------- 结果装饰钩子：分条发送 + 表情包 ----------

    @on_decorating_result()
    async def handle_result(self, event: AstrMessageEvent) -> None:
        """发送前处理 LLM 结果：分条、去句号、[表情包:情绪词] 换图片。"""
        if not self._persona_enabled(event):
            return
        if self._private_only() and not self._is_private(event):
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

        # 回复条数上限：超过则把后面的文本合并成一条（避免"发了一大堆"）
        max_parts = int(self.bcfg.get("reply", {}).get("max_reply_parts", 6))
        if len(parts) > max_parts:
            head, tail = parts[:max_parts], parts[max_parts:]
            tail_texts = [p[1] for p in tail if p[0] == "text" and p[1]]
            tail_imgs = [p for p in tail if p[0] == "img"]
            if tail_texts:
                head.append(("text", "\n".join(tail_texts)))
            head.extend(tail_imgs)
            parts = head
            logger.info(f"[hanhan] 回复条数超限，{len(tail)} 条合并为 1 条")

        # 补发表情包：LLM 这轮没标表情包时，按概率从回复文本推断情绪补一张
        # （提升使用频率；限频时跳过，避免刷屏）
        if (
            not any(kind == "img" for kind, _ in parts)
            and not self.stickers.is_rate_limited(sid)
            and random.random() < self.stickers.boost_prob
        ):
            emotion = infer_emotion(text, self.stickers.text_rules)
            if emotion:
                parts.append(("img", emotion))
                logger.info(f"[hanhan] 补发表情包（文本情绪推断: {emotion}）")

        # 先停掉"正在输入"指示再停顿——否则停顿期间用户一直看到"输入中"却没内容
        try:
            await event.stop_typing()
        except Exception:
            pass
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
        max_stickers = int(self.bcfg.get("reply", {}).get("max_stickers_per_reply", 1))
        sent_sticker = 0  # 每轮回复表情包上限（可配置）
        for kind, payload in parts:
            if kind == "img":
                if sent_sticker >= max_stickers or self.stickers.is_rate_limited(sid):
                    continue
                img = self.stickers.pick(sid, payload)
                if img is None:
                    logger.warning("[hanhan] stickers/ 文件夹为空或不存在，跳过表情包")
                    continue
                await event.send(MessageChain().file_image(str(ensure_sendable(img))))
                self.stickers.record_used(sid, img.name)
                self.stickers.record_sent(sid)
                sent_sticker += 1
            else:
                await event.send(MessageChain().message(payload))

    # ---------- 插件页面（WebUI）接口 ----------

    async def page_status(self) -> dict:
        """插件页面状态接口：返回憨憨的实时运行状态（只读）。

        路由 /astrbot_plugin_hanhan/status 在 __init__ 中注册，由
        pages/overview/index.html 通过 bridge.apiGet("status") 调用。
        """
        logger.info("[hanhan] 插件页面调用 /status")
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
                    "active": self.persona_name,
                    "templates": self.list_persona_templates(),
                    "private_only": self._private_only(),
                    "params": {
                        "reply": self.bcfg.get("reply", {}),
                        "proactive": {
                            "night_prob": self.proactive.night_prob,
                            "day_prob": self.proactive.day_prob,
                            "night_window": list(self.proactive.night_window),
                            "day_window": list(self.proactive.day_window),
                        },
                        "sticker": {
                            "avoid_repeat": self.stickers.avoid_repeat,
                            "rate_max": self.stickers.rate_max,
                            "rate_window": self.stickers.rate_window,
                            "boost_prob": self.stickers.boost_prob,
                        },
                    },
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
        """插件页面配置接口：更新人格模板与表现参数（白名单字段，热生效并持久化）。

        支持字段：
        - persona: 切换人格模板（文件名不含 .md，须在 persona/ 目录，需重载插件生效）
        - private_only: bool
        - reply.*: base_delay/busy_delay/gap_extra_* 为 [min,max] 秒；max_delay 秒；
          max_stickers_per_reply 张；boost_prob 0~1
        - sticker.rate_max / rate_window / avoid_repeat
        - proactive.night_prob / day_prob（0~1，次日计划生效）；night_window / day_window
          为 [起,止] 绝对分钟
        """
        logger.info("[hanhan] 插件页面调用 /config")
        try:
            data = await request.json(default={})
        except Exception as e:
            return error_response(f"请求体解析失败: {e}", status_code=400)
        if not isinstance(data, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)

        cfg = self._load_behavior_config()
        updated: dict = {}
        try:
            # 人格模板切换（需重载生效）
            if "persona" in data:
                name = str(data["persona"]).strip()
                if name not in self.list_persona_templates():
                    return error_response(f"人格模板不存在：{name}", status_code=400)
                cfg["persona"] = name
                updated["persona"] = name
            if "private_only" in data:
                cfg["private_only"] = bool(data["private_only"])
                updated["private_only"] = cfg["private_only"]
            # 回复节奏
            rp = cfg.setdefault("reply", {})
            for key in ("base_delay", "busy_delay", "gap_extra_30m", "gap_extra_2h", "gap_extra_12h"):
                if key in data:
                    v = data[key]
                    if not (isinstance(v, list) and len(v) == 2
                            and all(isinstance(x, (int, float)) and 0 <= float(x) <= 3600 for x in v)):
                        raise ValueError
                    rp[key] = [float(x) for x in v]
                    updated[f"reply.{key}"] = rp[key]
            if "max_delay" in data:
                v = float(data["max_delay"])
                if not 0 <= v <= 3600:
                    raise ValueError
                rp["max_delay"], updated["reply.max_delay"] = v, v
            if "max_stickers_per_reply" in data:
                v = int(data["max_stickers_per_reply"])
                if not 1 <= v <= 10:
                    raise ValueError
                rp["max_stickers_per_reply"], updated["reply.max_stickers_per_reply"] = v, v
            if "boost_prob" in data:
                v = float(data["boost_prob"])
                if not 0 <= v <= 1:
                    raise ValueError
                rp["boost_prob"], updated["reply.boost_prob"] = v, v
            if "max_reply_parts" in data:
                v = int(data["max_reply_parts"])
                if not 1 <= v <= 20:
                    raise ValueError
                rp["max_reply_parts"], updated["reply.max_reply_parts"] = v, v
            if "sticker_only_prob" in data:
                v = float(data["sticker_only_prob"])
                if not 0 <= v <= 1:
                    raise ValueError
                rp["sticker_only_prob"], updated["reply.sticker_only_prob"] = v, v
            for key in ("merge_window", "merge_max_wait", "merge_max_total"):
                if key in data:
                    v = float(data[key])
                    if not 0 <= v <= 60:
                        raise ValueError
                    rp[key], updated[f"reply.{key}"] = v, v
            # 表情包限频
            st = cfg.setdefault("sticker", {})
            if "rate_max" in data:
                v = int(data["rate_max"])
                if not 1 <= v <= 20:
                    raise ValueError
                st["rate_max"], updated["sticker.rate_max"] = v, v
            if "rate_window" in data:
                v = float(data["rate_window"])
                if not 30 <= v <= 86400:
                    raise ValueError
                st["rate_window"], updated["sticker.rate_window"] = v, v
            if "avoid_repeat" in data:
                v = int(data["avoid_repeat"])
                if not 0 <= v <= 10:
                    raise ValueError
                st["avoid_repeat"], updated["sticker.avoid_repeat"] = v, v
            # 主动消息概率与窗口
            pa = cfg.setdefault("proactive", {})
            if "night_prob" in data:
                v = float(data["night_prob"])
                if not 0 <= v <= 1:
                    raise ValueError
                pa["night_prob"], updated["proactive.night_prob"] = v, v
            if "day_prob" in data:
                v = float(data["day_prob"])
                if not 0 <= v <= 1:
                    raise ValueError
                pa["day_prob"], updated["proactive.day_prob"] = v, v
            for key in ("night_window", "day_window"):
                if key in data:
                    v = data[key]
                    if not (isinstance(v, list) and len(v) == 2
                            and all(isinstance(x, (int, float)) for x in v)):
                        raise ValueError
                    pa[key] = [int(x) for x in v]
                    updated[f"proactive.{key}"] = pa[key]
        except (TypeError, ValueError):
            return error_response(
                "参数不合法：延迟/窗口为 [min,max] 数组，概率 0~1，rate_max 1~20，rate_window 30~86400",
                status_code=400,
            )
        # 热生效（persona 切换与 private_only 需重载）
        self.bcfg = cfg
        self.stickers.boost_prob = float(cfg.get("reply", {}).get("boost_prob", self.stickers.boost_prob))
        self.stickers.rate_max = int(cfg.get("sticker", {}).get("rate_max", self.stickers.rate_max))
        self.stickers.rate_window = float(cfg.get("sticker", {}).get("rate_window", self.stickers.rate_window))
        self.stickers.avoid_repeat = int(cfg.get("sticker", {}).get("avoid_repeat", self.stickers.avoid_repeat))
        self.proactive.night_prob = float(cfg.get("proactive", {}).get("night_prob", self.proactive.night_prob))
        self.proactive.day_prob = float(cfg.get("proactive", {}).get("day_prob", self.proactive.day_prob))
        self.proactive.night_window = tuple(cfg.get("proactive", {}).get("night_window", list(self.proactive.night_window)))
        self.proactive.day_window = tuple(cfg.get("proactive", {}).get("day_window", list(self.proactive.day_window)))
        self._save_behavior_config(cfg, list(updated.keys()))
        logger.info(f"[hanhan] 插件页面更新配置: {updated}")
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
