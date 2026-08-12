"""主动消息调度：深夜语音 / 白天分享，随机时段，按人格主动联系用户。

- 深夜窗口（23:00 ~ 次日 01:30）随机选一个时刻，以"深夜语音"的形式主动发消息
- 白天窗口（12:00 ~ 22:00）随机选一个时刻，主动分享一条
- 消息内容由 LLM 按人格生成（失败时用内置兜底句）
- 目标会话自动记录（用户私聊过即记住），也可用 /憨憨绑定 手动指定
- 状态持久化到插件目录 proactive_state.json，重启不丢
- 发送失败自动重试（上限 3 次），全部失败在状态里记录错误，不谎报"已发送"
"""

import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import MessageChain

from .reply_processor import parse_reply
from .sticker_bot import StickerBot, ensure_sendable

# 深夜窗口：23:00 ~ 次日 01:30（用跨天的绝对分钟数表示，1440 = 次日 00:00）
NIGHT_WINDOW = (23 * 60, 25 * 60 + 30)
# 白天分享窗口：12:00 ~ 22:00
DAY_WINDOW = (12 * 60, 22 * 60)
NIGHT_PROB = 0.3  # 每晚主动联系的概率（低频，回避型人设不会天天找）
DAY_PROB = 0.1  # 每天白天主动分享的概率
LOOP_INTERVAL = 60  # 调度检查间隔（秒）
MAX_SEND_RETRIES = 3  # 单轮主动消息最多重试次数，超过后放弃（避免坏目标每 60s 刷日志/重复烧 LLM）

# 兜底句（LLM 生成失败时使用），保持人格安全
CANNED = {
    "night": [
        "睡了吗\n我睡不着。",
        "刚洗完澡，躺床上发呆\n你干嘛呢",
        "做了一个奇怪的梦",
        "。。。突然想到你上次说的那个事\n[表情包:思考中]",
    ],
    "day": [
        "刚看到一个视频，发你看看\n[表情包:开心]",
        "中午吃什么了\n我这边食堂好难吃。。",
        "今天投了个简历，有点紧张\n[表情包:呕吼]",
        "看到一只猫，跟你上次发的那个好像\n[表情包:亲亲]",
    ],
}


def _cur_minute(dt: datetime) -> int:
    """当前时刻的绝对分钟数；凌晨 0-2 点归到前一天夜里（跨天窗口）。"""
    m = dt.hour * 60 + dt.minute
    if dt.hour < 2:
        m += 24 * 60
    return m


def _pick_future_minute(window: tuple[int, int], cur: int, prob: float) -> Optional[int]:
    """按概率决定是否安排；安排时只在窗口"尚未过去"的部分里随机选一个未来时刻。

    若窗口已完全过去（如白天计划在 22:00 后才生成），返回 None。
    cur 为当前的绝对分钟数。
    """
    if random.random() >= prob:
        return None
    lo, hi = window
    lo = max(lo, cur + 1)  # 至少 1 分钟后，保证不会"生成即触发"
    if lo > hi:
        return None
    return random.randint(lo, hi)


class ProactiveSender:
    """主动消息调度器。"""

    def __init__(
        self,
        context,
        persona_text: str,
        sticker_bot: StickerBot,
        state_file: Path,
    ):
        self.context = context
        self.persona_text = persona_text or ""
        self.stickers = sticker_bot
        self.state_file = Path(state_file)
        self.state: dict = {
            "targets": [],
            "active": True,
            "plan_date": "",
            "night_minute": None,
            "day_minute": None,
            "sent_night": False,
            "sent_day": False,
        }
        self._task: Optional[asyncio.Task] = None
        self._load_state()

    # ---------- 状态持久化 ----------

    def _load_state(self) -> None:
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.state.update({k: v for k, v in data.items() if k in self.state})
        except (OSError, ValueError):
            logger.warning(f"[hanhan] 主动消息状态文件损坏，使用默认状态: {self.state_file}")

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"[hanhan] 主动消息状态保存失败: {e}")

    # ---------- 目标会话 ----------

    def record_session(self, umo: str) -> None:
        """记录一个会话（用户私聊过即记住，作为主动消息目标）。"""
        if umo and umo not in self.state["targets"]:
            self.state["targets"].append(umo)
            self._save_state()
            logger.info(f"[hanhan] 已记录主动消息目标会话: {umo}")

    def bind_session(self, umo: str) -> bool:
        """手动绑定当前会话为主动消息目标。"""
        self.record_session(umo)
        return umo in self.state["targets"]

    def clear_targets(self) -> None:
        self.state["targets"] = []
        self._save_state()

    def set_active(self, active: bool) -> bool:
        self.state["active"] = bool(active)
        self._save_state()
        return self.state["active"]

    def is_active(self) -> bool:
        return bool(self.state["active"])

    def plan_info(self) -> dict:
        """今日调度计划状态（供插件页面展示）：计划时刻、已发状态、上次内容。"""
        fmt = lambda m: (  # noqa: E731
            f"{m % (24 * 60) // 60:02d}:{m % 60:02d}" if isinstance(m, int) else None
        )
        return {
            "plan_date": self.state.get("plan_date", ""),
            "night_at": fmt(self.state.get("night_minute")),
            "day_at": fmt(self.state.get("day_minute")),
            "sent_night": bool(self.state.get("sent_night")),
            "sent_day": bool(self.state.get("sent_day")),
            "last_night_at": self.state.get("last_sent_night_at", ""),
            "last_day_at": self.state.get("last_sent_day_at", ""),
            "last_night_text": self.state.get("last_sent_night_text", ""),
            "last_day_text": self.state.get("last_sent_day_text", ""),
            "targets": len(self.state.get("targets", [])),
        }

    def describe(self) -> str:
        """当前调度状态描述（含计划、已发状态、上次内容，用于 /憨憨主动状态）。"""
        if not self.state["active"]:
            return "主动消息已关闭（/憨憨主动 可开启）"
        if not self.state["targets"]:
            return "主动消息已开启，但还没有目标会话（私聊过就会自动绑定）"
        parts = [f"主动消息已开启，自动绑定 {len(self.state['targets'])} 个私聊会话"]

        nm = self.state["night_minute"]
        if nm is not None:
            # 跨天窗口的分钟数（>=1440）取模归一化到次日 00:00 之后
            hh, mm = (nm % (24 * 60)) // 60, nm % 60
            at = self.state.get("last_sent_night_at")
            if self.state["sent_night"] and at:
                parts.append(f"今晚深夜消息已发送（{at}）")
            elif self.state["sent_night"]:
                parts.append(
                    f"今晚深夜消息发送失败已放弃（{self.state.get('last_night_error', '?')}）"
                )
            else:
                parts.append(f"今晚深夜计划 ~{hh:02d}:{mm:02d} 触发")
        else:
            parts.append("今晚深夜无计划（低频随机）")
        if self.state["day_minute"] is not None:
            at = self.state.get("last_sent_day_at")
            if self.state["sent_day"] and at:
                parts.append(f"今天白天消息已发送（{at}）")
            elif self.state["sent_day"]:
                parts.append(
                    f"今天白天消息发送失败已放弃（{self.state.get('last_day_error', '?')}）"
                )
            else:
                hh, mm = self.state["day_minute"] // 60, self.state["day_minute"] % 60
                parts.append(f"今天白天计划 ~{hh:02d}:{mm:02d} 触发")
        else:
            parts.append("今天白天无计划（低频随机）")

        last = self.state.get("last_sent_night_text") or self.state.get("last_sent_day_text")
        if last:
            parts.append(f"最近一条主动消息：{last[:40]}{'…' if len(last) > 40 else ''}")
        return "；".join(parts)

    async def preview(self, kind: str) -> str:
        """生成一条主动消息但不发送（用于 /憨憨主动测试 预览她会说什么）。"""
        return await self._generate_text(kind)

    # ---------- 调度 ----------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info("[hanhan] 主动消息调度已启动")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("[hanhan] 主动消息调度已停止")

    def _plan_day(self, date_str: str, cur: int) -> None:
        """为新的一天生成随机调度计划（按概率决定是否安排，只选窗口内的未来时刻）。"""
        self.state["plan_date"] = date_str
        self.state["night_minute"] = _pick_future_minute(NIGHT_WINDOW, cur, NIGHT_PROB)
        self.state["day_minute"] = _pick_future_minute(DAY_WINDOW, cur, DAY_PROB)
        self.state["sent_night"] = False
        self.state["sent_day"] = False
        self._save_state()
        logger.info(
            f"[hanhan] 今日主动计划: 深夜={'有' if self.state['night_minute'] is not None else '无'}, "
            f"白天={'有' if self.state['day_minute'] is not None else '无'}"
        )

    async def _loop(self) -> None:
        while True:
            try:
                now = datetime.now()
                if not self.state["active"] or not self.state["targets"]:
                    await asyncio.sleep(LOOP_INTERVAL)
                    continue
                today = now.strftime("%Y-%m-%d")
                cur = _cur_minute(now)
                if self.state["plan_date"] != today and now.hour >= 2:
                    self._plan_day(today, cur)
                if (
                    not self.state["sent_night"]
                    and self.state["night_minute"] is not None
                    and cur >= self.state["night_minute"]
                ):
                    await self._try_send("night")
                if (
                    not self.state["sent_day"]
                    and self.state["day_minute"] is not None
                    and cur >= self.state["day_minute"]
                ):
                    await self._try_send("day")
            except asyncio.CancelledError:
                raise
            except BaseException:
                logger.error(f"[hanhan] 主动调度循环异常: ", exc_info=True)
            await asyncio.sleep(LOOP_INTERVAL)

    # ---------- 发送 ----------

    async def _try_send(self, kind: str) -> None:
        """触发一次主动发送；失败累计重试，超过上限放弃（防止坏目标每 60s 刷错误）。"""
        ok = await self._send(kind)
        if ok:
            self.state[f"sent_{kind}"] = True
            self.state[f"last_{kind}_failures"] = 0
        else:
            failures = self.state.get(f"last_{kind}_failures", 0) + 1
            self.state[f"last_{kind}_failures"] = failures
            if failures >= MAX_SEND_RETRIES:
                self.state[f"sent_{kind}"] = True  # 放弃本轮，次日再重新计划
                logger.error(
                    f"[hanhan] 主动消息({kind})连续 {failures} 次发送失败，已放弃本轮重试"
                )
        self._save_state()

    async def _send(self, kind: str) -> bool:
        """生成并发送主动消息；任一目标送达即视为成功（失败的不再重复打扰已送达的目标）。"""
        text = await self._generate_text(kind)
        results = [
            await self._send_text(target, text)
            for target in list(self.state["targets"])
        ]
        self.state[f"last_{kind}_text"] = text
        if any(results):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.state[f"last_sent_{kind}_at"] = now_str
            self.state[f"last_sent_{kind}_text"] = text
            self.state[f"last_{kind}_error"] = ""
            self._save_state()
            return True
        failed = [t for t, ok in zip(self.state["targets"], results) if not ok]
        self.state[f"last_{kind}_error"] = f"全部 {len(failed)} 个目标均未送达"
        self._save_state()
        return False

    async def _generate_text(self, kind: str) -> str:
        """按人格生成主动消息；LLM 不可用或失败时用兜底句。"""
        if not self.persona_text:
            return random.choice(CANNED[kind])
        now_str = datetime.now().strftime("%H:%M")
        if kind == "night":
            task = (
                f"现在是深夜 {now_str}，你还没睡，想给他发条消息。"
                "就像真的在微信里给前任发消息：1-2 句，口语化，想到哪说到哪，"
                "可以说自己睡不着、刚洗完澡、或者今天发生的小事。"
                "禁忌：不要文艺腔、不要绕弯子玩梗（绝对不能说'有点想你写的排序算法'这类硬编的话）、"
                "不要'你最近怎么样''最近忙吗''在吗'这种客套开场，不要解释自己为什么发消息。"
                "情绪特别到位（想他、有点脆弱、被逗笑）时才用 [表情包:情绪词]，平时纯文字。"
                "直接输出消息内容，不要任何解释或前缀。"
            )
        else:
            task = (
                f"现在是白天 {now_str}，你刷到/想到个有意思的东西，随手分享给他。"
                "就像真的在微信里给人分享：1-2 句，口语化，可以吐槽、可以大笑，"
                "别用'刚刚看到一个视频''跟你说个事'这种开场白，别 AI 腔。"
                "分享视频/图片时直接文字描述内容即可，不要输出 [视频] 这类标记。"
                "情绪丰富时用一个 [表情包:情绪词]，一般纯文字就好。"
                "直接输出消息内容，不要解释。"
            )
        provider = None
        try:
            provider = self.context.get_using_provider(
                self.state["targets"][0] if self.state["targets"] else None
            )
        except BaseException as e:
            logger.error(f"[hanhan] 获取 LLM 提供商失败: {e}")
        if provider is None:
            return random.choice(CANNED[kind])
        try:
            resp = await provider.text_chat(
                prompt=task,
                system_prompt=self.persona_text or None,
            )
            text = self._response_to_text(resp)
            if text:
                return text
        except BaseException as e:
            logger.error(f"[hanhan] 主动消息生成失败: {e}")
        return random.choice(CANNED[kind])

    @staticmethod
    def _response_to_text(resp) -> str:
        """从 text_chat 返回值提取纯文本。

        v4 的 text_chat 返回 LLMResponse 对象（不是字符串）：
        优先 result_chain（消息链），其次 completion_text / _completion_text。
        """
        if isinstance(resp, str):
            return resp.strip()
        if resp is None:
            return ""
        chain = getattr(resp, "result_chain", None)
        if chain is not None:
            text = "".join(
                getattr(c, "text", "") for c in getattr(chain, "chain", [])
            )
            if text.strip():
                return text.strip()
        return (
            getattr(resp, "completion_text", None)
            or getattr(resp, "_completion_text", "")
            or ""
        ).strip()

    async def _send_text(self, target: str, text: str) -> bool:
        """按人格解析文本并逐条发送（分条 + 表情包）；返回该目标是否至少送达一条。"""
        parts = parse_reply(text)
        if not parts:
            logger.warning(f"[hanhan] 主动消息内容为空，跳过目标: {target}")
            return True  # 没有可发送的内容不算失败，避免空转重试
        sent_any = False
        for kind, payload in parts:
            if kind == "img":
                img = self.stickers.pick(target, payload)
                if img is None:
                    logger.warning(f"[hanhan] 主动消息表情包缺失: {payload!r}")
                    continue
                img = ensure_sendable(img)  # webp 转 png，保证微信图片通道可识别
                logger.info(f"[hanhan] 主动消息发送表情包: {img.name}")
                self.stickers.record_used(target, img.name)  # 与被动回复一致，防连发同一张
                ok = await self._safe_send(target, MessageChain().file_image(str(img)))
            else:
                logger.info(f"[hanhan] 主动消息发送文字: {payload[:20]}")
                ok = await self._safe_send(target, MessageChain().message(payload))
            sent_any = sent_any or ok
        return sent_any

    async def _safe_send(self, target: str, chain: MessageChain) -> bool:
        """向一个目标发送一条消息；会话格式非法或发送异常时记录日志并跳过，不影响其余目标。

        对应官方文档的主动消息接口：self.context.send_message(unified_msg_origin, chains)。
        返回 True 只表示找到了匹配的平台（适配器是否真正送达由平台实现决定）。
        """
        try:
            return bool(await self.context.send_message(target, chain))
        except ValueError as e:
            logger.warning(f"[hanhan] 主动消息目标会话格式非法，已跳过: {target} → {e}")
            return False
        except Exception as e:
            logger.warning(f"[hanhan] 主动消息发送异常，已跳过: {target} → {e}")
            return False
