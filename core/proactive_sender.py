"""主动消息调度：深夜语音 / 白天分享，随机时段，按人格主动联系用户。

- 深夜窗口（23:00 ~ 次日 01:30）随机选一个时刻，以"深夜语音"的形式主动发消息
- 白天窗口（12:00 ~ 22:00）随机选一个时刻，主动分享一条
- 消息内容由 LLM 按人格生成（失败时用内置兜底句）
- 目标会话自动记录（用户私聊过即记住），也可用 /憨憨绑定 手动指定
- 状态持久化到插件目录 proactive_state.json，重启不丢
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
from .sticker_bot import StickerBot

# 深夜窗口：23:00 ~ 次日 01:30（用跨天的绝对分钟数表示，1440 = 次日 00:00）
NIGHT_WINDOW = (23 * 60, 25 * 60 + 30)
# 白天分享窗口：12:00 ~ 22:00
DAY_WINDOW = (12 * 60, 22 * 60)
NIGHT_PROB = 0.3  # 每晚主动联系的概率（低频，回避型人设不会天天找）
DAY_PROB = 0.1  # 每天白天主动分享的概率
LOOP_INTERVAL = 60  # 调度检查间隔（秒）

# 兜底句（LLM 生成失败时使用），保持人格安全
CANNED = {
    "night": [
        "睡了吗\n我睡不着。",
        "刚洗完澡，躺床上发呆\n你干嘛呢",
        "做了一个奇怪的梦\n梦到以前上算法课的时候",
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
            if self.state["sent_night"]:
                at = self.state.get("last_sent_night_at", "?")
                parts.append(f"今晚深夜消息已发送（{at}）")
            else:
                parts.append(f"今晚深夜计划 ~{hh:02d}:{mm:02d} 触发")
        else:
            parts.append("今晚深夜无计划（低频随机）")
        if self.state["day_minute"] is not None:
            if self.state["sent_day"]:
                at = self.state.get("last_sent_day_at", "?")
                parts.append(f"今天白天消息已发送（{at}）")
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

    def _plan_day(self, date_str: str) -> None:
        """为新的一天生成随机调度计划（按概率决定是否安排）。"""
        self.state["plan_date"] = date_str
        self.state["night_minute"] = (
            random.randint(*NIGHT_WINDOW) if random.random() < NIGHT_PROB else None
        )
        self.state["day_minute"] = (
            random.randint(*DAY_WINDOW) if random.random() < DAY_PROB else None
        )
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
                if self.state["plan_date"] != today and now.hour >= 2:
                    self._plan_day(today)
                cur = _cur_minute(now)
                if (
                    not self.state["sent_night"]
                    and self.state["night_minute"] is not None
                    and cur >= self.state["night_minute"]
                ):
                    await self._send("night")
                    self.state["sent_night"] = True
                    self._save_state()
                if (
                    not self.state["sent_day"]
                    and self.state["day_minute"] is not None
                    and cur >= self.state["day_minute"]
                ):
                    await self._send("day")
                    self.state["sent_day"] = True
                    self._save_state()
            except asyncio.CancelledError:
                raise
            except BaseException:
                logger.error(f"[hanhan] 主动调度循环异常: ", exc_info=True)
            await asyncio.sleep(LOOP_INTERVAL)

    # ---------- 发送 ----------

    async def _send(self, kind: str) -> None:
        text = await self._generate_text(kind)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.state[f"last_sent_{kind}_at"] = now_str
        self.state[f"last_sent_{kind}_text"] = text
        self._save_state()
        for target in list(self.state["targets"]):
            await self._send_text(target, text)

    async def _generate_text(self, kind: str) -> str:
        """按人格生成主动消息；LLM 不可用或失败时用兜底句。"""
        if not self.persona_text:
            return random.choice(CANNED[kind])
        now_str = datetime.now().strftime("%H:%M")
        if kind == "night":
            task = (
                f"现在是深夜 {now_str}。你睡不着（或刚洗完澡、或突然想起他），想给他发条消息。"
                "用文字发，不用刻意装成语音；1-2 条短消息，迂回、不加句号。"
                "情绪特别到位（想他、有点脆弱、被逗笑）时才用 [表情包:情绪词]，平时纯文字就好。"
                "直接输出消息内容，不要任何解释或前缀。"
            )
        else:
            task = (
                f"现在是白天 {now_str}。你看到/想到个东西（视频、趣事、或者找工作的事），主动给他分享一条。"
                "短、口语化、不加句号。情绪丰富时用一个 [表情包:情绪词]，一般纯文字就好。"
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
            text = (resp or "").strip()
            if text:
                return text
        except BaseException as e:
            logger.error(f"[hanhan] 主动消息生成失败: {e}")
        return random.choice(CANNED[kind])

    async def _send_text(self, target: str, text: str) -> None:
        """按人格解析文本并逐条发送（分条 + 表情包）。"""
        for kind, payload in parse_reply(text):
            if kind == "img":
                img = self.stickers.pick(target, payload)
                if img is None:
                    continue
                ok = await self.context.send_message(
                    target, MessageChain().file_image(str(img))
                )
            else:
                ok = await self.context.send_message(target, MessageChain().message(payload))
            if not ok:
                logger.warning(f"[hanhan] 主动消息发送失败（平台未找到）: {target}")
                return
