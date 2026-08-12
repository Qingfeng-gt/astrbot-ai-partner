"""会话情景感知：时间间隔、话题突变、遗忘、忙碌状态。

在每个 LLM 请求前生成一段【情景感知】注入提示词，让模型感知：
- 距上次对话多久了（隔得越久，语气越淡）
- 她说过去忙/睡了（回复节奏变慢）
- 话题突变（先疑惑，反应过来了再聊）
- 上下文被截断（更早的事记不清了，即"遗忘"）
"""

import re
import time
from datetime import datetime
from typing import Optional

# 寒暄/短应答不算话题突变
_GREETINGS = {"在吗", "在", "嗯", "哦", "好", "行", "哈哈", "诶", "喂", "嗨", "hello", "hi", "？", "?"}
# 她说"要去忙/睡"的常见说法
_BUSY_RE = re.compile(
    r"(先去|去忙|要忙|去学|要学|去睡|要睡|睡了|先睡|去洗澡|洗澡了|去吃饭|吃饭了|"
    r"上课|去.{0,2}图书馆|写作业|要复习|去复习|去看书|要背|熄灯了)"
)
# "陪我去图书馆"是邀约不是去忙，需要排除
_INVITE_RE = re.compile(r"陪.{0,4}去.{0,2}图书馆")
_MAX_CONTEXTS = 30  # 保留最近 N 条上下文，更早的视为遗忘
_TOPIC_SHIFT_MIN_OVERLAP = 0.2  # 与上一条消息字符重叠低于此值视为话题突变
_TOPIC_SHIFT_MAX_GAP = 600.0  # 距上一条消息 10 分钟内才算"突然"换话题
_BUSY_WINDOW = 7200.0  # "去忙"状态持续 2 小时

# 时间线表解析：时间节点不写死在代码里，由人格提示词的【时间线】章节驱动。
# 支持的行格式：
#   | 2026-08 | 描述 |                    单月
#   | 2026-09 ~ 2026-11 | 描述 |          区间（~ 或 至）
#   | 2026-08 之前 | 描述 |               开区间（以前/之前）
#   | 2027-04 及以后 | 描述 |             开区间（以后/及以后/之后）
_TIME_ROW_RE = re.compile(
    r"^\|\s*(\d{4})-(\d{2})([^|]*?)\|\s*(.+?)\s*\|",
    re.MULTILINE,
)
_OPEN_START = ("以前", "之前")
_OPEN_END = ("以后", "及以后", "之后")
# 人格提示词里没有时间线表时的兜底
_DEFAULT_STAGE = "按你人格设定里的既有身份说话，身份时点不确定"


def extract_timeline(
    persona_text: str,
) -> list[tuple[tuple[int, int], Optional[tuple[int, int]], str, str]]:
    """解析人格提示词【时间线】表。

    返回 [(起始年月, 结束年月或 None, 类型, 描述)]，
    类型：month / range / before / after。
    """
    rows: list[tuple[tuple[int, int], Optional[tuple[int, int]], str, str]] = []
    for m in _TIME_ROW_RE.finditer(persona_text or ""):
        start = (int(m.group(1)), int(m.group(2)))
        rest = m.group(3)
        desc = m.group(4).strip()
        end, kind = None, "month"
        date2 = re.search(r"(\d{4})-(\d{2})", rest)
        if date2:
            end, kind = (int(date2.group(1)), int(date2.group(2))), "range"
        elif any(k in rest for k in _OPEN_START):
            kind = "before"
        elif any(k in rest for k in _OPEN_END):
            kind = "after"
        rows.append((start, end, kind, desc))
    return rows


def _match_stage(
    now_ym: tuple[int, int],
    rows: list[tuple[tuple[int, int], Optional[tuple[int, int]], str, str]],
) -> str:
    """当前年月在时间表中匹配身份阶段；无匹配取第一行。"""
    for start, end, kind, desc in rows:
        if kind == "month" and now_ym == start:
            return desc
        if kind == "range" and start <= now_ym <= end:
            return desc
        if kind == "before" and now_ym < start:
            return desc
        if kind == "after" and now_ym >= start:
            return desc
    return rows[0][3] if rows else _DEFAULT_STAGE


class MemoryEngine:
    """记录每个会话的交互时间与内容，生成情景感知提示。"""

    def __init__(self, persona_text: str = ""):
        # 人格提示词文本：其中的【时间线】表是身份阶段推算的唯一依据（不写死）
        self.persona_text = persona_text or ""
        self.last_user_msg: dict[str, str] = {}
        self.last_user_time: dict[str, float] = {}
        self.last_reply_time: dict[str, float] = {}
        self.busy_until: dict[str, float] = {}

    def _current_stage(self, now_dt: datetime) -> str:
        """取当前时间与人格提示词时间表比较，得到她当下的身份阶段。"""
        rows = extract_timeline(self.persona_text)
        if not rows:
            return ""
        return _match_stage((now_dt.year, now_dt.month), rows)

    def on_user_message(self, session_id: str, text: str) -> None:
        self.last_user_msg[session_id] = text
        self.last_user_time[session_id] = time.time()

    def on_reply(self, session_id: str, text: str) -> None:
        """记录回复；若她说要去忙/睡了（排除"陪我去图书馆"这类邀约），
        接下来一段时间回复节奏变慢。"""
        self.last_reply_time[session_id] = time.time()
        if _BUSY_RE.search(text) and not _INVITE_RE.search(text):
            self.busy_until[session_id] = time.time() + _BUSY_WINDOW

    def trim_contexts(self, contexts: list) -> tuple[list, bool]:
        """截断过长上下文（遗忘机制）；返回 (截断后列表, 是否发生截断)。"""
        if len(contexts) <= _MAX_CONTEXTS:
            return contexts, False
        return contexts[-_MAX_CONTEXTS:], True

    def _topic_shift(self, session_id: str, current: str) -> bool:
        """判断用户是否突然换了话题：与上一条消息重叠低且在短时间内。"""
        prev = self.last_user_msg.get(session_id, "")
        if not prev:
            return False
        if time.time() - self.last_user_time.get(session_id, 0) > _TOPIC_SHIFT_MAX_GAP:
            return False  # 隔太久不算"突然"
        if current.strip() in _GREETINGS or len(current) <= 4:
            return False  # 寒暄/短应答
        a, b = set(prev), set(current)
        if not a or not b:
            return False
        overlap = len(a & b) / min(len(a), len(b))
        return overlap < _TOPIC_SHIFT_MIN_OVERLAP

    def build_situation(
        self, session_id: str, current: str, trimmed: bool
    ) -> str:
        """生成【情景感知】注入文本。

        始终包含：当前时间、当前阶段（按日期推算）、时段提示；
        按需包含：时间间隔、话题突变、遗忘提示。
        """
        now = time.time()
        now_dt = datetime.now()
        parts: list[str] = [f"当前时间：{now_dt.strftime('%Y-%m-%d %H:%M')}"]
        stage = self._current_stage(now_dt)
        if stage:
            parts.append(f"当前阶段：{stage}")

        # 时段提示：让回复贴合当下的她
        hour = now_dt.hour
        if hour >= 23 or hour < 5:
            parts.append("现在是深夜——你放下防备、话多的时段，可以说心里话、发语音式的话")
        elif hour < 18:
            parts.append("现在是白天——你在实习/投简历/刷题，回消息是忙里偷闲")

        # 时间间隔：距她上次回复/上次对方消息
        gap = None
        last_reply = self.last_reply_time.get(session_id)
        last_user = self.last_user_time.get(session_id)
        if last_reply is not None:
            gap = now - last_reply
        elif last_user is not None:
            gap = now - last_user
        if gap is not None:
            if gap > 12 * 3600:
                parts.append(
                    f"- 你们隔了很久没说话了（{self._fmt_gap(gap)}）。你看到消息时淡淡的，"
                    "像刚想起来回，不会热情，不会装作热络。"
                )
            elif gap > 2 * 3600:
                parts.append(
                    f"- 隔了挺久才又联系（{self._fmt_gap(gap)}）。你的语气不会太热络。"
                )
            elif (
                gap > 180
                and session_id in self.busy_until
                and now < self.busy_until[session_id]
            ):
                parts.append("- 你刚才说要去忙/睡了，回消息会按自己的节奏来，不着急秒回。")

        # 话题突变
        if self._topic_shift(session_id, current):
            parts.append(
                "- 他刚才的话题跟之前完全接不上，你有点没反应过来：先表达疑惑"
                "（“啊？”“啥？”或发个疑问/思考中的表情包），反应过来了再聊新话题。"
            )

        # 遗忘
        if trimmed:
            parts.append(
                "- 你们更早的聊天内容你已经记不太清了，不要主动提起那些细节，"
                "被问起时模糊、不确定。"
            )

        return "【情景感知】（仅本轮生效，用于判断当前气氛）：\n" + "\n".join(parts)

    @staticmethod
    def _fmt_gap(seconds: float) -> str:
        if seconds >= 86400:
            return f"{int(seconds // 86400)}天"
        if seconds >= 3600:
            return f"{int(seconds // 3600)}小时"
        return f"{int(seconds // 60)}分钟"
