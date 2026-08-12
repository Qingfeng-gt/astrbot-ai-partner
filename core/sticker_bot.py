"""表情包选择器：情绪词匹配 + 同义词兜底 + 防连续重复 + 发送频率控制。

匹配优先级：
1. 情绪词直接匹配文件名（如 [表情包:开心] → catbug-开心.webp）
2. 情绪词同义词表（如 "无语" 无直接命中时尝试 呕吼/思考中/不要）
3. 随机兜底
"""

import random
import time
from pathlib import Path
from typing import Optional

STICKER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# 情绪词 → 文件名关键词候选（按序尝试，第一个有命中的集合里随机抽）
EMOTION_TO_KEYWORDS: dict[str, list[str]] = {
    "开心": ["开心", "扭一扭", "哈哈"],
    "亲亲": ["亲亲", "亲脸", "偷偷给心"],
    "无语": ["呕吼", "思考中", "不要"],
    "生气": ["咬你", "吃掉你", "不准色色"],
    "害羞": ["偷偷看", "亲亲", "亲脸"],
    "疑问": ["思考中"],
    "难过": ["呕吼", "再见"],
    "敷衍": ["不要", "呕吼"],
    "思考中": ["思考中"],
}


class StickerBot:
    """从 stickers/ 目录挑选表情包。

    - pick(session_id, keyword)：按情绪匹配，每会话避免连续重复同一张
    - is_rate_limited(session_id)：限制单位时间窗口内的发送量（防刷屏）
    """

    def __init__(
        self,
        sticker_dir: Path,
        avoid_repeat: int = 2,
        rate_window: float = 600.0,
        rate_max: int = 3,
    ):
        self.sticker_dir = Path(sticker_dir)
        self.avoid_repeat = avoid_repeat
        self.rate_window = rate_window
        self.rate_max = rate_max
        self._recent_used: dict[str, list[str]] = {}  # 会话 -> 最近用过的文件名
        self._sent_times: dict[str, list[float]] = {}  # 会话 -> 发送时间戳

    def _all_files(self) -> list[Path]:
        try:
            return [
                p
                for p in self.sticker_dir.iterdir()
                if p.is_file() and p.suffix.lower() in STICKER_EXTS
            ]
        except OSError:
            return []

    @staticmethod
    def _candidates(files: list[Path], keyword: Optional[str]) -> list[Path]:
        """按关键词筛文件（文件名包含关键词，大小写不敏感）。"""
        if not keyword:
            return []
        kw = keyword.lower()
        return [p for p in files if kw in p.stem.lower()]

    def pick(self, session_id: str, keyword: Optional[str] = None) -> Optional[Path]:
        """按情绪词挑选表情包；匹配不到返回随机一张；目录为空返回 None。"""
        files = self._all_files()
        if not files:
            return None

        # 排除最近用过的，避免连发同一张
        recent = set(self._recent_used.get(session_id, [])[-self.avoid_repeat:])
        pool = [p for p in files if p.name not in recent] or files

        # 1) 情绪词直接匹配
        matched = self._candidates(pool, keyword)
        # 2) 同义词候选
        if not matched and keyword:
            for alt in EMOTION_TO_KEYWORDS.get(keyword, []):
                matched = self._candidates(pool, alt)
                if matched:
                    break
        # 3) 随机兜底
        if not matched:
            return random.choice(pool) if pool else None
        return random.choice(matched)

    def record_used(self, session_id: str, name: str) -> None:
        """记录本会话用过的表情包（用于避免重复）。"""
        self._recent_used.setdefault(session_id, []).append(name)

    def is_rate_limited(self, session_id: str) -> bool:
        """窗口时间内发送数达到上限则限流。"""
        now = time.time()
        times = [
            t for t in self._sent_times.get(session_id, []) if now - t < self.rate_window
        ]
        self._sent_times[session_id] = times
        return len(times) >= self.rate_max

    def record_sent(self, session_id: str) -> None:
        """记录一次表情包发送。"""
        self._sent_times.setdefault(session_id, []).append(time.time())
