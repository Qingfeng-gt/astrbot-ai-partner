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


def ensure_sendable(path) -> "Path":
    """把可能不受微信图片通道支持的格式（webp）转为 png；失败时原样返回。

    转换产物缓存到原目录下 .media_cache/，避免重复转换。
    """
    p = Path(path)
    if p.suffix.lower() != ".webp":
        return p
    try:
        from PIL import Image as PILImage
    except ImportError:
        return p
    try:
        cache_dir = p.parent / ".media_cache"
        cache_dir.mkdir(exist_ok=True)
        out = cache_dir / f"{p.stem}.png"
        if not out.exists():
            with PILImage.open(p) as im:
                im.convert("RGBA").save(out, "PNG")
        return out
    except Exception:
        return p

# 情绪词 → 文件名关键词候选（按序尝试，第一个有命中的集合里随机抽）
# 关键词可混用 catbug 中文名与 blobcat 英文名（如 "cry" 命中 blobcatcry.png）
EMOTION_TO_KEYWORDS: dict[str, list[str]] = {
    "开心": ["开心", "扭一扭", "扭屁股", "摇摇晃晃", "wave", "melt"],
    "得意": ["棒", "点赞", "牛", "膜拜", "triumph"],
    "夸赞": ["棒", "点赞", "牛", "膜拜"],
    "爱你": ["爱你", "爱你2", "贴贴", "love", "kissheart"],
    "亲亲": ["亲亲", "亲脸", "偷偷给心", "kissheart"],
    "贴贴": ["贴贴", "爱你", "扭屁股", "snuggle"],
    "撒娇": ["扭屁股", "贴贴", "摇摇晃晃", "melt"],
    "无语": ["呕吼", "不要", "无奈", "吹风", "dead", "facepalm"],
    "无奈": ["无奈", "呕吼", "摇摇晃晃", "dead", "disturbed"],
    "尴尬": ["脸红", "facepalm", "无奈"],
    "生气": ["咬你", "吃掉你", "不准色色", "生气"],
    "害羞": ["脸红", "偷偷看", "亲亲", "亲脸"],
    "疑问": ["问号", "思考中", "umm"],
    "思考中": ["思考中", "问号", "think"],
    "难过": ["cry", "sadreach", "heartbroken", "再见", "呕吼"],
    "委屈": ["cry", "sadreach", "不要"],
    "害怕": ["scared", "shocked"],
    "震惊": ["shocked", "openmouth", "呕吼"],
    "惊讶": ["openmouth", "shocked", "问号"],
    "敷衍": ["不要", "呕吼", "无奈", "dead"],
    "紧张": ["sipsweat", "脸红", "偷偷看"],
    "困": ["睡觉", "dead"],
    "心碎": ["heartbroken", "cry"],
    "安慰": ["snuggle", "贴贴", "爱你"],
    "色色": ["和我色色", "舔屏", "不准色色"],
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
