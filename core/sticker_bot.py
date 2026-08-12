"""表情包选择器：多标签匹配 + 文本情绪推断 + 防连续重复 + 发送频率控制。

多标签匹配：
- 每个表情包有 0..N 个标签（情绪词）。标签来源：
  1. sticker_config.json 里手写的 tags（文件名 → 标签列表，最可靠，可覆盖自动推断）
  2. 自动推断：文件名包含情绪词或其关键词（如 catbug-开心.webp → 开心；blobcatcry.png → 难过）
- pick() 按"命中标签数"打分：请求情绪命中标签最多的表情包优先，同分随机
- 全部无命中时退回文件名关键词匹配（同义词表），再退回随机

频率控制：
- 限频（窗口内最多 N 张）+ 连续不重复同一张
- boost_prob：LLM 没标表情包时，回复补发一张的概率（由 main.py 调用）
"""

import random
import re
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

# 回复文本 → 情绪词（补发表情包时从文本推断情绪）
TEXT_EMOTION_RULES: list[tuple[str, str]] = [
    (r"哈哈|笑死|好耶|嘿嘿|乐死|太好啦|开心|高兴|爽死", "开心"),
    (r"好看|漂亮|可爱|好美|好帅|喜欢死|太好看", "夸赞"),
    (r"厉害|太强|牛|优秀|膜拜|佩服|好棒", "得意"),
    (r"无语|救命|绝了|什么鬼|服了|受不了|麻了|离大谱", "无语"),
    (r"唉|哎|难过|伤心|委屈|想哭|难受|破防|emo", "难过"),
    (r"气死|生气|烦死|可恶|气人", "生气"),
    (r"害羞|不好意思|脸红|羞死", "害羞"),
    (r"哇|震惊|竟然|居然|我的天|天哪|惊了", "震惊"),
    (r"想想|让我想想|思考|琢磨|纠结", "思考中"),
    (r"困|睡觉|晚安|熬不住|好累", "困"),
    (r"想你了|想你|亲亲|么么|抱抱|爱你", "爱你"),
    (r"？|啥|什么|为什么|真的吗|是这样吗", "疑问"),
]


def infer_emotion(text: str, rules: Optional[list] = None) -> Optional[str]:
    """从回复文本推断情绪词（用于 LLM 没标表情包时补发）；无命中返回 None。

    rules 可传入自定义规则列表 [(正则, 情绪词), ...]，缺省用模块级 TEXT_EMOTION_RULES。
    """
    for pattern, emotion in rules or TEXT_EMOTION_RULES:
        if re.search(pattern, text):
            return emotion
    return None


class StickerBot:
    """从 stickers/ 目录挑选表情包。

    - pick(session_id, keyword)：多标签匹配（命中数打分），每会话避免连续重复同一张
    - is_rate_limited(session_id)：限制单位时间窗口内的发送量（防刷屏）
    - tags：手写标签（文件名 → 标签列表），与自动推断合并使用
    """

    def __init__(
        self,
        sticker_dir: Path,
        avoid_repeat: int = 2,
        rate_window: float = 600.0,
        rate_max: int = 3,
        boost_prob: float = 0.3,
        tags: Optional[dict] = None,
        emotion_keywords: Optional[dict] = None,
        text_rules: Optional[list] = None,
    ):
        self.sticker_dir = Path(sticker_dir)
        self.avoid_repeat = avoid_repeat
        self.rate_window = rate_window
        self.rate_max = rate_max
        self.boost_prob = boost_prob
        # 情绪词 → 文件名关键词表（可整体替换以适配不同人格）
        self.emotion_keywords: dict[str, list[str]] = (
            dict(emotion_keywords) if emotion_keywords else dict(EMOTION_TO_KEYWORDS)
        )
        # 文本 → 情绪推断规则（正则列表，可整体替换）
        self.text_rules: list[tuple[str, str]] = (
            [(str(p), str(e)) for p, e in text_rules]
            if text_rules
            else list(TEXT_EMOTION_RULES)
        )
        # 手写标签：文件名(小写) -> 标签列表
        self.tags: dict[str, list[str]] = {
            str(k).lower(): [str(t).strip() for t in v if str(t).strip()]
            for k, v in (tags or {}).items()
        }
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

    def _auto_tags(self, p: Path) -> list[str]:
        """自动推断标签：文件名包含情绪词或其任一关键词即打上该情绪标签。"""
        stem = p.stem.lower()
        return [
            emo
            for emo, kws in self.emotion_keywords.items()
            if emo.lower() in stem or any(kw in stem for kw in kws)
        ]

    def tags_of(self, p: Path) -> list[str]:
        """表情包的完整标签：手写标签 + 自动推断（去重保序）。"""
        seen: list[str] = []
        for tag in [*self.tags.get(p.name.lower(), []), *self._auto_tags(p)]:
            if tag not in seen:
                seen.append(tag)
        return seen

    def _tag_stats(self) -> dict:
        """标签统计：每个标签覆盖的表情包数量（供页面展示）。"""
        stats: dict[str, int] = {}
        for p in self._all_files():
            for tag in self._auto_tags(p):
                stats[tag] = stats.get(tag, 0) + 1
        return stats

    def _tagged_count(self) -> int:
        """至少有一个标签的表情包数量（手写或自动均可）。"""
        return sum(1 for p in self._all_files() if self.tags_of(p))

    @staticmethod
    def _candidates(files: list[Path], keyword: Optional[str]) -> list[Path]:
        """按关键词筛文件（文件名包含关键词，大小写不敏感）。"""
        if not keyword:
            return []
        kw = keyword.lower()
        return [p for p in files if kw in p.stem.lower()]

    def pick(self, session_id: str, keyword: Optional[str] = None) -> Optional[Path]:
        """按情绪词挑选表情包（多标签命中数优先）；无命中随机；目录为空返回 None。"""
        files = self._all_files()
        if not files:
            return None

        # 排除最近用过的，避免连发同一张
        recent = set(self._recent_used.get(session_id, [])[-self.avoid_repeat:])
        pool = [p for p in files if p.name not in recent] or files

        if keyword and keyword.strip():
            kw = keyword.strip()
            # 1) 多标签匹配：命中标签数越多越优先，同分随机
            scored = [(self.tags_of(p).count(kw), p) for p in pool]
            best = max(s for s, _ in scored)
            if best > 0:
                top = [p for s, p in scored if s == best]
                return random.choice(top)
            # 2) 无命中：同义词关键词兜底（现有逻辑）
            for alt in self.emotion_keywords.get(kw, []):
                matched = self._candidates(pool, alt)
                if matched:
                    return random.choice(matched)
        # 3) 随机兜底
        return random.choice(pool) if pool else None

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
