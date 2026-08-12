"""图片识别：调用百炼（阿里云 MaaS）OpenAI 兼容多模态端点，让憨憨"看到"用户发的图片。

- 主对话仍走现有 LLM（人设不变）；这里只负责把图片翻译成文字描述
- 识别结果由 main.py 注入 system_prompt，她据此自然回应
- 支持图片来源：网络 URL / base64:// / 本地文件路径（转 base64 data URI）
- 失败静默返回空串，不阻塞主流程（没配 key 或识别失败就当没看到图）
"""

import asyncio
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from astrbot.api import logger

# API Key 优先从环境变量读取（不进代码/仓库）；端点可被插件配置覆盖
_ENV_API_KEY = "HANHAN_BAILIAN_API_KEY"
# 用户专属推理服务（MaaS）的 OpenAI 兼容端点，可被插件配置覆盖
_DEFAULT_ENDPOINT = (
    "https://llm-pb05pratj1qqv8yw.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
_MAX_IMAGES = 3  # 单轮最多识别张数
_TIMEOUT = 30.0

_DESCRIBE_PROMPT = (
    "你是憨憨的眼睛。用户刚刚给憨憨发了一张图片，请客观、简短地描述图片内容："
    "画面主体、场景、图片里的文字（如果有）。200 字以内，只描述，不评价，"
    "不要揣测发图人的意图。"
)


class VisionEngine:
    """百炼多模态识别引擎（配置为空时自动关闭）。"""

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "",
        key_file: str = "",
        max_images: int = _MAX_IMAGES,
        timeout: float = _TIMEOUT,
        describe_prompt: str = _DESCRIBE_PROMPT,
    ):
        """key 读取优先级：环境变量 > 插件配置 > 插件目录下的 key 文件（bailian.key）。

        key 文件内容为纯文本 key（自动 strip），不进代码/仓库。
        max_images/timeout/describe_prompt 可由 behavior_config.json vision 段覆盖。
        """
        file_key = ""
        if key_file:
            try:
                file_key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError:
                pass
        self.api_key = (
            os.environ.get(_ENV_API_KEY) or api_key or file_key or ""
        ).strip()
        self.endpoint = (endpoint or _DEFAULT_ENDPOINT).strip().rstrip("/")
        self.model = (model or "").strip()
        self.max_images = int(max_images)
        self.timeout = float(timeout)
        self.describe_prompt = str(describe_prompt)

    def enabled(self) -> bool:
        """是否已配置（有 key 即启用）。"""
        return bool(self.api_key)

    @staticmethod
    def _image_data_url(comp) -> Optional[str]:
        """把消息链里的 Image 组件转成百炼可用的 URL / data URI。"""
        file = getattr(comp, "file", "") or ""
        url = getattr(comp, "url", "") or ""
        path = getattr(comp, "path", "") or ""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if file.startswith("http://") or file.startswith("https://"):
            return file
        if file.startswith("base64://"):
            return "data:image/png;base64," + file[len("base64://"):]
        # 本地文件：优先 path；否则去掉 file:// 前缀
        local = path
        if not local and file.startswith("file://"):
            local = file[len("file://"):]
        if not local:
            return None
        try:
            data = Path(local).read_bytes()
            ext = Path(local).suffix.lower().lstrip(".") or "png"
            if ext not in ("png", "jpg", "jpeg", "webp", "gif"):
                ext = "png"
            return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
        except OSError as e:
            logger.warning(f"[hanhan] 图片本地文件读取失败: {local} → {e}")
            return None

    async def describe(self, images: list) -> str:
        """识别一组图片，返回简短描述；任何失败都返回空串（不阻塞）。"""
        urls = []
        for comp in images[: self.max_images]:
            u = self._image_data_url(comp)
            if u:
                urls.append(u)
        if not urls:
            return ""
        payload = {
            "model": self.model or "qwen-vl-plus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": u}} for u in urls
                    ]
                    + [{"type": "text", "text": self.describe_prompt}],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "astrbot-plugin-hanhan/1.0",
            },
            method="POST",
        )
        try:
            raw = await asyncio.to_thread(self._post, req)  # urllib 是阻塞的，丢线程池
            text = self._extract(raw)
            if text:
                logger.info(f"[hanhan] 图片识别成功: {text[:50]}…")
            return text
        except Exception as e:
            logger.warning(f"[hanhan] 图片识别失败（本轮按没看到图处理）: {e}")
            return ""

    def _post(self, req: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            # 把响应体带出来，便于定位 401/404/模型名等问题
            body = e.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"HTTP {e.code}: {body}") from e

    @staticmethod
    def _extract(raw: str) -> str:
        data = json.loads(raw)
        return (data["choices"][0]["message"]["content"] or "").strip()
