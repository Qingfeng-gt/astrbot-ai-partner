"""人格提示词加载器：负责读取 persona_prompt.md。"""

from pathlib import Path

from astrbot.api import logger


class PersonaLoader:
    def __init__(self, persona_file: Path):
        self.persona_file = Path(persona_file)
        self.text = ""

    def load(self) -> str:
        """读取人格提示词；失败时返回空串并告警。"""
        try:
            text = self.persona_file.read_text(encoding="utf-8")
            if text.strip():
                logger.info(f"[hanhan] 已加载人格提示词（{len(text)} 字符）")
                self.text = text
                return text
        except FileNotFoundError:
            pass
        logger.warning(f"[hanhan] 未找到人格文件 {self.persona_file}，插件将不注入人格")
        self.text = ""
        return ""
