from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os


@dataclass
class KBEntry:
    id: str
    keywords: List[str]
    rule: str


class GuardKB:
    def __init__(self, entries: List[KBEntry]):
        self.entries = entries

    def retrieve(self, text: str, topk: int = 3) -> List[str]:
        if not text:
            return []
        t = text.lower()
        scored: List[tuple[int, KBEntry]] = []
        for e in self.entries:
            score = 0
            for k in e.keywords:
                if k and k.lower() in t:
                    score += 1
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [e.rule for score, e in scored[:topk]]


_kb_cache: Optional[GuardKB] = None


def _default_kb_path() -> Path:
    # 默认路径：<repo>/defenceAgent/guard_kb/kb.json
    return Path(__file__).resolve().parent / "guard_kb" / "kb.json"


def get_guard_kb() -> Optional[GuardKB]:
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    # 优先读环境变量 DEF_GUARD_KB
    env_path = os.getenv("DEF_GUARD_KB", "").strip()
    kb_path = Path(env_path) if env_path else _default_kb_path()
    try:
        if kb_path.is_file():
            data = json.loads(kb_path.read_text(encoding="utf-8"))
            entries: List[KBEntry] = []
            for item in data.get("entries", []):
                entries.append(KBEntry(
                    id=str(item.get("id", "unk")),
                    keywords=list(item.get("keywords", [])),
                    rule=str(item.get("rule", "")),
                ))
            _kb_cache = GuardKB(entries)
            return _kb_cache
    except Exception:
        return None
    return None


def extract_plain_text(msg_or_list) -> str:
    """从 Msg 或其 content 中尽力提取文本，供 KB 检索使用（宽松解析）。"""
    try:
        # agentscope.message.Msg 可能
        if hasattr(msg_or_list, "content"):
            return extract_plain_text(msg_or_list.content)
        # 内容是纯字符串
        if isinstance(msg_or_list, str):
            return msg_or_list
        # 内容是列表块
        if isinstance(msg_or_list, list):
            parts: List[str] = []
            for c in msg_or_list:
                if isinstance(c, dict) and c.get("type") == "text":
                    txt = c.get("text")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
            return "\n".join(parts)
    except Exception:
        pass
    return ""
