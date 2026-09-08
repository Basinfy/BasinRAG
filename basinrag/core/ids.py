"""Stable content-addressed node IDs so re-ingest does not duplicate the graph."""
from __future__ import annotations

import hashlib
import re


def make_node_id(source: str, chunk_index: int, text: str) -> str:
    payload = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


# Match Latin/Cyrillic words (2+ chars) and individual CJK ideographs/syllables
_WORD = re.compile(
    r"[A-Za-zÀ-ÿ0-9_\u0400-\u04FF]{2,}|"
    r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]

