"""Stable content-addressed node IDs so re-ingest does not duplicate the graph."""
from __future__ import annotations

import hashlib
import re


def make_node_id(source: str, chunk_index: int, text: str) -> str:
    payload = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


# Match Latin/Cyrillic words (including hyphenated scientific terms and alphanumeric codes) and individual CJK ideographs/syllables
_WORD = re.compile(
    r"[A-Za-zÀ-ÿ0-9_\u0400-\u04FF]+(?:[-_][A-Za-zÀ-ÿ0-9_\u0400-\u04FF]+)*|"
    r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for m in _WORD.finditer(text or ""):
        tok = m.group(0).lower()
        tokens.append(tok)
        # If token is hyphenated (e.g., 'sars-cov-2' or 'il-6'), also index constituent parts
        if "-" in tok:
            for part in tok.split("-"):
                if part and part != tok:
                    tokens.append(part)
    return tokens

