"""Stable content-addressed node IDs so re-ingest does not duplicate the graph."""
from __future__ import annotations

import hashlib
import re


def make_node_id(source: str, chunk_index: int, text: str) -> str:
    payload = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]
