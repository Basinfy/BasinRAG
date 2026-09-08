"""Extractive L0–L3 condensation. No LLM required on the live index path."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from ..core.ids import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def extract_l1(text: str, top_n: int = 8) -> str:
    counts = Counter(t for t in tokenize(text) if len(t) > 3)
    return ", ".join(w for w, _ in counts.most_common(top_n))


def extract_l2(text: str, max_chars: int = 280) -> str:
    parts = _SENTENCE.split((text or "").strip())
    first = parts[0].strip() if parts else (text or "").strip()
    if len(first) > max_chars:
        return first[: max_chars - 1] + "…"
    return first


def basin_l3_label(texts: List[str], source: str = "") -> str:
    blob = " ".join(texts[:8])
    keywords = extract_l1(blob, top_n=5)
    stem = source.replace("\\", "/").split("/")[-1] if source else "basin"
    if keywords:
        return f"{stem}: {keywords}"
    return stem


def node_layers(text: str) -> Dict[str, str]:
    return {
        "l1": extract_l1(text),
        "l2": extract_l2(text),
    }
