"""Token-budgeted briefing: L0 hubs, L1/L2 neighbors, L3 satellites."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

MIN_CONFIDENCE = 0.15
SATELLITE_CHARS = 400
MAX_SATELLITES = 4


def estimate_tokens(text: str) -> int:
    """Conservative script-aware fallback when a generator tokenizer is unavailable."""
    if not text:
        return 0
    def is_cjk(char):
        return (
            "\u3040" <= char <= "\u30ff"
            or "\u3400" <= char <= "\u9fff"
            or "\uac00" <= char <= "\ud7af"
        )

    cjk = sum(1 for char in text if is_cjk(char))
    other = sum(1 for char in text if not char.isspace() and not is_cjk(char))
    return cjk + max(0, (other + 1) // 2)


def cap_satellite(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= SATELLITE_CHARS:
        return text
    return text[:SATELLITE_CHARS].rstrip()


@dataclass
class BriefingPacket:
    hubs: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    satellites: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def texts_for_rerank(self) -> List[str]:
        """Raw passages only — never mix L3 headers into the cross-encoder."""
        return list(self.hubs) + list(self.neighbors)

    def as_context(self, max_tokens: int = 2000, token_counter=None) -> str:
        chunks: List[str] = []
        count = token_counter or estimate_tokens
        for label, items in (("Hubs", self.hubs), ("Neighbors", self.neighbors), ("Satellites", self.satellites[:MAX_SATELLITES])):
            body: List[str] = []
            for item in items:
                rendered = f"[{label}]\n" + "\n\n".join(body + [cap_satellite(item) if label == "Satellites" else item])
                trial = "\n\n".join(chunks + [rendered])
                if count(trial) > max_tokens:
                    break
                body.append(cap_satellite(item) if label == "Satellites" else item)
            if body:
                chunks.append(f"[{label}]\n" + "\n\n".join(body))
        return "\n\n".join(chunks)
