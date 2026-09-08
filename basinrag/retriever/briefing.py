"""Token-budgeted briefing: L0 hubs, L1/L2 neighbors, L3 satellites."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

MIN_CONFIDENCE = 0.15
SATELLITE_CHARS = 400
MAX_SATELLITES = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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

    def as_context(self, max_tokens: int = 2000) -> str:
        chunks: List[str] = []
        used = 0
        sats = [cap_satellite(s) for s in self.satellites[:MAX_SATELLITES] if s]
        if sats:
            sat = " | ".join(sats)
            chunks.append(f"[Satellites]\n{sat}")
            used += estimate_tokens(sat)
        for label, items in (("Hubs", self.hubs), ("Neighbors", self.neighbors)):
            body: List[str] = []
            for item in items:
                cost = estimate_tokens(item)
                if used + cost > max_tokens:
                    break
                body.append(item)
                used += cost
            if body:
                chunks.append(f"[{label}]\n" + "\n\n".join(body))
        return "\n\n".join(chunks)
