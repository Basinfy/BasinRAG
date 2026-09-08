"""Lexical seed projector — entity/term overlap."""
from __future__ import annotations

from typing import List

from ..core.ids import tokenize
from ..indexer.bm25 import _STOP


def seed_node_ids(engine, query: str, max_k: int = 6) -> List[str]:
    terms = [t for t in tokenize(query) if len(t) > 3 and t not in _STOP]
    if not terms or engine.graph.number_of_nodes() == 0:
        return []
    scored = []
    for nid, data in engine.graph.nodes(data=True):
        blob = f"{data.get('l1', '')} {data.get('text', '')[:500]}"
        tokens = set(tokenize(blob))
        hit = sum(1 for t in terms if t in tokens)
        if hit:
            hops = int(data.get("hops", 0))
            scored.append((hit, -hops, nid))
    scored.sort(reverse=True)
    return [nid for _, _, nid in scored[:max_k]]
