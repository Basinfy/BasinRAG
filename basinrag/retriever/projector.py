"""Lexical seed projector — entity/term overlap."""
from __future__ import annotations

from typing import List

from ..core.ids import tokenize
from ..indexer.bm25 import _STOP


def seed_node_ids(engine, query: str, max_k: int = 6) -> List[str]:
    terms = [t for t in tokenize(query) if len(t) > 3 and t not in _STOP]
    if not terms or engine.graph.number_of_nodes() == 0:
        return []

    bm25 = getattr(engine, "bm25", None)
    if bm25 is not None and getattr(bm25, "n", 0) > 0 and hasattr(bm25, "postings"):
        from ..indexer.bm25 import stem_token
        stemmed_terms = [stem_token(t) for t in terms]
        doc_hits = {}
        for st in stemmed_terms:
            if st in bm25.postings:
                for doc_idx, count in bm25.postings[st]:
                    doc_hits[doc_idx] = doc_hits.get(doc_idx, 0) + 1
        if doc_hits:
            scored = []
            for doc_idx, hits in doc_hits.items():
                if doc_idx < len(bm25.doc_ids):
                    nid = bm25.doc_ids[doc_idx]
                    if nid in engine.graph:
                        hops = int(engine.graph.nodes[nid].get("hops", 0))
                        scored.append((hits, -hops, nid))
            scored.sort(reverse=True)
            return [nid for _, _, nid in scored[:max_k]]

    # Fallback se BM25 ainda não estiver disponível
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

