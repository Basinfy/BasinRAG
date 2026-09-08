"""Weighted RRF (α=0.55, k=60) fused on node ids, plus hop prior."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

HYBRID_ALPHA = 0.55
RRF_K = 60
HOP_LAMBDA = 0.35


def weighted_rrf(
    bm25_ids: Sequence[str],
    semantic_ids: Sequence[str],
    alpha: float = HYBRID_ALPHA,
    k: int = RRF_K,
    max_depth: int = 500,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, nid in enumerate(bm25_ids[:max_depth]):
        scores[nid] = scores.get(nid, 0.0) + alpha / (k + rank + 1)
    for rank, nid in enumerate(semantic_ids[:max_depth]):
        scores[nid] = scores.get(nid, 0.0) + (1.0 - alpha) / (k + rank + 1)
    return scores


def apply_hop_prior(
    scores: Dict[str, float],
    hops: Dict[str, int],
    lam: float = HOP_LAMBDA,
) -> Dict[str, float]:
    out = {}
    for nid, s in scores.items():
        h = hops.get(nid, 0)
        # Floor at 0.7 so a BM25 hit at the end of a section is not erased.
        out[nid] = s * (0.7 + 0.3 * math.exp(-h * lam))
    return out


def ranked_ids(scores: Dict[str, float], top_k: int) -> List[str]:
    return [nid for nid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
