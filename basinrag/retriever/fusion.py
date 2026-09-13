"""Weighted RRF (α=0.15, k=60) fused on node ids, with optional experimental signals."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# Dense-led fusion: BM25 is a light lexical vote inside the dense pool.
HYBRID_ALPHA = 0.15
RRF_K = 60
HOP_LAMBDA = 0.35
# Prefer neutral for production recall: unreachable lexical hits keep seed-tier weight.
DEFAULT_HOP_MISSING = "neutral"

# Multi-signal DRF weights (calibrated from BasinMind architecture)
W_BM25 = 0.40
W_HOPS = 0.20
W_COHESION = 0.15
W_CENTRALITY = 0.15
W_MEMORY = 0.10


def tanh_soft_brake(x: float) -> float:
    """Bound a fusion score with a smooth tanh curve."""
    return 0.5 + 0.5 * math.tanh(x - 0.5)


def weighted_rrf(
    bm25_ids: Sequence[str],
    semantic_ids: Sequence[str],
    alpha: float = HYBRID_ALPHA,
    k: int = RRF_K,
    max_depth: int = 500,
    bm25_allowlist: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion between BM25 and semantic rankings.

    When ``bm25_allowlist`` is set, BM25 votes only for those ids. Lexical-only
    outsiders cannot enter the fused ranking; they were the SciFact drop vs dense.
    """
    allowed = None if bm25_allowlist is None else set(bm25_allowlist)
    scores: Dict[str, float] = {}
    for rank, nid in enumerate(bm25_ids[:max_depth]):
        if allowed is not None and nid not in allowed:
            continue
        scores[nid] = scores.get(nid, 0.0) + alpha / (k + rank + 1)
    for rank, nid in enumerate(semantic_ids[:max_depth]):
        scores[nid] = scores.get(nid, 0.0) + (1.0 - alpha) / (k + rank + 1)
    return scores


def apply_hop_prior(
    scores: Dict[str, float],
    hops: Dict[str, int],
    lam: float = HOP_LAMBDA,
    enabled: bool = True,
    missing: str = DEFAULT_HOP_MISSING,
) -> Dict[str, float]:
    """Scale fused scores by hop distance.

    ``missing="penalty" treats nodes the BFS never reached as farther than
    any observed hop (gate SciFact control). ``missing="neutral" (default)
    awards unreachable nodes the same floor as hop-0 so isolated BM25/dense
    hits are not expelled from the top-k.
    """
    if not enabled:
        return dict(scores)
    max_known = max(hops.values(), default=0) if hops else 0
    missing_h = max_known + 1 if missing == "penalty" else 0
    out = {}
    for nid, s in scores.items():
        h = hops[nid] if nid in hops else missing_h
        # Floor at 0.7 so a BM25 hit at the end of a section is not erased.
        out[nid] = s * (0.7 + 0.3 * math.exp(-h * lam))
    return out


def multi_signal_drf(
    rrf_scores: Dict[str, float],
    hops: Dict[str, int],
    cohesion: Optional[Dict[str, float]] = None,
    centrality: Optional[Dict[str, float]] = None,
    memory: Optional[Dict[str, float]] = None,
    enabled: bool = False,
) -> Dict[str, float]:
    """Multi-Signal Deterministic Rank Fusion with Bakhshali Brake.

    Experimental aggregation of RRF and available topology signals.
    Unavailable signals are omitted and the remaining weights are normalized.
    This score is not calibrated as a probability and is disabled by default.
    """
    if not enabled:
        return dict(rrf_scores)
    cohesion = cohesion or {}
    centrality = centrality or {}
    memory = memory or {}
    out = {}
    max_base = max(rrf_scores.values(), default=1.0) or 1.0
    for nid, base_score in rrf_scores.items():
        norm_base = base_score / max_base
        weighted = [(W_BM25, norm_base)]
        if nid in hops:
            weighted.append((W_HOPS, math.exp(-HOP_LAMBDA * max(0, hops[nid]))))
        if nid in cohesion:
            weighted.append((W_COHESION, cohesion[nid]))
        if nid in centrality:
            weighted.append((W_CENTRALITY, centrality[nid]))
        if nid in memory:
            weighted.append((W_MEMORY, memory[nid]))
        weight_sum = sum(weight for weight, _ in weighted)
        raw = sum(weight * signal for weight, signal in weighted) / weight_sum
        out[nid] = tanh_soft_brake(raw)
    return out


def ranked_ids(scores: Dict[str, float], top_k: int) -> List[str]:
    """Return top_k node IDs sorted by descending score."""
    return [nid for nid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
