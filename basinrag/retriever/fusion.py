"""Weighted RRF fused on node ids, with optional experimental signals.

Flat indexes (SciFact-style, one node per document) use α=0.15 and a dense
allowlist. Chunked long-doc indexes use α=0.40 and may union BM25@10 documents
into RRF membership; SciFact never does.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence

# Dense-led fusion: BM25 is a light lexical vote inside the dense pool.
HYBRID_ALPHA = 0.15
LONGDOC_HYBRID_ALPHA = 0.40
LONGDOC_CANDIDATE_K_MIN = 200
LONGDOC_CANDIDATE_K_MULT = 10
LONGDOC_RESCUE_M = 2
LONGDOC_RESCUE_FROM = 80
LONGDOC_RESCUE_SLOT = 2
LONGDOC_UNION_BM25_DOCS = 10
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


def index_is_flat(engine: Any) -> bool:
    """True when each document is a single node (SciFact / MTEB control).

    Same geometry as the MTEB wrapper: every ``chunk_index`` is 0 and basins
    are ~1:1 with nodes. Chunked long-doc indexes are not flat.
    """
    graph = getattr(engine, "graph", None)
    if graph is None:
        return True
    n = graph.number_of_nodes()
    if n == 0:
        return True
    chunk_indexes = set()
    for _, data in graph.nodes(data=True):
        chunk_indexes.add(int(data.get("chunk_index", 0) or 0))
        if len(chunk_indexes) > 1:
            return False
    if chunk_indexes != {0}:
        return False
    n_basins = len(getattr(engine, "basins", None) or {})
    return n_basins >= max(1, int(0.9 * n))


def resolve_hybrid_alpha(engine: Any, alpha: Optional[float] = None) -> float:
    """Return the RRF α for this index. Explicit ``alpha`` wins over geometry."""
    if alpha is not None:
        return float(alpha)
    return HYBRID_ALPHA if index_is_flat(engine) else LONGDOC_HYBRID_ALPHA


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


def unique_document_membership(
    engine: Any,
    node_ids: Sequence[str],
    max_docs: int = LONGDOC_UNION_BM25_DOCS,
) -> List[str]:
    """First node of each of the first ``max_docs`` unique documents.

    Long-doc RRF membership may include these BM25@10 papers. Flat indexes
    must not call this into the allowlist.
    """
    if max_docs <= 0:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for nid in node_ids:
        doc = node_document_id(engine, nid)
        if not doc or doc in seen:
            continue
        seen.add(doc)
        out.append(str(nid))
        if len(out) >= max_docs:
            break
    return out


def node_document_id(engine: Any, nid: str) -> str:
    graph = getattr(engine, "graph", None)
    if graph is None or nid not in graph:
        return str(nid)
    data = graph.nodes[nid]
    meta = data.get("metadata") or {}
    return str(meta.get("doc_id") or data.get("source") or nid)


def diversify_by_document(
    engine: Any,
    order: Sequence[str],
    scores: Dict[str, float],
    top_k: int,
) -> List[str]:
    """Rank documents by max chunk score, then round-robin chunks.

    Long-doc first-stage: a paper with many mid chunks must not occupy the
    entire top-k before a second paper's best chunk is seen. Flat one-node
    corpora are a no-op (each id is already its document).
    """
    if top_k <= 0 or not order:
        return []
    by_doc: Dict[str, List[str]] = defaultdict(list)
    for nid in order:
        by_doc[node_document_id(engine, nid)].append(nid)
    doc_score = {
        doc: max(scores.get(nid, 0.0) for nid in nids)
        for doc, nids in by_doc.items()
    }
    docs_ranked = sorted(doc_score, key=lambda doc: doc_score[doc], reverse=True)
    queues = {doc: deque(by_doc[doc]) for doc in docs_ranked}
    out: List[str] = []
    while len(out) < top_k:
        progressed = False
        for doc in docs_ranked:
            if queues[doc]:
                out.append(queues[doc].popleft())
                progressed = True
                if len(out) >= top_k:
                    break
        if not progressed:
            break
    return out


def rescue_lexical_documents(
    engine: Any,
    order: Sequence[str],
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    top_k: int,
    m: int = LONGDOC_RESCUE_M,
    head: int = LONGDOC_RESCUE_FROM,
    slot: int = LONGDOC_RESCUE_SLOT,
) -> List[str]:
    """Splice up to ``m`` BM25-only documents into the ranked list.

    Allowlist still blocks lexical-only ids from RRF. On long-doc corpora the
    stronger BM25 arm would otherwise never reach nDCG@10. Flat indexes must
    not call this. Extras replace the tail so ``top_k`` is unchanged.
    """
    if top_k <= 0 or m <= 0:
        return list(order[:top_k])
    dense = set(dense_ids)
    seen_docs = {node_document_id(engine, nid) for nid in order}
    extra: List[str] = []
    for nid in bm25_ids[:head]:
        if nid in dense:
            continue
        doc = node_document_id(engine, nid)
        if not doc or doc in seen_docs:
            continue
        extra.append(nid)
        seen_docs.add(doc)
        if len(extra) >= m:
            break
    if not extra:
        return list(order[:top_k])
    extra_set = set(extra)
    kept = [nid for nid in order if nid not in extra_set]
    insert_at = min(max(slot, 0), max(0, top_k - len(extra)))
    return (kept[:insert_at] + extra + kept[insert_at:])[:top_k]
