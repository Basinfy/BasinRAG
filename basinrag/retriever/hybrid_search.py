from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.topology import BasinTopologyEngine
from ..indexer.bm25 import BM25Index
from .briefing import MIN_CONFIDENCE
from .fusion import (
    DEFAULT_HOP_MISSING,
    LONGDOC_CANDIDATE_K_MIN,
    LONGDOC_CANDIDATE_K_MULT,
    LONGDOC_UNION_BM25_DOCS,
    apply_hop_prior,
    diversify_by_document,
    index_is_flat,
    multi_signal_drf,
    ranked_ids,
    rescue_lexical_documents,
    resolve_hybrid_alpha,
    unique_document_membership,
    weighted_rrf,
)
from .local_search import TopologicalLocalSearch


def union_bm25_hits(
    primary: Sequence[Tuple[str, float]],
    secondary: Sequence[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse two BM25 passes by RRF; keep the first pass score when present."""
    scores: Dict[str, float] = {}
    raw: Dict[str, float] = {}
    for rank, (nid, score) in enumerate(primary):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
        raw[nid] = float(score)
    for rank, (nid, score) in enumerate(secondary):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
        raw.setdefault(nid, float(score))
    return [(nid, raw[nid]) for nid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


class HybridSearch:
    """BM25 + topological FAISS fused with weighted RRF on node ids."""

    def __init__(self, engine: BasinTopologyEngine, local_search: TopologicalLocalSearch):
        self.engine = engine
        self.local_search = local_search
        self._bm25: Optional[BM25Index] = getattr(engine, "bm25", None)

    def _rebuild_bm25(self):
        if self._bm25 is None:
            return
        nodes = list(self.engine.graph.nodes(data=True))
        if not nodes:
            return
        ids = [n[0] for n in nodes]
        texts = [n[1].get("text", "") for n in nodes]
        self._bm25.build(ids, texts)

    @staticmethod
    def resolve_candidate_k(
        top_k: int,
        candidate_k: Optional[int] = None,
        *,
        long_doc: bool = False,
    ) -> int:
        """Flat: max(50, top_k*5). Long-doc: max(200, top_k*10)."""
        if candidate_k is not None:
            return max(1, int(candidate_k))
        if long_doc:
            return max(LONGDOC_CANDIDATE_K_MIN, top_k * LONGDOC_CANDIDATE_K_MULT)
        return max(50, top_k * 5)

    def _expand_graph_candidates(
        self,
        seed_ids: Sequence[str],
        max_extra: int = 100,
        sequential_radius: int = 2,
    ) -> List[str]:
        """Pull new nodes (sequential / basin siblings / virtual-edge) before RRF."""
        if not seed_ids or max_extra <= 0:
            return []
        seeds = [nid for nid in seed_ids if nid in self.engine.graph]
        if not seeds:
            return []

        expanded: List[str] = []
        seen: Set[str] = set(seeds)

        def _add(nid: str) -> bool:
            if nid in seen or nid not in self.engine.graph:
                return False
            seen.add(nid)
            expanded.append(nid)
            return len(expanded) >= max_extra

        for seed in seeds:
            frontier = {seed}
            for _ in range(max(0, sequential_radius)):
                nxt: Set[str] = set()
                for nid in frontier:
                    for nbr in self.local_search._neighbors_of_type(nid, "sequential"):
                        if _add(nbr):
                            return expanded
                        nxt.add(nbr)
                frontier = nxt

        for seed in seeds:
            basin_id = self.engine.basin_id_of(seed)
            if not basin_id or basin_id not in self.engine.basins:
                continue
            for nid in self.engine.basins[basin_id].rho_tree.nodes:
                if _add(nid):
                    return expanded

        fringe_src = list(seeds) + list(expanded)
        for nid in fringe_src:
            for nbr in self.local_search._neighbors_of_type(nid, "virtual-edge"):
                if _add(nbr):
                    return expanded

        return expanded

    def _lexical_expand_query(
        self,
        query: str,
        bm25_hits: Sequence[Tuple[str, float]],
        n_feedback: int = 3,
    ) -> str:
        expand = getattr(self._bm25, "expansion_terms", None)
        if not callable(expand) or not query or not bm25_hits:
            return query
        texts = []
        for nid, _ in bm25_hits[:n_feedback]:
            if nid not in self.engine.graph:
                continue
            texts.append(str(self.engine.graph.nodes[nid].get("text") or ""))
        terms = expand(query, texts)
        if not terms:
            return query
        return query + " " + " ".join(terms)

    def search_nodes(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        use_hop_prior: bool = False,
        use_multi_signal_drf: bool = False,
        use_confidence_gate: bool = False,
        candidate_k: Optional[int] = None,
        hop_missing: str = DEFAULT_HOP_MISSING,
        expand_graph: bool = False,
        expand_max_extra: int = 100,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if self._bm25 is None or self._bm25.n == 0:
            hits = self.local_search.dense_hits(query_embedding, top_k=top_k)
            if use_confidence_gate and (not hits or hits[0]["score"] < MIN_CONFIDENCE):
                return []
            for item in hits:
                dense_score = max(0.0, float(item.get("score", 0.0)))
                item["rank_score"] = float(item.get("score", 0.0))
                item["dense_score"] = dense_score
                item["confidence"] = dense_score
            return hits

        long_doc = not index_is_flat(self.engine)
        ck = self.resolve_candidate_k(top_k, candidate_k, long_doc=long_doc)
        bm25_hits = self._bm25.score(query, top_k=ck)
        if long_doc:
            expanded_query = self._lexical_expand_query(query, bm25_hits)
            if expanded_query != query:
                bm25_hits = union_bm25_hits(
                    bm25_hits, self._bm25.score(expanded_query, top_k=ck)
                )
        bm25_ids = [nid for nid, _ in bm25_hits]
        semantic = self.local_search.dense_hits(query_embedding, top_k=ck)
        semantic_ids = [item["id"] for item in semantic]

        best_dense = semantic[0]["score"] if semantic else 0.0
        best_bm25 = bm25_hits[0][1] if bm25_hits else 0.0
        if use_confidence_gate and best_dense < MIN_CONFIDENCE and best_bm25 < 0.5:
            return []

        # Graph expansion adds nodes that BM25/dense missed (multi-evidence recall).
        seed_pool = list(dict.fromkeys(semantic_ids[:5] + bm25_ids[:5]))
        expanded_ids: List[str] = []
        if expand_graph:
            expanded_ids = self._expand_graph_candidates(
                seed_pool, max_extra=expand_max_extra, sequential_radius=2
            )

        all_candidate_ids = list(dict.fromkeys(bm25_ids + semantic_ids + expanded_ids))
        candidate_set = set(all_candidate_ids)
        # Expanded-only nodes get weak semantic ranks after the dense list for RRF.
        fused_semantic_ids = list(semantic_ids)
        for nid in expanded_ids:
            if nid not in semantic_ids and nid not in bm25_ids:
                fused_semantic_ids.append(nid)

        seeds = list(dict.fromkeys(semantic_ids[:3] + bm25_ids[:3]))
        seeds = [nid for nid in seeds if nid in candidate_set]
        if not seeds:
            seeds = all_candidate_ids[:3]

        hops: Dict[str, int] = {s: 0 for s in seeds}
        if use_hop_prior or use_multi_signal_drf:
            q_bfs = deque((s, 0) for s in seeds)
            while q_bfs:
                curr, d = q_bfs.popleft()
                if curr in self.engine.graph:
                    for nbr in self.engine.graph.neighbors(curr):
                        if nbr in candidate_set and nbr not in hops:
                            hops[nbr] = d + 1
                            q_bfs.append((nbr, d + 1))

        # Flat: BM25 votes only inside the dense/expanded pool (SciFact).
        # Long-doc: union BM25@10 unique papers into membership as well.
        bm25_allowlist = list(dict.fromkeys(list(semantic_ids) + list(expanded_ids)))
        if long_doc:
            bm25_allowlist = list(dict.fromkeys(
                bm25_allowlist
                + unique_document_membership(
                    self.engine, bm25_ids, LONGDOC_UNION_BM25_DOCS
                )
            ))
        scores = weighted_rrf(
            bm25_ids,
            fused_semantic_ids,
            alpha=resolve_hybrid_alpha(self.engine, alpha),
            bm25_allowlist=bm25_allowlist,
        )
        if use_hop_prior:
            scores = apply_hop_prior(scores, hops, enabled=True, missing=hop_missing)

        if use_multi_signal_drf:
            cohesion_map = {}
            for nid in all_candidate_ids:
                bid = self.engine.basin_id_of(nid)
                if bid and bid in self.engine.basins:
                    cohesion_map[nid] = self.engine.basins[bid].cohesion
            scores = multi_signal_drf(
                scores, hops, cohesion=cohesion_map, enabled=True
            )

        order = ranked_ids(scores, len(scores))
        if long_doc:
            order = diversify_by_document(self.engine, order, scores, top_k)
            order = rescue_lexical_documents(
                self.engine, order, bm25_ids, semantic_ids, top_k
            )
        else:
            order = order[:top_k]

        dense_map = {item["id"]: max(0.0, float(item["score"])) for item in semantic}
        bm25_raw_map = {nid: max(0.0, float(score)) for nid, score in bm25_hits}
        max_bm25 = max(bm25_raw_map.values()) if bm25_raw_map else 1.0
        bm25_norm_map = {nid: score / (max_bm25 + 1e-6) for nid, score in bm25_raw_map.items()}

        results = []
        for nid in order:
            if nid not in self.engine.graph:
                continue
            data = self.engine.graph.nodes[nid]
            d_score = dense_map.get(nid, 0.0)
            b_score = bm25_norm_map.get(nid, 0.0)
            rank_score = float(scores.get(nid, 0.0))
            results.append({
                "id": nid,
                "text": data.get("text", ""),
                "hops": hops.get(nid, 0),
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                # Keep the rank score separate from the raw dense-similarity
                # support signal used by the chat abstention heuristic.
                "score": rank_score,
                "rank_score": rank_score,
                "confidence": d_score,
                "dense_score": d_score,
                "bm25_score": b_score,
                "metadata": data.get("metadata") or {},
            })
        return results
