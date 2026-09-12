from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from ..core.topology import BasinTopologyEngine
from ..indexer.bm25 import BM25Index
from .briefing import MIN_CONFIDENCE
from .fusion import DEFAULT_HOP_MISSING, apply_hop_prior, ranked_ids, weighted_rrf
from .local_search import TopologicalLocalSearch


class HybridSearch:
    """BM25 + topological FAISS fused with weighted RRF on node ids."""

    def __init__(self, engine: BasinTopologyEngine, local_search: TopologicalLocalSearch):
        self.engine = engine
        self.local_search = local_search
        self._bm25 = engine.bm25
        if self._bm25 is None:
            self._bm25 = BM25Index()
            self._rebuild_bm25()
            engine.bm25 = self._bm25

    def _rebuild_bm25(self):
        nodes = list(self.engine.graph.nodes(data=True))
        if not nodes:
            return
        ids = [n[0] for n in nodes]
        texts = [n[1].get("text", "") for n in nodes]
        self._bm25.build(ids, texts)

    @staticmethod
    def resolve_candidate_k(top_k: int, candidate_k: Optional[int] = None) -> int:
        """Align production pool with the gate harness: max(50, top_k*5)."""
        if candidate_k is not None:
            return max(1, int(candidate_k))
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

    def search_nodes(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        use_hop_prior: bool = True,
        use_confidence_gate: bool = True,
        candidate_k: Optional[int] = None,
        hop_missing: str = DEFAULT_HOP_MISSING,
        expand_graph: bool = True,
        expand_max_extra: int = 100,
    ) -> List[Dict[str, Any]]:
        if self._bm25 is None or self._bm25.n == 0:
            hits = self.local_search.dense_hits(query_embedding, top_k=top_k)
            if use_confidence_gate and (not hits or hits[0]["score"] < MIN_CONFIDENCE):
                return []
            return hits

        ck = self.resolve_candidate_k(top_k, candidate_k)
        bm25_hits = self._bm25.score(query, top_k=ck)
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

        all_candidate_ids = set(bm25_ids) | set(semantic_ids) | set(expanded_ids)
        # Expanded-only nodes get weak semantic ranks after the dense list for RRF.
        fused_semantic_ids = list(semantic_ids)
        for nid in expanded_ids:
            if nid not in semantic_ids and nid not in bm25_ids:
                fused_semantic_ids.append(nid)

        seeds = (set(semantic_ids[:3]) | set(bm25_ids[:3])) & all_candidate_ids
        if not seeds:
            seeds = set(list(all_candidate_ids)[:3])

        hops: Dict[str, int] = {s: 0 for s in seeds}
        q_bfs = deque((s, 0) for s in seeds)
        while q_bfs:
            curr, d = q_bfs.popleft()
            if curr in self.engine.graph:
                for nbr in self.engine.graph.neighbors(curr):
                    if nbr in all_candidate_ids and nbr not in hops:
                        hops[nbr] = d + 1
                        q_bfs.append((nbr, d + 1))

        scores = weighted_rrf(bm25_ids, fused_semantic_ids)
        scores = apply_hop_prior(
            scores, hops, enabled=use_hop_prior, missing=hop_missing
        )
        order = ranked_ids(scores, top_k)

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
            calibrated_score = float(np.clip(scores.get(nid, 0.0) * 60.0, 0.0, 1.0))
            results.append({
                "id": nid,
                "text": data.get("text", ""),
                "hops": hops.get(nid, 0),
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "score": calibrated_score,
                "dense_score": d_score,
                "bm25_score": b_score,
                "metadata": data.get("metadata") or {},
            })
        return results
