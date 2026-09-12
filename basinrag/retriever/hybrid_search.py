from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
from ..core.topology import BasinTopologyEngine
from ..indexer.bm25 import BM25Index
from .local_search import TopologicalLocalSearch
from .fusion import weighted_rrf, apply_hop_prior, ranked_ids
from .briefing import MIN_CONFIDENCE


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

    def search_nodes(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        use_hop_prior: bool = True,
        use_confidence_gate: bool = True,
    ) -> List[Dict[str, Any]]:
        if self._bm25 is None or self._bm25.n == 0:
            hits = self.local_search.dense_hits(query_embedding, top_k=top_k)
            if use_confidence_gate and (not hits or hits[0]["score"] < MIN_CONFIDENCE):
                return []
            return hits

        candidate_k = max(20, top_k * 3)
        bm25_hits = self._bm25.score(query, top_k=candidate_k)
        bm25_ids = [nid for nid, _ in bm25_hits]
        semantic = self.local_search.dense_hits(query_embedding, top_k=candidate_k)
        semantic_ids = [item["id"] for item in semantic]

        best_dense = semantic[0]["score"] if semantic else 0.0
        best_bm25 = bm25_hits[0][1] if bm25_hits else 0.0
        if use_confidence_gate and best_dense < MIN_CONFIDENCE and best_bm25 < 0.5:
            return []

        # Dynamic geodesic hops from top query seeds
        all_candidate_ids = set(bm25_ids) | set(semantic_ids)
        seeds = (set(semantic_ids[:3]) | set(bm25_ids[:3])) & all_candidate_ids
        if not seeds:
            seeds = set(list(all_candidate_ids)[:3])

        from collections import deque
        hops: Dict[str, int] = {s: 0 for s in seeds}
        q_bfs = deque((s, 0) for s in seeds)
        while q_bfs:
            curr, d = q_bfs.popleft()
            if curr in self.engine.graph:
                for nbr in self.engine.graph.neighbors(curr):
                    if nbr in all_candidate_ids and nbr not in hops:
                        hops[nbr] = d + 1
                        q_bfs.append((nbr, d + 1))

        scores = weighted_rrf(bm25_ids, semantic_ids)
        scores = apply_hop_prior(scores, hops, enabled=use_hop_prior)
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
            })
        return results

