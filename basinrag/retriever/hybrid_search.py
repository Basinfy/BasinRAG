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

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        return_ids: bool = False,
    ) -> List[Any]:
        ranked = self.search_nodes(query, query_embedding, top_k=top_k)
        if return_ids:
            return ranked
        return [r["text"] for r in ranked]

    def search_nodes(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self._bm25 is None or self._bm25.n == 0:
            hits = self.local_search.dense_hits(query_embedding, top_k=top_k)
            if not hits or hits[0]["score"] < MIN_CONFIDENCE:
                return []
            return hits

        bm25_hits = self._bm25.score(query, top_k=10)
        bm25_ids = [nid for nid, _ in bm25_hits]
        semantic = self.local_search.dense_hits(query_embedding, top_k=10)
        semantic_ids = [item["id"] for item in semantic]
        best_dense = semantic[0]["score"] if semantic else 0.0
        best_bm25 = bm25_hits[0][1] if bm25_hits else 0.0
        if max(best_dense, best_bm25) < MIN_CONFIDENCE:
            return []

        hops = {
            nid: int(self.engine.graph.nodes[nid].get("hops", 0))
            for nid in set(bm25_ids) | set(semantic_ids)
            if nid in self.engine.graph
        }
        scores = weighted_rrf(bm25_ids, semantic_ids)
        scores = apply_hop_prior(scores, hops)
        order = ranked_ids(scores, top_k)

        by_id = {item["id"]: item for item in semantic}
        results = []
        for nid in order:
            if nid in by_id:
                results.append(by_id[nid])
                continue
            data = self.engine.graph.nodes[nid]
            results.append({
                "id": nid,
                "text": data.get("text", ""),
                "hops": int(data.get("hops", 0)),
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "score": scores.get(nid, 0.0),
            })
        return results
