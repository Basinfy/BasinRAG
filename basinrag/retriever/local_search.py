from typing import List, Dict, Any
import numpy as np
from collections import deque
from ..core.topology import BasinTopologyEngine
from ..core.vector_index import build_ip_index
from .fusion import HOP_LAMBDA
from .briefing import MIN_CONFIDENCE
import math


class TopologicalLocalSearch:
    """FAISS entry points, then hydrate the seed basin plus a one-hop fringe."""

    def __init__(self, engine: BasinTopologyEngine):
        self.engine = engine
        self._index = None
        self._node_ids = None
        self._build_index()

    def _build_index(self):
        nodes = list(self.engine.graph.nodes(data=True))
        if not nodes:
            return
        ids = []
        vectors = []
        for nid, data in nodes:
            emb = data.get("embedding")
            if emb is None:
                continue
            ids.append(nid)
            vectors.append(emb)
        if not ids:
            return
        self._node_ids = ids
        embeddings = np.array(vectors, dtype=np.float32)
        self._index = build_ip_index(embeddings)

    def _faiss_hits(self, query_embedding: np.ndarray, top_k: int):
        import faiss
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        k = min(max(1, top_k), self._index.ntotal)
        D, I = self._index.search(query, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            sim = float(score)
            if sim < 0.05 and hits:
                continue
            hits.append((self._node_ids[int(idx)], sim))
        return hits

    def dense_hits(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Corpus-wide FAISS, no basin filter — the hybrid semantic arm."""
        if self._index is None:
            return []
        results = []
        for nid, sim in self._faiss_hits(query_embedding, top_k):
            data = self.engine.graph.nodes[nid]
            results.append({
                "id": nid,
                "text": data.get("text", ""),
                "hops": int(data.get("hops", 0)),
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "score": sim,
            })
        return results

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        max_tokens: int = 2000,
        return_ids: bool = False,
    ) -> List[Any]:
        ranked = self.search_nodes(query_embedding, top_k=top_k, max_tokens=max_tokens)
        if return_ids:
            return ranked
        return [r["text"] for r in ranked]

    def _neighbors_of_type(self, nid: str, edge_type: str) -> List[str]:
        if nid not in self.engine.graph:
            return []
        out = []
        for nbr in self.engine.graph.neighbors(nid):
            edge = self.engine.graph.edges[nid, nbr]
            if edge.get("type") == edge_type:
                out.append(nbr)
        return out

    def search_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        max_tokens: int = 2000,
    ) -> List[Dict[str, Any]]:
        if self._index is None:
            return []

        raw_hits = self._faiss_hits(query_embedding, min(10, self._index.ntotal))
        if not raw_hits or raw_hits[0][1] < MIN_CONFIDENCE:
            return []
        sim_map = dict(raw_hits)
        entrypoints = [
            nid for nid, sim in raw_hits[:3]
            if sim >= MIN_CONFIDENCE
        ]
        if not entrypoints:
            return []

        allowed = set()
        for ep in entrypoints:
            basin_id = self.engine.basin_id_of(ep)
            if basin_id and basin_id in self.engine.basins:
                allowed.update(self.engine.basins[basin_id].rho_tree.nodes)
            else:
                allowed.add(ep)

        for nid, _sim in raw_hits:
            allowed.add(nid)
            allowed.update(self._neighbors_of_type(nid, "virtual-edge"))

        seq_extra = set()
        for nid in allowed:
            seq_extra.update(self._neighbors_of_type(nid, "sequential"))
        allowed.update(seq_extra)

        scored = []
        visited = set()
        queue = deque(entrypoints)
        while queue:
            curr = queue.popleft()
            if curr in visited or curr not in allowed:
                continue
            visited.add(curr)
            data = self.engine.graph.nodes[curr]
            hops = int(data.get("hops", 0))
            hop_w = 0.7 + 0.3 * math.exp(-hops * HOP_LAMBDA)
            if curr in sim_map:
                score = sim_map[curr] * hop_w
            else:
                score = 0.05 * hop_w
            scored.append({
                "id": curr,
                "text": data.get("text", ""),
                "hops": hops,
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "score": score,
            })
            for neighbor in self.engine.graph.neighbors(curr):
                if neighbor not in visited and neighbor in allowed:
                    queue.append(neighbor)

        for nid in allowed - visited:
            if nid not in self.engine.graph:
                continue
            data = self.engine.graph.nodes[nid]
            hops = int(data.get("hops", 0))
            hop_w = 0.7 + 0.3 * math.exp(-hops * HOP_LAMBDA)
            score = sim_map[nid] * hop_w if nid in sim_map else 0.05 * hop_w
            scored.append({
                "id": nid,
                "text": data.get("text", ""),
                "hops": hops,
                "basin_id": data.get("basin_id", ""),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "score": score,
            })

        scored.sort(key=lambda x: (-x["score"], x["hops"]))
        results = []
        tokens = 0
        for item in scored:
            if len(results) >= top_k:
                break
            cost = max(1, len(item["text"]) // 4)
            if tokens + cost > max_tokens and results:
                break
            results.append(item)
            tokens += cost
        return results
