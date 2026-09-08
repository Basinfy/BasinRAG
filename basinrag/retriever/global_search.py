import hashlib
import numpy as np
from typing import Any, Dict, List, Optional
from ..core.topology import BasinTopologyEngine


class TopologicalGlobalSearch:
    """Map-reduce over basins ranked by L3 (fallback: attractor embedding)."""

    def __init__(self, engine: BasinTopologyEngine, encoder: Optional[Any] = None):
        self.engine = engine
        self.encoder = encoder
        self._l3_cache: Dict[str, np.ndarray] = {}

    def _l3_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text or self.encoder is None:
            return None
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._l3_cache:
            vec = self.encoder.encode(text)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            self._l3_cache[key] = vec
        return self._l3_cache[key]

    def search(
        self,
        query: str,
        top_k_basins: int = 5,
        max_nodes_per_basin: int = 3,
        return_structured: bool = False,
    ) -> List[Any]:
        packet_parts = self.search_structured(query, top_k_basins, max_nodes_per_basin)
        if return_structured:
            return packet_parts
        texts = []
        for part in packet_parts:
            texts.extend(part.get("hubs", []))
            texts.extend(part.get("neighbors", []))
        return texts

    def search_structured(
        self,
        query: str,
        top_k_basins: int = 5,
        max_nodes_per_basin: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self.engine.basins:
            return []

        query_emb = None
        if self.encoder is not None:
            query_emb = self.encoder.encode(query)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        basin_scores = []
        for basin_id, basin in self.engine.basins.items():
            if not basin.rho_tree.has_node(basin_id):
                continue
            data = basin.rho_tree.nodes[basin_id]
            l3 = ""
            score = 0.0
            if query_emb is not None:
                if data.get("l3_source") == "llm":
                    l3 = data.get("l3_summary", "") or ""
                    l3_emb = self._l3_embedding(l3) if l3 else None
                    if l3_emb is not None:
                        score = float(np.dot(query_emb, l3_emb))
                if score == 0.0:
                    attractor_emb = data.get("embedding")
                    if attractor_emb is not None:
                        attractor_emb = np.array(attractor_emb, dtype=np.float32)
                        norm = np.linalg.norm(attractor_emb)
                        if norm > 0:
                            attractor_emb = attractor_emb / norm
                        score = float(np.dot(query_emb, attractor_emb))
                l3 = l3 or data.get("l3_label", "") or data.get("l3_summary", "")
            basin_scores.append((basin_id, score, l3))

        basin_scores.sort(key=lambda x: x[1], reverse=True)
        if not basin_scores or basin_scores[0][1] <= 1e-9:
            return []
        top_basins = basin_scores[:top_k_basins]

        results = []
        for basin_id, score, l3_summary in top_basins:
            basin = self.engine.basins[basin_id]
            ranked_nodes = []
            for node_id, node_data in basin.rho_tree.nodes(data=True):
                text = node_data.get("text", "")
                node_score = 0.0
                emb = node_data.get("embedding")
                if query_emb is not None and emb is not None:
                    emb = np.array(emb, dtype=np.float32)
                    nrm = np.linalg.norm(emb)
                    if nrm > 0:
                        node_score = float(np.dot(query_emb, emb / nrm))
                ranked_nodes.append((node_score, int(node_data.get("hops", 999)), node_id, text))
            ranked_nodes.sort(key=lambda x: (-x[0], x[1]))
            hubs = []
            neighbors = []
            ids = []
            for node_score, hops, node_id, text in ranked_nodes[:max_nodes_per_basin]:
                ids.append(node_id)
                hubs.append(text)
            results.append({
                "basin_id": basin_id,
                "score": score,
                "l3": l3_summary,
                "hubs": hubs,
                "neighbors": neighbors,
                "node_ids": ids,
            })
        return results
