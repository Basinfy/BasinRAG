import hashlib
import numpy as np
from typing import Any, Dict, List, Optional
from ..core.topology import BasinTopologyEngine


class TopologicalGlobalSearch:
    """Map-reduce over basins ranked by L3 (fallback: attractor embedding)."""

    def __init__(self, engine: BasinTopologyEngine, encoder: Optional[Any] = None, *, encoder_revision: str = "unresolved", query_prompt: str = ""):
        self.engine = engine
        self.encoder = encoder
        self.encoder_revision = encoder_revision
        self.query_prompt = query_prompt
        self._l3_cache: Dict[tuple[str, str, str], np.ndarray] = {}

    def _l3_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        if not texts or self.encoder is None:
            return {}
        build_id = getattr(self.engine, "build_id", "")
        text_hashes = {
            text: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in set(texts) if text
        }
        missing = [
            text for text, digest in text_hashes.items()
            if (build_id, self.encoder_revision, digest) not in self._l3_cache
        ]
        if missing:
            vectors = np.asarray(
                self.encoder.encode(missing, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for text, vector in zip(missing, vectors):
                vector = vector / (np.linalg.norm(vector) + 1e-10)
                digest = text_hashes[text]
                self._l3_cache[(build_id, self.encoder_revision, digest)] = vector
        return {
            text: self._l3_cache[(build_id, self.encoder_revision, text_hashes[text])]
            for text in text_hashes
        }

    def search_structured(
        self,
        query: str,
        top_k_basins: int = 5,
        max_nodes_per_basin: int = 8,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not self.engine.basins:
            return []

        query_emb = query_embedding
        if query_emb is None and self.encoder is not None:
            text = f"{self.query_prompt}{query}" if self.query_prompt else query
            query_emb = self.encoder.encode(text, normalize_embeddings=True)
        if query_emb is not None:
            query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        l3_texts = []
        if query_emb is not None:
            for basin_id, basin in self.engine.basins.items():
                if basin.rho_tree.has_node(basin_id):
                    data = basin.rho_tree.nodes[basin_id]
                    if data.get("l3_source") == "llm" and data.get("l3_summary"):
                        l3_texts.append(data["l3_summary"])
        l3_vectors = self._l3_embeddings(l3_texts)

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
                    l3_emb = l3_vectors.get(l3) if l3 else None
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

        basin_scores.sort(key=lambda x: (-x[1], str(x[0])))
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
            ranked_nodes.sort(key=lambda x: (-x[0], x[1], str(x[2])))
            hubs: List[str] = []
            neighbors: List[str] = []
            ids: List[str] = []
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
