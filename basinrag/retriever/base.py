from typing import List, Any, Optional
import numpy as np
from pydantic import ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from .local_search import TopologicalLocalSearch
from .global_search import TopologicalGlobalSearch
from .hybrid_search import HybridSearch
from .reranker import CrossEncoderReranker
from .router import IntelligentQueryRouter
from .briefing import BriefingPacket, cap_satellite
from .prompts import QUERY_PROMPT


class BasinRAGRetriever(BaseRetriever):
    engine: Any
    encoder: Any
    search_type: str = "auto"
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    query_prompt: str = QUERY_PROMPT
    use_rerank: bool = True

    _local: TopologicalLocalSearch = None
    _global: TopologicalGlobalSearch = None
    _hybrid: HybridSearch = None
    _reranker: CrossEncoderReranker = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_cache = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self):
        self._local = TopologicalLocalSearch(self.engine)
        self._global = TopologicalGlobalSearch(self.engine, self.encoder)
        self._hybrid = HybridSearch(self.engine, self._local)
        self._reranker = CrossEncoderReranker(
            model_name=self.reranker_model,
            max_length=512,
        )
        if self.encoder is not None and getattr(IntelligentQueryRouter, "_global_centroid", None) is None:
            IntelligentQueryRouter.train_centroids(self.encoder)

    def configure(self, search_type: Optional[str] = None, top_k: Optional[int] = None):
        if search_type is not None:
            self.search_type = search_type
        if top_k is not None:
            self.top_k = top_k

    def _encode_query(self, query: str, apply_prompt: bool = True) -> np.ndarray:
        """Encode query; by default prefix the BGE instruction prompt."""
        text = query
        if apply_prompt and self.query_prompt and not query.startswith(self.query_prompt):
            text = f"{self.query_prompt}{query}"
        if text not in self._query_cache:
            if len(self._query_cache) >= 512:
                self._query_cache.pop(next(iter(self._query_cache)))
            emb = self.encoder.encode(text)
            emb = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            self._query_cache[text] = emb
        return self._query_cache[text]

    def brief(self, query: str, search_type: Optional[str] = None, top_k: Optional[int] = None) -> BriefingPacket:
        strategy = search_type if search_type is not None else self.search_type
        k = top_k if top_k is not None else self.top_k
        if strategy == "auto":
            strategy = IntelligentQueryRouter.route_with_embeddings(query)

        query_emb = self._encode_query(query)

        packet = BriefingPacket()
        if strategy == "global":
            parts = self._global.search_structured(query, top_k_basins=k)
            if not parts:
                return packet
            for part in parts:
                l3 = cap_satellite(part.get("l3") or "")
                if l3:
                    packet.satellites.append(l3)
                packet.hubs.extend(part.get("hubs") or [])
                packet.neighbors.extend(part.get("neighbors") or [])
                packet.node_ids.extend(part.get("node_ids") or [])
            packet.confidence = max(float(p.get("score") or 0.0) for p in parts)
        elif strategy == "local":
            nodes = self._local.search_nodes(query_emb, top_k=max(k * 2, 20))
            if not nodes:
                # Fallback: hybrid seeds then local expansion via hybrid graph expand
                nodes = self._hybrid.search_nodes(
                    query, query_emb, top_k=max(k * 2, 20), use_confidence_gate=False
                )
            if not nodes:
                return packet
            self._fill_from_nodes(packet, nodes)
        else:
            nodes = self._hybrid.search_nodes(
                query, query_emb, top_k=k * 2, use_confidence_gate=False
            )
            if not nodes:
                return packet
            self._fill_from_nodes(packet, nodes)

        items = list(zip(packet.node_ids, packet.hubs + packet.neighbors))
        if items and self.use_rerank and self._reranker:
            ordered = self._reranker.rerank_items(query, items, top_k=k)
            packet.node_ids = [nid for nid, _ in ordered]
            packet.hubs = [text for _, text in ordered]
            packet.neighbors = []
        elif items:
            packet.node_ids = packet.node_ids[:k]
            packet.hubs = (packet.hubs + packet.neighbors)[:k]
            packet.neighbors = []
        return packet

    def _fill_from_nodes(self, packet: BriefingPacket, nodes, seed_ids=None):
        seen = set()
        best = 0.0
        for nid in seed_ids or []:
            if nid in seen or nid not in self.engine.graph:
                continue
            seen.add(nid)
            data = self.engine.graph.nodes[nid]
            packet.hubs.append(data.get("text", ""))
            packet.node_ids.append(nid)
        for item in nodes:
            nid = item["id"]
            best = max(best, float(item.get("score") or 0.0))
            if nid in seen:
                continue
            seen.add(nid)
            packet.node_ids.append(nid)
            packet.hubs.append(self._parent_context_text(nid, item.get("text", "")))
            l3 = ""
            basin_id = item.get("basin_id")
            if basin_id and basin_id in self.engine.basins:
                tree = self.engine.basins[basin_id].rho_tree
                if tree.has_node(basin_id):
                    data = tree.nodes[basin_id]
                    l3 = data.get("l3_summary") or data.get("l3_label") or ""
            l3 = cap_satellite(l3)
            if l3 and l3 not in packet.satellites:
                packet.satellites.append(l3)
        packet.confidence = best

    def _parent_context_text(self, nid: str, child_text: str) -> str:
        """Retrieve child, surface parent window (± sequential neighbors)."""
        if nid not in self.engine.graph:
            return child_text
        parts = [child_text]
        seen = {child_text}
        for nbr in self.engine.graph.neighbors(nid):
            edge = self.engine.graph.edges[nid, nbr]
            if edge.get("type") != "sequential":
                continue
            txt = self.engine.graph.nodes[nbr].get("text") or ""
            if txt and txt not in seen:
                seen.add(txt)
                parts.append(txt)
        if len(parts) == 1:
            return child_text
        # Keep child first for ranking fidelity; append neighbors as parent context.
        return "\n\n".join(parts)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        packet = self.brief(query)
        docs = []
        texts = packet.texts_for_rerank()[: self.top_k]
        for i, text in enumerate(texts):
            meta = {}
            if i < len(packet.node_ids):
                nid = packet.node_ids[i]
                if hasattr(self.engine, "graph") and nid in self.engine.graph:
                    node_data = self.engine.graph.nodes[nid]
                    meta = dict(node_data.get("metadata") or {})
                    meta["node_id"] = nid
                    meta["source"] = node_data.get("source", "")
                    meta["doc_id"] = meta.get("doc_id") or node_data.get("source", "")
            docs.append(Document(page_content=text, metadata=meta))
        return docs


# Dual-Layer Adapter alias
TopologicalRetriever = BasinRAGRetriever
