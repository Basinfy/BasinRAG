from typing import List, Any, Optional
import numpy as np
from pydantic import ConfigDict
from ..contracts import SearchType
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
    search_type: SearchType = "auto"
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    query_prompt: str = QUERY_PROMPT
    use_rerank: bool = True
    ranking_mode: str = "hybrid_rrf"

    _local: Optional[TopologicalLocalSearch] = None
    _global: Optional[TopologicalGlobalSearch] = None
    _hybrid: Optional[HybridSearch] = None
    _reranker: Optional[CrossEncoderReranker] = None
    _router: Optional[IntelligentQueryRouter] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_cache = {}
        self._rebuild_indexes()
        if self._router is None:
            self._router = IntelligentQueryRouter(self.encoder, query_prompt=self.query_prompt)

    def _rebuild_indexes(self):
        self._local = TopologicalLocalSearch(self.engine)
        metadata = getattr(self.engine, "index_metadata", {}) or {}
        reranker_revision = metadata.get("reranker_revision")
        if reranker_revision in {None, "", "unresolved", "unknown", "disabled"}:
            reranker_revision = None
        self._global = TopologicalGlobalSearch(
            self.engine,
            self.encoder,
            encoder_revision=str(metadata.get("encoder_revision", "unresolved")),
            query_prompt=self.query_prompt,
        )
        self._hybrid = HybridSearch(self.engine, self._local)
        self._reranker = CrossEncoderReranker(
            model_name=self.reranker_model,
            max_length=512,
            revision=reranker_revision,
        )
        self._router = IntelligentQueryRouter(self.encoder, query_prompt=self.query_prompt)

    def configure(self, search_type: Optional[SearchType] = None, top_k: Optional[int] = None):
        if search_type is not None:
            if search_type not in {"auto", "local", "global", "hybrid"}:
                raise ValueError("search_type must be one of: auto, local, global, hybrid")
            self.search_type = search_type
        if top_k is not None:
            if not 1 <= int(top_k) <= 50:
                raise ValueError("top_k must be between 1 and 50")
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

    def brief(
        self, query: str, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> BriefingPacket:
        strategy = search_type if search_type is not None else self.search_type
        k = top_k if top_k is not None else self.top_k
        if strategy not in {"auto", "local", "global", "hybrid"}:
            raise ValueError("search_type must be one of: auto, local, global, hybrid")
        if not 1 <= int(k) <= 50:
            raise ValueError("top_k must be between 1 and 50")
        query_emb = self._encode_query(query)
        assert self._local is not None
        assert self._global is not None
        assert self._hybrid is not None
        assert self._router is not None
        if strategy == "auto":
            try:
                strategy = self._router.route_with_embeddings(query, query_emb)
            except TypeError:
                # Compatibility for custom router adapters that accept only query text.
                strategy = self._router.route_with_embeddings(query)

        packet = BriefingPacket()
        context_nodes: List[dict[str, Any]] = []

        if self.ranking_mode != "experimental_topology":
            # Keep BM25 + FAISS RRF as the seed ranker for every route. The
            # local/global choice only controls briefing expansion below.
            nodes = self._hybrid.search_nodes(
                query,
                query_emb,
                top_k=max(k * 2, 20),
                use_confidence_gate=False,
                use_hop_prior=False,
                use_multi_signal_drf=False,
                expand_graph=False,
            )
            if not nodes:
                return packet
            self._fill_from_nodes(
                packet,
                nodes,
                confidence=self._dense_confidence(query_emb, [item["id"] for item in nodes]),
            )
            seed_count = len(packet.node_ids)

            if strategy == "local":
                # PPR/hops can contribute context, but never replace or
                # reorder the RRF-ranked seed passages.
                context_nodes = self._local.search_nodes(query_emb, top_k=max(k, 1))
            elif strategy == "global":
                parts = self._global_search(query, k, query_emb)
                for part in parts:
                    l3 = cap_satellite(part.get("l3") or "")
                    if l3 and l3 not in packet.satellites:
                        packet.satellites.append(l3)
                    for nid, text in zip(part.get("node_ids") or [], part.get("hubs") or []):
                        context_nodes.append({"id": nid, "text": text})

            seen = set(packet.node_ids)
            for item in context_nodes:
                nid = item.get("id")
                if not nid or nid in seen or nid not in self.engine.graph:
                    continue
                seen.add(nid)
                packet.node_ids.append(nid)
                packet.neighbors.append(
                    self._parent_context_text(nid, item.get("text") or self.engine.graph.nodes[nid].get("text", ""))
                )
        elif strategy == "global":
            parts = self._global_search(query, k, query_emb)
            if not parts:
                return packet
            global_node_ids: List[str] = []
            for part in parts:
                l3 = cap_satellite(part.get("l3") or "")
                if l3:
                    packet.satellites.append(l3)
                packet.hubs.extend(part.get("hubs") or [])
                packet.neighbors.extend(part.get("neighbors") or [])
                node_ids = part.get("node_ids") or []
                packet.node_ids.extend(node_ids)
                global_node_ids.extend(node_ids)
            packet.confidence = self._dense_confidence(query_emb, global_node_ids)
            seed_count = len(packet.node_ids)
        elif strategy == "local":
            nodes = self._local.search_nodes(query_emb, top_k=max(k * 2, 20))
            if not nodes:
                # Explicit experimental mode retains the existing hybrid
                # fallback, including its opt-in topological signals.
                nodes = self._hybrid.search_nodes(
                    query,
                    query_emb,
                    top_k=max(k * 2, 20),
                    use_confidence_gate=False,
                    use_hop_prior=True,
                    use_multi_signal_drf=True,
                    expand_graph=True,
                )
            if not nodes:
                return packet
            self._fill_from_nodes(
                packet,
                nodes,
                confidence=self._dense_confidence(query_emb, [item["id"] for item in nodes]),
            )
            seed_count = len(packet.node_ids)
        else:
            nodes = self._hybrid.search_nodes(
                query,
                query_emb,
                top_k=k * 2,
                use_confidence_gate=False,
                use_hop_prior=True,
                use_multi_signal_drf=True,
                expand_graph=True,
            )
            if not nodes:
                return packet
            self._fill_from_nodes(
                packet,
                nodes,
                confidence=self._dense_confidence(query_emb, [item["id"] for item in nodes]),
            )
            seed_count = len(packet.node_ids)

        # Only the seed list enters ranking/reranking. Route-specific topology
        # remains briefing context appended after those seed passages.
        seed_items = list(zip(packet.node_ids[:seed_count], packet.hubs[:seed_count]))
        context_ids = packet.node_ids[seed_count:]
        if seed_items and self.use_rerank and self._reranker:
            ordered = self._reranker.rerank_items(query, seed_items, top_k=k)
            packet.node_ids = [nid for nid, _ in ordered] + context_ids
            packet.hubs = [text for _, text in ordered]
        elif seed_items:
            packet.node_ids = packet.node_ids[: min(seed_count, k)] + context_ids
            packet.hubs = packet.hubs[:k]
        # Abstention uses dense support for the returned seed passages only.
        # Do not let an unused candidate, RRF/PPR score, hop prior, or context
        # expansion raise the confidence of the answer.
        packet.confidence = self._dense_confidence(query_emb, packet.node_ids[: len(packet.hubs)])
        return packet

    def _global_search(self, query: str, top_k_basins: int, query_embedding: np.ndarray):
        assert self._global is not None
        try:
            return self._global.search_structured(
                query, top_k_basins=top_k_basins, query_embedding=query_embedding
            )
        except TypeError as exc:
            if "query_embedding" not in str(exc):
                raise
            return self._global.search_structured(query, top_k_basins=top_k_basins)

    def _dense_confidence(self, query_embedding: np.ndarray, node_ids) -> float:
        """Best raw dense cosine support; deliberately independent of rank/topology scores."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 0:
            return 0.0
        query = query / query_norm
        best = 0.0
        for nid in node_ids:
            if nid not in self.engine.graph:
                continue
            embedding = self.engine.graph.nodes[nid].get("embedding")
            if embedding is None:
                continue
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vector.shape != query.shape:
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            score = float(np.dot(query, vector / norm))
            if np.isfinite(score):
                best = max(best, score)
        return max(0.0, best)

    def _fill_from_nodes(self, packet: BriefingPacket, nodes, seed_ids=None, confidence=None):
        seen = set()
        for nid in seed_ids or []:
            if nid in seen or nid not in self.engine.graph:
                continue
            seen.add(nid)
            data = self.engine.graph.nodes[nid]
            packet.hubs.append(data.get("text", ""))
            packet.node_ids.append(nid)
        for item in nodes:
            nid = item["id"]
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
        if confidence is None:
            # Accept only explicitly named dense-support fields. `score` and
            # `rank_score` may include RRF/PPR/hops and must never gate chat.
            confidence = max(
                (float(item.get("confidence", item.get("dense_score", 0.0)) or 0.0) for item in nodes),
                default=0.0,
            )
        packet.confidence = max(0.0, float(confidence or 0.0))

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
