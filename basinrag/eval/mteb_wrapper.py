import sys

# Patch PyTorch DTensor para compatibilidade do SentenceTransformers/Transformers no PyTorch 2.4
try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

import numpy as np
from typing import Dict, Any, Optional

from mteb.models.model_meta import ModelMeta


class SkipLargeCorpus(RuntimeError):
    """Corpus exceeds the configured doc limit for this machine/run."""


class BasinRAGMTEBWrapper:
    """
    Wrapper para o MTEB (v2.20+) que implementa o SearchProtocol oficial para o BasinRAG
    com suporte a reranking seletivo de alta precisão (Top-K) e scores contínuos reais.
    """
    def __init__(
        self,
        rag_instance,
        search_type: str = "hybrid",
        rerank_top_k: int = 50,
        query_prompt: str = "Represent this sentence for searching relevant passages: ",
        max_queries: Optional[int] = None,
        force_reindex: bool = False,
        title_boost: int = 1,
        use_hop_prior: bool = True,
        use_rerank: Optional[bool] = None,
        cache_tag: str = "",
        auto_disable_rerank_on_flat: bool = True,
        max_corpus_docs: Optional[int] = None,
    ):
        self.rag = rag_instance
        self.search_type = search_type
        self.rerank_top_k = rerank_top_k
        self.query_prompt = query_prompt
        self.max_queries = max_queries
        self.force_reindex = force_reindex
        self.title_boost = max(1, int(title_boost))
        self.use_hop_prior = use_hop_prior
        # Default off for flat (1 node/doc) corpora — CE was the main published SciFact drop.
        self.use_rerank = use_rerank
        self.auto_disable_rerank_on_flat = auto_disable_rerank_on_flat
        self.cache_tag = cache_tag
        self.max_corpus_docs = max_corpus_docs
        self._is_flat_index = False

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        encode_kwargs: Any = None,
        num_proc: Optional[int] = None,
        **kwargs
    ) -> None:
        from basinrag.indexer.condensation import node_layers

        task_name = getattr(task_metadata, "name", None) or "unknown"
        expected_tag = self.cache_tag or f"{task_name}|{hf_split}|{hf_subset}"
        stored_tag = getattr(self.rag.engine, "gate_cache_tag", "")
        cache_ok = (
            not self.force_reindex
            and self.rag.persistence.load_topology(self.rag.engine)
            and len(self.rag.engine.graph) > 0
            and (not stored_tag or stored_tag == expected_tag)
        )
        if cache_ok:
            print(f"[MTEB Wrapper] Cache hit ({len(self.rag.engine.graph)} nós, tag={expected_tag})")
            self.rag._attach_bm25()
            self.rag.retriever = None
            self._is_flat_index = self._detect_flat_index()
            return

        def _join_title_text(title: str, body: str) -> str:
            t = (title or "").strip()
            x = (body or "").strip()
            if t and x:
                return f"{t} {x}" if self.title_boost <= 1 else " ".join([t] * self.title_boost + [x])
            return t or x

        if isinstance(corpus, dict):
            doc_ids = list(corpus.keys())
            doc_texts = []
            for did in doc_ids:
                item = corpus[did]
                doc_texts.append(_join_title_text(item.get("title") or "", item.get("text") or ""))
        else:
            # HuggingFace Dataset
            doc_ids = list(corpus["id"] if "id" in corpus.column_names else corpus["_id"])
            titles = list(corpus["title"]) if "title" in corpus.column_names else [""] * len(doc_ids)
            texts = list(corpus["text"])
            doc_texts = [_join_title_text(t, x) for t, x in zip(titles, texts)]

        print(f"\n[MTEB Wrapper] Indexando {len(doc_texts)} documentos no BasinRAG ({self.search_type})...")
        if self.max_corpus_docs is not None and len(doc_texts) > self.max_corpus_docs:
            raise SkipLargeCorpus(
                f"{len(doc_texts)} docs > max_corpus_docs={self.max_corpus_docs} "
                f"(use --max-corpus-docs 0 para forçar)"
            )

        batch_size = 64
        all_embeddings = self.rag.ingestor.encoder.encode(
            doc_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        nodes = []
        for doc_id, text, emb in zip(doc_ids, doc_texts, all_embeddings):
            layers = node_layers(text)
            nodes.append({
                "id": str(doc_id),
                "text": text,
                "embedding": emb,
                "source": str(doc_id),
                "chunk_index": 0,
                "l1": layers["l1"],
                "l2": layers["l2"],
                "metadata": {"doc_id": str(doc_id), "id": str(doc_id)},
            })

        print("[MTEB Wrapper] Construindo Bacias de Atração e Grafo Funcional...")
        self.rag.engine.encoder_model = self.rag.config.encoder_model
        self.rag.engine.gate_cache_tag = expected_tag
        self.rag.engine.build_graph(nodes)
        self.rag.engine.partition_into_basins()
        self.rag._attach_bm25()
        self.rag.retriever = None
        self.rag.persistence.save_topology(self.rag.engine)
        self._is_flat_index = True  # MTEB wrapper always indexes 1 node per doc

    def _detect_flat_index(self) -> bool:
        engine = self.rag.engine
        n = engine.graph.number_of_nodes()
        if n == 0:
            return True
        chunk_indexes = {
            int(data.get("chunk_index", 0))
            for _, data in engine.graph.nodes(data=True)
        }
        return chunk_indexes == {0} and len(engine.basins) >= max(1, int(0.9 * n))

    def _rerank_enabled(self) -> bool:
        if self.use_rerank is not None:
            return bool(self.use_rerank)
        if self.auto_disable_rerank_on_flat and self._is_flat_index:
            return False
        return True

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        top_k: int = 10,
        encode_kwargs: Any = None,
        top_ranked: Any = None,
        num_proc: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        if isinstance(queries, dict):
            query_dict = queries
        else:
            q_ids = list(queries["id"] if "id" in queries.column_names else queries["_id"])
            q_texts = list(queries["text"])
            query_dict = {str(qid): text for qid, text in zip(q_ids, q_texts)}

        if self.max_queries and len(query_dict) > self.max_queries:
            query_dict = dict(list(query_dict.items())[: self.max_queries])

        use_ce = self._rerank_enabled()
        print(
            f"[MTEB Wrapper] Busca para {len(query_dict)} queries "
            f"(top_k={top_k}, rerank={use_ce}, flat={self._is_flat_index})..."
        )
        results: Dict[str, Dict[str, float]] = {}

        self.rag._ensure_retriever(search_type=self.search_type, top_k=top_k)
        retriever = self.rag.retriever
        reranker = retriever._reranker

        for idx, (qid, qtext) in enumerate(query_dict.items(), 1):
            if idx % 50 == 0 or idx == 1 or idx == len(query_dict):
                print(f"[MTEB Wrapper] Progresso: {idx}/{len(query_dict)} queries ({idx/len(query_dict)*100:.0f}%)...", flush=True)

            # 1. Query embedding — retriever applies BGE prompt canonically
            query_emb = retriever._encode_query(qtext)

            # 2. Busca híbrida ampla (BM25 + Grafo Topológico)
            pool_k = max(50, top_k * 5) if top_k < 50 else top_k
            if self.search_type == "local":
                ranked_nodes = retriever._local.search_nodes(query_emb, top_k=pool_k)
            else:
                ranked_nodes = retriever._hybrid.search_nodes(
                    qtext,
                    query_emb,
                    top_k=pool_k,
                    use_hop_prior=self.use_hop_prior,
                    use_confidence_gate=False,
                    expand_graph=not self._is_flat_index,
                )

            if not ranked_nodes:
                results[str(qid)] = {}
                continue

            # 3. Reranking seletivo com Cross-Encoder no Top K candidatos
            k_rerank = min(len(ranked_nodes), self.rerank_top_k)
            top_candidates = ranked_nodes[:k_rerank]
            tail_candidates = ranked_nodes[k_rerank:]

            doc_scores: Dict[str, float] = {}

            if use_ce and reranker and getattr(reranker, "_model", None) != "disabled" and k_rerank > 0:
                passages = [c["text"] for c in top_candidates]
                raw_scores = reranker.predict_scores(qtext, passages)

                top_scored = []
                for cand, raw_score in zip(top_candidates, raw_scores):
                    # Monotonic: keep the cross-encoder score as-is. Ranking only needs order.
                    top_scored.append((cand, float(raw_score)))

                top_scored.sort(key=lambda x: x[1], reverse=True)

                min_top_score = 1.0
                for cand, final_score in top_scored:
                    doc_id = str(cand.get("metadata", {}).get("doc_id") or cand.get("id"))
                    if doc_id and doc_id not in doc_scores:
                        doc_scores[doc_id] = final_score
                        min_top_score = min(min_top_score, final_score)

                # Cauda longa (k_rerank até top_k): decaimento contínuo respeitando o score híbrido
                base_tail = min_top_score * 0.95
                tail_max_s = max((float(c.get("score") or 0.0) for c in tail_candidates), default=1.0) or 1.0
                for rank_offset, c in enumerate(tail_candidates, 1):
                    doc_id = str(c.get("metadata", {}).get("doc_id") or c.get("id"))
                    if doc_id and doc_id not in doc_scores:
                        c_score = float(c.get("score") or 0.0) / tail_max_s
                        decay = 1.0 / (1.0 + 0.003 * rank_offset)
                        doc_scores[doc_id] = base_tail * (0.5 * c_score + 0.5 * decay)
                    if len(doc_scores) >= top_k:
                        break
            else:
                for rank, c in enumerate(ranked_nodes):
                    doc_id = str(c.get("metadata", {}).get("doc_id") or c.get("id"))
                    if doc_id and doc_id not in doc_scores:
                        score = float(c.get("score") or (1.0 / (rank + 1.0)))
                        doc_scores[doc_id] = score
                    if len(doc_scores) >= top_k:
                        break

            results[str(qid)] = doc_scores

        return results

    @property
    def mteb_model_meta(self) -> ModelMeta:
        from .. import __version__
        encoder_name = getattr(self.rag.config, "encoder_model", "BAAI/bge-base-en-v1.5")
        embed_dim = 768 if "base" in encoder_name else 384
        reranker_name = getattr(self.rag.config, "reranker_model", "") or ""
        n_params = 110_000_000 if embed_dim == 768 else 33_000_000
        if "bge-reranker-v2-m3" in reranker_name:
            n_params += 568_000_000
        return ModelMeta(
            name="Basinfy/BasinRAG",
            revision=__version__,
            release_date="2026-09-10",
            languages=["eng", "por"],
            framework=["PyTorch", "Sentence Transformers"],
            similarity_fn_name="cosine",
            use_instructions=bool(self.query_prompt),
            reference="https://github.com/Basinfy/BasinRAG",
            license="apache-2.0",
            model_type=["hybrid"],
            loader=None,
            n_parameters=n_params,
            memory_usage_mb=450.0 if embed_dim == 384 else 1400.0,
            max_tokens=512,
            embed_dim=embed_dim,
            open_weights=True,
            public_training_code="https://github.com/Basinfy/BasinRAG",
            public_training_data=None,
            training_datasets=None,
        )


