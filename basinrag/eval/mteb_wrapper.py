import sys

# Patch PyTorch DTensor para compatibilidade do SentenceTransformers/Transformers no PyTorch 2.4
try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

from typing import Dict, Any, Optional

from mteb.models.model_meta import ModelMeta


class BasinRAGMTEBWrapper:
    """
    Wrapper para o MTEB (v2.20+) que implementa o SearchProtocol oficial para o BasinRAG 2.0.
    """
    def __init__(self, rag_instance, search_type: str = "hybrid"):
        self.rag = rag_instance
        self.search_type = search_type

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

        if isinstance(corpus, dict):
            doc_ids = list(corpus.keys())
            doc_texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip() for did in doc_ids]
        else:
            # HuggingFace Dataset
            doc_ids = list(corpus["id"] if "id" in corpus.column_names else corpus["_id"])
            titles = list(corpus["title"]) if "title" in corpus.column_names else [""] * len(doc_ids)
            texts = list(corpus["text"])
            doc_texts = [f"{t} {x}".strip() for t, x in zip(titles, texts)]

        print(f"\n[MTEB Wrapper] Indexando {len(doc_texts)} documentos no BasinRAG ({self.search_type})...")

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
        self.rag.engine.build_graph(nodes)
        self.rag.engine.partition_into_basins()
        self.rag.engine.build_meta_basins()
        self.rag._attach_bm25()
        self.rag.retriever = None

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

        print(f"[MTEB Wrapper] Executando busca para {len(query_dict)} queries no BasinRAG (top_k={top_k})...")
        results: Dict[str, Dict[str, float]] = {}

        for qid, qtext in query_dict.items():
            docs = self.rag.query(qtext, search_type=self.search_type, top_k=max(20, top_k * 2))
            doc_scores: Dict[str, float] = {}
            for rank, d in enumerate(docs):
                meta = getattr(d, "metadata", {}) or {}
                found_id = str(meta.get("doc_id") or meta.get("id") or meta.get("source") or meta.get("node_id") or "")
                if found_id and found_id not in doc_scores:
                    score = float(getattr(d, "score", 0.0) or (1.0 / (rank + 1.0)))
                    doc_scores[found_id] = score
                if len(doc_scores) >= top_k:
                    break
            results[str(qid)] = doc_scores

        return results

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return ModelMeta(
            name="alexmart1ns/BasinRAG-2.0",
            revision="2.0.0",
            release_date="2026-09-05",
            languages=["eng", "por"],
            framework=["PyTorch", "Sentence Transformers"],
            similarity_fn_name="cosine",
            use_instructions=False,
            reference="https://github.com/alexmart1ns/BasinRAG",
            license="mit",
            model_type=["hybrid"],
            loader=None,
            n_parameters=118_000_000,
            memory_usage_mb=450.0,
            max_tokens=512,
            embed_dim=384,
            open_weights=True,
            public_training_code="https://github.com/alexmart1ns/BasinRAG",
            public_training_data=None,
            training_datasets=None,
        )


