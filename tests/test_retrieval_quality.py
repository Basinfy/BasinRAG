import numpy as np
import pytest
from unittest.mock import patch

from basinrag.eval.datasets import get_mmarco_subset
from basinrag.eval.metrics import evaluate_retrieval


class DummyEncoder:
    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.stack([self.encode(t) for t in text])
        rng = np.random.default_rng(abs(hash(str(text))) % (2**32))
        vec = rng.random(384).astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm


class DummyIngestor:
    def __init__(self, *args, **kwargs):
        self.encoder = DummyEncoder()
        self.chunk_size = kwargs.get("chunk_size", 1000)
        self.chunk_overlap = kwargs.get("chunk_overlap", 100)

    def ingest(self, path):
        return []

    def ingest_directory(self, path):
        return []


@pytest.fixture(scope="module")
def mmarco_rag(tmp_path_factory):
    storage = tmp_path_factory.mktemp("mmarco_idx")
    with patch("basinrag.factory.BasinIngestor", DummyIngestor):
        from basinrag.factory import BasinRAG, BasinRAGConfig
        rag = BasinRAG(BasinRAGConfig(storage_dir=str(storage)))

    from basinrag.indexer.condensation import node_layers

    corpus, _ = get_mmarco_subset()
    chunks = []
    for i, doc in enumerate(corpus):
        chunks.append({
            "id": doc["id"],
            "text": doc["text"],
            "embedding": np.random.rand(384).astype(np.float32),
            "source": "mock",
            "chunk_index": i,
            "l1": node_layers(doc["text"])["l1"],
            "l2": node_layers(doc["text"])["l2"],
            "metadata": {"id": doc["id"]},
        })
    rag.engine.build_graph(chunks)
    rag.engine.partition_into_basins()
    rag._attach_bm25()
    rag._ensure_retriever()
    rag.retriever._reranker._model = "disabled"
    return rag


def test_hybrid_mrr_above_threshold(mmarco_rag):
    corpus, queries = get_mmarco_subset()
    id_to_text = {d["id"]: d["text"] for d in corpus}
    qrels = {q["qid"]: set(id_to_text[rid] for rid in q["relevant_doc_ids"]) for q in queries}

    results = {}
    for q in queries:
        docs = mmarco_rag.query(q["query"], search_type="hybrid", top_k=5)
        results[q["qid"]] = [str(d.page_content) for d in docs]

    metrics = evaluate_retrieval(qrels, results, k=5)
    assert metrics["mrr"] > 0.0


def test_faquad_document_retrieval():
    qrels = {"q1": {"doc1"}}
    results = {"q1": ["doc2", "doc1"]}
    metrics = evaluate_retrieval(qrels, results, k=5)
    assert metrics["mrr"] == 0.5
    assert metrics["hit_rate"] == 1.0
