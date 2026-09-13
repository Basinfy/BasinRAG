import hashlib
import re

import numpy as np
import pytest
from unittest.mock import patch

from basinrag.eval.datasets import get_mmarco_subset
from basinrag.eval.metrics import evaluate_retrieval


class DummyEncoder:
    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.stack([self.encode(t) for t in text])
        vec = np.zeros(384, dtype=np.float32)
        for token in re.findall(r"[\w]+", str(text).lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % len(vec)
            vec[slot] += 1.0
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
        rag = BasinRAG(BasinRAGConfig(
            storage_dir=str(storage), use_rerank=False, query_prompt=""
        ))

    rag.ingestor = DummyIngestor()
    from basinrag.indexer.condensation import node_layers

    corpus, _ = get_mmarco_subset()
    chunks = []
    for i, doc in enumerate(corpus):
        chunks.append({
            "id": doc["id"],
            "text": doc["text"],
            "embedding": DummyEncoder().encode(doc["text"]),
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
    return rag


def test_hybrid_mrr_above_threshold(mmarco_rag):
    _, queries = get_mmarco_subset()
    qrels = {q["qid"]: set(q["relevant_doc_ids"]) for q in queries}

    results = {}
    for q in queries:
        packet = mmarco_rag.brief(q["query"], search_type="hybrid", top_k=5)
        results[q["qid"]] = packet.node_ids[:10]

    metrics = evaluate_retrieval(qrels, results, k=10)
    assert metrics["mrr"] >= 0.9, (metrics, results)
    assert metrics["ndcg"] >= 0.9, (metrics, results)
    assert metrics["recall"] >= 0.95, (metrics, results)


def test_faquad_document_retrieval():
    qrels = {"q1": {"doc1"}}
    results = {"q1": ["doc2", "doc1"]}
    metrics = evaluate_retrieval(qrels, results, k=5)
    assert metrics["mrr"] == 0.5
    assert metrics["hit_rate"] == 1.0
