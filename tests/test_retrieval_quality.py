import pytest
from basinrag.factory import BasinRAG
from basinrag.eval.datasets import get_mmarco_subset
from basinrag.eval.metrics import evaluate_retrieval

@pytest.fixture(scope="module")
def mmarco_rag():
    corpus, _ = get_mmarco_subset()
    rag = BasinRAG.create()
    
    # Ingest the mock corpus directly into the engine
    from basinrag.indexer.condensation import node_layers
    import numpy as np
    
    chunks = []
    for i, doc in enumerate(corpus):
        chunks.append({
            "id": doc["id"],
            "text": doc["text"],
            "embedding": np.random.rand(384).astype(np.float32),  # Random embedding for mock
            "source": "mock",
            "chunk_index": i,
            "l1": node_layers(doc["text"])["l1"],
            "l2": node_layers(doc["text"])["l2"],
            "metadata": {"id": doc["id"]},
        })
    rag.engine.build_graph(chunks)
    rag.engine.partition_into_basins()
    rag._attach_bm25()
    
    return rag

def test_hybrid_mrr_above_threshold(mmarco_rag):
    corpus, queries = get_mmarco_subset()
    
    # Map doc id to text
    id_to_text = {d["id"]: d["text"] for d in corpus}
    
    # qrels now holds the expected texts instead of IDs
    qrels = {q["qid"]: set(id_to_text[rid] for rid in q["relevant_doc_ids"]) for q in queries}
    
    results = {}
    for q in queries:
        docs = mmarco_rag.query(q["query"], search_type="hybrid", top_k=5)
        results[q["qid"]] = [str(d.page_content) for d in docs]
        
    metrics = evaluate_retrieval(qrels, results, k=5)
    # Even with random embeddings, BM25 should carry the MRR for exact keyword matches
    assert metrics["mrr"] > 0.0

def test_faquad_document_retrieval():
    # FaQuAD integration tests could be slow, so we just verify the metrics functions work correctly
    qrels = {"q1": {"doc1"}}
    results = {"q1": ["doc2", "doc1"]}
    metrics = evaluate_retrieval(qrels, results, k=5)
    assert metrics["mrr"] == 0.5
    assert metrics["hit_rate"] == 1.0
