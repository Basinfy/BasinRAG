from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from basinrag.eval.gate import GateSearcher
from basinrag.factory import BasinRAG, BasinRAGConfig
from basinrag.indexer.ingestor import (
    ADAPTIVE_THRESHOLD_CHARS,
    split_document_chunks,
)
from basinrag.retriever.briefing import BriefingPacket
from basinrag.retriever.hybrid_search import HybridSearch


def test_query_skips_missing_graph_nodes():
    rag = BasinRAG.__new__(BasinRAG)
    packet = BriefingPacket()
    packet.hubs = ["present text", "ghost text"]
    packet.node_ids = ["present", "ghost"]
    packet.neighbors = []

    class Graph:
        def __init__(self):
            self.nodes = {
                "present": {"text": "present text", "source": "a.txt", "metadata": {}},
            }

        def __contains__(self, node_id):
            return node_id in self.nodes

    engine = SimpleNamespace(graph=Graph())
    retriever = SimpleNamespace(brief=lambda *_args, **_kwargs: packet, top_k=5)
    rag._capture_retriever = lambda: (retriever, engine)
    docs = rag.query("question")
    assert [doc.metadata["node_id"] for doc in docs] == ["present"]


def test_search_nodes_confidence_gate_defaults_off():
    import inspect

    default = inspect.signature(HybridSearch.search_nodes).parameters["use_confidence_gate"].default
    assert default is False


def test_gate_and_product_share_candidate_k_contract():
    assert HybridSearch.resolve_candidate_k(10) == 50
    assert HybridSearch.resolve_candidate_k(10, long_doc=True) == 200
    assert HybridSearch.resolve_candidate_k(10, candidate_k=30) == 30


def test_gate_skips_cross_encoder_on_flat_index():
    graph = nx.Graph()
    graph.add_node("n1", metadata={"doc_id": "doc-a"}, source="src-a", text="alpha", chunk_index=0)

    class Hybrid:
        def search_nodes(self, *_args, **_kwargs):
            return [{"id": "n1", "text": "alpha", "score": 1.0}]

    class Reranker:
        def predict_scores(self, *_args, **_kwargs):
            raise AssertionError("flat index must skip cross-encoder")

    rag = SimpleNamespace(
        engine=SimpleNamespace(graph=graph, basins={"b1": object()}, bm25=None),
        retriever=SimpleNamespace(
            _hybrid=Hybrid(),
            _local=SimpleNamespace(dense_hits=lambda *_a, **_k: []),
        ),
        ingestor=SimpleNamespace(encoder=SimpleNamespace(encode=lambda text: np.array([1.0, 0.0]))),
        _ensure_retriever=lambda **_kwargs: None,
    )
    searcher = GateSearcher(rag, reranker=Reranker())
    assert searcher.search("q", "hybrid_min_rerank", top_k=10) == ["doc-a"]


def test_load_rejects_unresolved_encoder_revision():
    rag = BasinRAG.__new__(BasinRAG)
    rag.config = BasinRAGConfig(
        storage_dir=".basinrag-test",
        encoder_model="unconfigured",
        encoder_revision="a" * 40,
        chunk_size=512,
        chunk_overlap=128,
    )
    rag._ingestor = SimpleNamespace(
        splitter=SimpleNamespace(tokenizer=None),
        model_revision="a" * 40,
    )
    rag.engine = SimpleNamespace(
        index_schema_version=3,
        encoder_model="unconfigured",
        index_metadata={
            "format_version": 3,
            "encoder_model": "unconfigured",
            "encoder_revision": "unresolved",
            "tokenizer": "unconfigured",
            "tokenizer_revision": "a" * 40,
            "chunk_policy_version": 2,
            "chunking_mode": "adaptive",
            "chunk_size": 512,
            "chunk_overlap": 128,
            "adaptive_threshold_chars": 4000,
            "ranking_mode": "hybrid_rrf",
            "bm25_stemming": False,
            "bm25_stemmer_version": "disabled",
        },
    )
    with pytest.raises(ValueError, match="unresolved"):
        rag._validate_index_compatibility()


def test_validate_query_requires_persisted_bm25():
    rag = BasinRAG.__new__(BasinRAG)
    rag._loaded = True
    rag.config = BasinRAGConfig(use_rerank=False)
    rag.engine = SimpleNamespace(bm25=None)
    rag._capture_retriever = lambda: (SimpleNamespace(_reranker=None), rag.engine)
    with pytest.raises(RuntimeError, match="BM25"):
        rag.validate_query_dependencies()


def test_long_document_uses_sentence_window_leaves():
    splitter = SimpleNamespace(split_text=lambda text: [text])
    short, short_policy = split_document_chunks("short enough", splitter=splitter)
    assert short_policy == "legacy_char"
    assert short == ["short enough"]

    long_text = " ".join(
        f"Section {index} describes a distinct experimental method for retrieval."
        for index in range(120)
    )
    assert len(long_text) > ADAPTIVE_THRESHOLD_CHARS
    leaves, policy = split_document_chunks(long_text, splitter=splitter)
    assert policy == "sentence_window"
    assert len(leaves) > 1
    forced, forced_policy = split_document_chunks(
        "tiny", force_policy="sentence_window", long_size=80, long_overlap=20
    )
    assert forced_policy == "sentence_window"
    assert forced


def test_qasper_official_source_records_tarball_sha(monkeypatch):
    from basinrag.eval import run_gate

    official = run_gate._qasper_rows_from_official(
        {
            "1912.01214": {
                "title": "T",
                "abstract": "A",
                "full_text": [{"section_name": "S", "paragraphs": ["P"]}],
                "qas": [{"question": "Q", "question_id": "qid"}],
            }
        }
    )
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("qasper.py")),
    )
    monkeypatch.setattr(run_gate, "_load_official_qasper_v03", lambda: official)
    _corpus, queries, _qrels = run_gate.load_qasper(None, 1, revision="b" * 40)
    assert queries == {"qid": "Q"}
    assert run_gate.load_qasper.source == "official_v0.3_json"
    assert run_gate.load_qasper.dataset_revision == run_gate.QASPER_OFFICIAL_V03_SHA256
    assert len(run_gate.load_qasper.dataset_revision) == 64
