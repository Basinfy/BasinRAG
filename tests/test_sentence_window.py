from types import SimpleNamespace

import networkx as nx
import numpy as np

from basinrag.eval import run_gate
from basinrag.indexer.ingestor import SENTENCE_WINDOW_RADIUS, split_sentence_leaves


class _FakeEngine:
    def __init__(self):
        self.graph = nx.Graph()
        self.basins = {}
        self.encoder_model = ""
        self.build_id = ""
        self.index_metadata = {}

    def build_graph(self, nodes):
        self.graph.clear()
        for node in nodes:
            self.graph.add_node(node["id"], **node)

    def partition_into_basins(self):
        self.basins = {node_id: object() for node_id in self.graph}


class _FakeEncoder:
    tokenizer = SimpleNamespace(name_or_path="fake/tokenizer")

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array(
            [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


class _FakePersistence:
    def __init__(self):
        self.save_calls = 0

    def current_build_id(self):
        return "base-build"

    def save_topology(self, engine):
        self.save_calls += 1
        return True


class _FakeRag:
    def __init__(self):
        self.engine = _FakeEngine()
        self.persistence = _FakePersistence()
        self.ingestor = SimpleNamespace(
            encoder=_FakeEncoder(),
            splitter=SimpleNamespace(tokenizer=_FakeEncoder.tokenizer),
            model_revision="a" * 40,
        )
        self.config = SimpleNamespace(
            encoder_model="fake/encoder",
            encoder_revision="a" * 40,
            reranker_model="fake/reranker",
            reranker_revision="b" * 40,
            ranking_mode="hybrid_rrf",
        )
        self.retriever = None
        self.attach_calls = 0

    def _attach_bm25(self):
        self.attach_calls += 1


def test_split_sentence_leaves_keeps_short_units_under_cap():
    text = "First sentence. Second sentence. Third sentence. Fourth one."
    leaves = split_sentence_leaves(text, max_chars=80, overlap_chars=20, max_sentences=2)
    assert len(leaves) >= 2
    assert "First sentence" in leaves[0]
    assert all(len(leaf) <= 80 for leaf in leaves)
    assert split_sentence_leaves("   ") == []


def test_index_long_docs_builds_sentence_window_children(monkeypatch):
    monkeypatch.setattr(run_gate, "_set_gate_manifest", lambda *args, **kwargs: None)
    rag = _FakeRag()
    run_gate.index_long_docs(
        rag,
        {
            "paper": (
                "Methods use a convolutional encoder. Results improve nDCG. "
                "Limitations remain for long documents. Future work is retrieval."
            )
        },
        80,
        20,
    )
    assert len(rag.engine.graph) > 1
    for _, data in rag.engine.graph.nodes(data=True):
        assert data["source"] == "paper"
        span = data["metadata"]["parent_span"]
        assert span[0] >= 0
        assert span[1] >= span[0]
        assert data["metadata"]["role"] == "child"
        assert data["metadata"]["parent_doc"] == "paper"
    assert rag.attach_calls == 1
    assert rag.persistence.save_calls == 1
    assert SENTENCE_WINDOW_RADIUS == 3
