import networkx as nx
import numpy as np
import pytest

from basinrag.retriever.base import BasinRAGRetriever
from basinrag.retriever.briefing import BriefingPacket
from basinrag.retriever.hybrid_search import HybridSearch
from basinrag.retriever.router import IntelligentQueryRouter


class _Engine:
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_node("seed-a", text="seed A", embedding=np.array([0.8, 0.6], dtype=np.float32))
        self.graph.add_node("seed-b", text="seed B", embedding=np.array([0.6, 0.8], dtype=np.float32))
        self.graph.add_node("topology-c", text="topology context", embedding=np.array([0.25, 0.96824586], dtype=np.float32))
        self.basins = {}


class _Encoder:
    def encode(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class _Hybrid:
    def __init__(self):
        self.calls = []

    def search_nodes(self, query, query_embedding, **kwargs):
        self.calls.append(kwargs)
        return [
            {"id": "seed-a", "text": "seed A", "score": 0.02, "rank_score": 0.02},
            {"id": "seed-b", "text": "seed B", "score": 0.01, "rank_score": 0.01},
        ]


class _Local:
    def search_nodes(self, query_embedding, top_k=5):
        return [{"id": "topology-c", "text": "topology context", "score": 1000.0}]


class _Global:
    def search_structured(self, query, top_k_basins=5):
        return [
            {
                "basin_id": "basin-c",
                "score": 1000.0,
                "l3": "A global summary",
                "hubs": ["topology context"],
                "neighbors": [],
                "node_ids": ["topology-c"],
            }
        ]


def _make_retriever(monkeypatch, *, ranking_mode="hybrid_rrf", search_type="auto"):
    engine = _Engine()
    hybrid = _Hybrid()
    local = _Local()
    global_search = _Global()

    def install_stubs(self):
        self._local = local
        self._global = global_search
        self._hybrid = hybrid
        self._reranker = None

    monkeypatch.setattr(BasinRAGRetriever, "_rebuild_indexes", install_stubs)
    retriever = BasinRAGRetriever(
        engine=engine,
        encoder=_Encoder(),
        search_type=search_type,
        top_k=2,
        ranking_mode=ranking_mode,
        use_rerank=False,
    )
    return retriever, hybrid


@pytest.mark.parametrize(
    ("search_type", "auto_route"),
    [
        ("hybrid", None),
        ("local", None),
        ("global", None),
        ("auto", "hybrid"),
        ("auto", "local"),
        ("auto", "global"),
    ],
)
def test_default_mode_keeps_rrf_seed_order_across_routes(monkeypatch, search_type, auto_route):
    retriever, hybrid = _make_retriever(monkeypatch, search_type=search_type)
    if auto_route is not None:
        monkeypatch.setattr(
            IntelligentQueryRouter,
            "route_with_embeddings",
            staticmethod(lambda _query: auto_route),
        )

    packet = retriever.brief("a representative question", top_k=2)

    assert packet.node_ids[:2] == ["seed-a", "seed-b"]
    assert packet.hubs == ["seed A", "seed B"]
    assert packet.confidence == pytest.approx(0.8)
    assert hybrid.calls == [
        {
            "top_k": 20,
            "use_confidence_gate": False,
            "use_hop_prior": False,
            "use_multi_signal_drf": False,
            "expand_graph": False,
        }
    ]


@pytest.mark.parametrize("search_type", ["local", "global"])
def test_experimental_route_preserves_topological_override_but_not_its_confidence(
    monkeypatch, search_type
):
    retriever, hybrid = _make_retriever(
        monkeypatch, ranking_mode="experimental_topology", search_type=search_type
    )

    packet = retriever.brief("a representative question", top_k=2)

    assert packet.node_ids[0] == "topology-c"
    assert packet.confidence == pytest.approx(0.25)
    assert packet.confidence != 1000.0
    assert hybrid.calls == []


def test_experimental_hybrid_mode_keeps_explicit_topology_signals(monkeypatch):
    retriever, hybrid = _make_retriever(
        monkeypatch, ranking_mode="experimental_topology", search_type="hybrid"
    )

    packet = retriever.brief("a representative question", top_k=2)

    assert packet.node_ids == ["seed-a", "seed-b"]
    assert packet.confidence == pytest.approx(0.8)
    assert hybrid.calls[0]["use_hop_prior"] is True
    assert hybrid.calls[0]["use_multi_signal_drf"] is True
    assert hybrid.calls[0]["expand_graph"] is True


def test_langchain_retriever_invoke_returns_rrf_ranked_documents(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch, search_type="global")

    documents = retriever.invoke("a representative question")

    assert [document.metadata["node_id"] for document in documents] == ["seed-a", "seed-b"]


def test_fill_from_nodes_never_uses_ambiguous_rank_score_as_confidence(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch)
    packet = BriefingPacket()

    retriever._fill_from_nodes(
        packet,
        [{"id": "topology-c", "text": "topology context", "score": 1000.0, "rank_score": 1000.0}],
    )

    assert packet.confidence == 0.0


def test_default_hybrid_candidate_order_is_invariant_to_graph_topology():
    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("seed-b", 2.0), ("seed-a", 1.0)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            hits = [
                {"id": "seed-b", "score": 0.9},
                {"id": "seed-a", "score": 0.8},
            ]
            return hits[:top_k]

    engine = _Engine()
    engine.bm25 = _BM25()
    search = HybridSearch(engine, _DenseSearch())

    before = search.search_nodes("question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False)
    engine.graph.add_edge("seed-a", "topology-c", type="virtual-edge", weight=1.0)
    engine.graph.add_edge("seed-b", "seed-a", type="sequential", weight=1.0)
    engine.graph.nodes["seed-a"]["basin_id"] = "basin-a"
    engine.graph.nodes["seed-b"]["basin_id"] = "basin-b"
    engine.basins = {"basin-a": object(), "basin-b": object()}
    after = search.search_nodes("question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False)

    assert [item["id"] for item in before] == [item["id"] for item in after]
    assert [item["rank_score"] for item in before] == [item["rank_score"] for item in after]
    assert [item["confidence"] for item in before] == pytest.approx(
        [item["confidence"] for item in after]
    )


def test_hybrid_does_not_insert_lexical_only_outsiders():
    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("lexical-only", 9.0), ("seed-a", 1.0)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.add_node("lexical-only", text="rare token match")
    engine.bm25 = _BM25()
    hits = HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert [item["id"] for item in hits] == ["seed-a", "seed-b"]
