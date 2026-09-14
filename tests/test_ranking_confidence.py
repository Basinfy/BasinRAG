from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from basinrag.indexer.bm25 import BM25Index
from basinrag.retriever.base import BasinRAGRetriever
from basinrag.retriever.briefing import BriefingPacket
from basinrag.retriever.hybrid_search import HybridSearch, union_bm25_hits
from basinrag.retriever.router import IntelligentQueryRouter


class _Engine:
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_node("seed-a", text="seed A", embedding=np.array([0.8, 0.6], dtype=np.float32), chunk_index=0)
        self.graph.add_node("seed-b", text="seed B", embedding=np.array([0.6, 0.8], dtype=np.float32), chunk_index=0)
        self.graph.add_node("topology-c", text="topology context", embedding=np.array([0.25, 0.96824586], dtype=np.float32), chunk_index=0)
        self.basins = {nid: object() for nid in ("seed-a", "seed-b", "topology-c")}


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


def test_default_mode_hydrates_basin_siblings_after_rrf_seeds(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch, search_type="hybrid")
    tree = nx.DiGraph()
    tree.add_nodes_from(["seed-a", "topology-c"])
    retriever.engine.graph.nodes["seed-a"]["basin_id"] = "basin-a"
    retriever.engine.basins = {"basin-a": SimpleNamespace(rho_tree=tree)}

    packet = retriever.brief("a representative question", top_k=2)

    assert packet.node_ids[:2] == ["seed-a", "seed-b"]
    assert packet.hubs == ["seed A", "seed B"]
    assert "topology-c" in packet.node_ids
    assert "topology context" in packet.neighbors


def test_brief_skips_cross_encoder_by_default_on_longdoc(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch, search_type="hybrid")
    assert retriever.use_rerank is False
    retriever.engine.graph.nodes["seed-b"]["chunk_index"] = 1
    retriever.engine.basins = {"b": object()}
    calls = []

    class _RR:
        def rerank_items(self, _query, items, top_k=None):
            calls.append(list(items))
            return list(reversed(items))[:top_k]

    retriever._reranker = _RR()
    packet = retriever.brief("a representative question", top_k=2)
    assert calls == []
    assert packet.node_ids[:2] == ["seed-a", "seed-b"]


def test_brief_skips_cross_encoder_on_flat_index(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch, search_type="hybrid")
    retriever.use_rerank = True
    calls = []

    class _RR:
        def rerank_items(self, _query, items, top_k=None):
            calls.append(list(items))
            return list(items)[:top_k]

    retriever._reranker = _RR()
    packet = retriever.brief("a representative question", top_k=2)
    assert calls == []
    assert packet.node_ids[:2] == ["seed-a", "seed-b"]


def test_brief_reranks_longdoc_pool_after_rrf(monkeypatch):
    retriever, _hybrid = _make_retriever(monkeypatch, search_type="hybrid")
    retriever.use_rerank = True
    retriever.engine.graph.nodes["seed-b"]["chunk_index"] = 1
    retriever.engine.basins = {"b": object()}
    calls = []

    class _RR:
        def rerank_items(self, _query, items, top_k=None):
            calls.append(list(items))
            return list(reversed(items))[:top_k]

    retriever._reranker = _RR()
    packet = retriever.brief("a representative question", top_k=2)
    assert calls
    assert packet.node_ids[:2] == ["seed-b", "seed-a"]


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


def test_hybrid_uses_floor_alpha_on_flat_index(monkeypatch):
    captured = {}

    def _capture(bm25_ids, semantic_ids, **kwargs):
        captured["alpha"] = kwargs.get("alpha")
        captured["allowlist"] = kwargs.get("bm25_allowlist")
        from basinrag.retriever.fusion import weighted_rrf as orig
        return orig(bm25_ids, semantic_ids, **kwargs)

    monkeypatch.setattr("basinrag.retriever.hybrid_search.weighted_rrf", _capture)

    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("seed-a", 1.0), ("seed-b", 0.5)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.bm25 = _BM25()
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert captured["alpha"] == pytest.approx(0.15)


def test_hybrid_uses_longdoc_alpha_when_chunked(monkeypatch):
    captured = {}

    def _capture(bm25_ids, semantic_ids, **kwargs):
        captured["alpha"] = kwargs.get("alpha")
        from basinrag.retriever.fusion import weighted_rrf as orig
        return orig(bm25_ids, semantic_ids, **kwargs)

    monkeypatch.setattr("basinrag.retriever.hybrid_search.weighted_rrf", _capture)

    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("seed-a", 1.0), ("seed-b", 0.5)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.nodes["seed-b"]["chunk_index"] = 1
    engine.basins = {"basin-a": object()}
    engine.bm25 = _BM25()
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert captured["alpha"] == pytest.approx(0.40)


def test_hybrid_longdoc_rescues_bm25_only_papers():
    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("lexical-only", 9.0), ("seed-a", 1.0)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.nodes["seed-b"]["chunk_index"] = 1
    engine.graph.add_node(
        "lexical-only",
        text="rare token match",
        chunk_index=2,
        source="paper-lex",
        metadata={"doc_id": "paper-lex"},
    )
    engine.basins = {"b": object()}
    engine.bm25 = _BM25()
    hits = HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert [item["id"] for item in hits] == ["seed-a", "lexical-only"]


def test_hybrid_longdoc_ranks_documents_before_repeating_chunks():
    class _BM25:
        n = 4

        def score(self, _query, top_k):
            return [("a0", 4.0), ("a1", 3.0), ("a2", 2.0), ("b0", 1.0)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [
                {"id": "a0", "score": 0.99},
                {"id": "a1", "score": 0.98},
                {"id": "a2", "score": 0.97},
                {"id": "b0", "score": 0.90},
            ][:top_k]

    engine = _Engine()
    for nid, source, chunk_index in (
        ("a0", "paper-a", 0),
        ("a1", "paper-a", 1),
        ("a2", "paper-a", 2),
        ("b0", "paper-b", 0),
    ):
        engine.graph.add_node(
            nid,
            text=nid,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            chunk_index=chunk_index,
            source=source,
            metadata={"doc_id": source},
        )
    engine.basins = {"b": object()}
    engine.bm25 = _BM25()
    hits = HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert [item["id"] for item in hits] == ["a0", "b0"]


def test_hybrid_longdoc_requests_a_deeper_candidate_pool():
    seen = []

    class _BM25:
        n = 2

        def score(self, _query, top_k):
            seen.append(top_k)
            return [("a0", 1.0), ("b0", 0.5)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            seen.append(("dense", top_k))
            return [{"id": "a0", "score": 0.9}, {"id": "b0", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.nodes["seed-b"]["chunk_index"] = 1
    engine.graph.add_node("a0", text="a", chunk_index=0, metadata={"doc_id": "A"})
    engine.graph.add_node("b0", text="b", chunk_index=1, metadata={"doc_id": "B"})
    engine.basins = {"b": object()}
    engine.bm25 = _BM25()
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=10, use_confidence_gate=False
    )
    assert seen[0] == 200
    assert seen[1] == ("dense", 200)


def test_hybrid_longdoc_runs_second_bm25_pass_without_llm():
    calls = []

    class _BM25:
        n = 2

        def score(self, query, top_k):
            calls.append(query)
            return [("seed-a", 1.0)][:top_k]

        def expansion_terms(self, _query, _texts, n_terms=8):
            return ["mitochondria"]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.nodes["seed-b"]["chunk_index"] = 1
    engine.graph.nodes["seed-a"]["text"] = "mitochondria produce atp"
    engine.basins = {"b": object()}
    engine.bm25 = _BM25()
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert calls == ["question", "question mitochondria"]


def test_hybrid_longdoc_unions_bm25_only_docs_into_allowlist(monkeypatch):
    captured = {}

    def _capture(bm25_ids, semantic_ids, **kwargs):
        captured["allowlist"] = kwargs.get("bm25_allowlist")
        from basinrag.retriever.fusion import weighted_rrf as orig
        return orig(bm25_ids, semantic_ids, **kwargs)

    monkeypatch.setattr("basinrag.retriever.hybrid_search.weighted_rrf", _capture)

    class _BM25:
        n = 2

        def score(self, _query, top_k):
            return [("lexical-only", 9.0), ("seed-a", 1.0)][:top_k]

    class _DenseSearch:
        def dense_hits(self, _query_embedding, top_k):
            return [{"id": "seed-a", "score": 0.9}, {"id": "seed-b", "score": 0.8}][:top_k]

    engine = _Engine()
    engine.graph.nodes["seed-b"]["chunk_index"] = 1
    engine.graph.add_node(
        "lexical-only",
        text="rare token match",
        chunk_index=2,
        source="paper-lex",
        metadata={"doc_id": "paper-lex"},
    )
    engine.basins = {"b": object()}
    engine.bm25 = _BM25()
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert "lexical-only" in captured["allowlist"]


def test_hybrid_flat_allowlist_still_excludes_lexical_only(monkeypatch):
    captured = {}

    def _capture(bm25_ids, semantic_ids, **kwargs):
        captured["allowlist"] = kwargs.get("bm25_allowlist")
        from basinrag.retriever.fusion import weighted_rrf as orig
        return orig(bm25_ids, semantic_ids, **kwargs)

    monkeypatch.setattr("basinrag.retriever.hybrid_search.weighted_rrf", _capture)

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
    HybridSearch(engine, _DenseSearch()).search_nodes(
        "question", np.array([1.0, 0.0]), top_k=2, use_confidence_gate=False
    )
    assert "lexical-only" not in captured["allowlist"]


def test_union_bm25_hits_keeps_primary_order_and_appends_new_ids():
    merged = union_bm25_hits(
        [("a", 5.0), ("b", 4.0)],
        [("c", 9.0), ("a", 1.0)],
    )
    ids = [nid for nid, _ in merged]
    assert ids[0] == "a"
    assert "c" in ids
    assert dict(merged)["a"] == pytest.approx(5.0)


def test_bm25_expansion_terms_are_idf_weighted_and_exclude_query():
    index = BM25Index()
    index.build(
        ["d1", "d2", "d3"],
        [
            "mitochondria produce atp in eukaryotic cells",
            "chloroplasts capture sunlight in plant cells",
            "ribosomes synthesize proteins in all cells",
        ],
    )
    terms = index.expansion_terms(
        "what do mitochondria produce",
        ["mitochondria produce atp in eukaryotic cells"],
        n_terms=5,
    )
    assert "atp" in terms
    assert "mitochondria" not in terms
    assert "produce" not in terms
    assert index.expansion_terms("empty", [], n_terms=5) == []
