"""Deterministic coverage for graph primitives and retrieval edge cases.

These tests use tiny vectors and synthetic graph nodes only; they never load a
model or access the network.
"""
from __future__ import annotations

import numpy as np
import pytest

from basinrag.core.functional_graph import (
    adaptive_section_breaks,
    detect_attractors,
    members_by_attractor,
    reverse_hops,
    rho_parent_child,
    sequential_successor,
    sequential_successor_adaptive,
)
from basinrag.core.ids import make_node_id, tokenize
from basinrag.core.kv_store import DiskKVStore
from basinrag.core.topology import BasinTopologyEngine
from basinrag.core.vector_index import HNSW_THRESHOLD, build_ip_index
from basinrag.retriever.briefing import BriefingPacket, cap_satellite, estimate_tokens
from basinrag.retriever.global_search import TopologicalGlobalSearch
from basinrag.retriever.hybrid_search import HybridSearch
from basinrag.retriever.local_search import TopologicalLocalSearch
from basinrag.retriever.router import IntelligentQueryRouter


def _chunk(
    source: str,
    index: int,
    text: str,
    vector: tuple[float, ...],
) -> dict:
    return {
        "id": make_node_id(source, index, text),
        "text": text,
        "embedding": np.asarray(vector, dtype=np.float32),
        "source": source,
        "chunk_index": index,
        "metadata": {"fixture": True},
    }


def _engine(chunks: list[dict]) -> BasinTopologyEngine:
    engine = BasinTopologyEngine(open_stores=False, section_size=4)
    engine.build_graph(chunks)
    engine.partition_into_basins()
    return engine


class AxisEncoder:
    """Tiny phrase encoder with optional single-text and batch support."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()
        if any(word in lowered for word in ("overview", "main theme", "tema principal")):
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if any(word in lowered for word in ("local context", "this section", "neste parágrafo")):
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if "alpha" in lowered:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    def encode(self, texts, **_kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            return self._vector(texts)
        return np.stack([self._vector(text) for text in texts])


def test_functional_graph_handles_sections_cycles_missing_links_and_groups():
    assert sequential_successor({"a": ["a0", "a1", "a2"]}, section_size=0) == {
        "a0": None,
        "a1": None,
        "a2": None,
    }
    assert sequential_successor({"a": ["a0", "a1", "a2"]}, section_size=2) == {
        "a0": None,
        "a1": "a0",
        "a2": None,
    }

    successors = {"x": "y", "y": "x", "tail": "x", "sink": None, "orphan": "absent"}
    attractors = detect_attractors(successors)
    assert attractors == {
        "x": "x",
        "y": "x",
        "tail": "x",
        "sink": "sink",
        "orphan": "absent",
        "absent": "absent",
    }
    assert reverse_hops(successors, attractors) == {
        "x": 0,
        "y": 1,
        "tail": 1,
        "sink": 0,
        "orphan": 1,
        "absent": 0,
    }
    assert members_by_attractor(attractors) == {
        "x": ["x", "y", "tail"],
        "sink": ["sink"],
        "absent": ["orphan", "absent"],
    }
    assert rho_parent_child(successors, ["x", "y", "outside"]) == [("y", "x"), ("x", "y")]


def test_adaptive_sections_cover_missing_embeddings_similarity_and_maximum():
    ids = [f"n{i}" for i in range(7)]
    vectors = {
        "n0": [1.0, 0.0],
        "n1": [1.0, 0.0],
        "n2": [1.0, 0.0],
        "n3": [-1.0, 0.0],
        # n4 is missing and n5 has a zero vector; neither creates a break.
        "n5": [0.0, 0.0],
        "n6": [1.0, 0.0],
    }
    assert adaptive_section_breaks(
        ids,
        vectors,
        similarity_threshold=0.2,
        max_section=5,
        min_section=3,
    ) == [0, 3, 6]
    assert sequential_successor_adaptive(
        {"doc": ids[:3]}, {}, fallback_section_size=2
    ) == {"n0": None, "n1": "n0", "n2": None}
    assert sequential_successor_adaptive(
        {"doc": ids[:4]}, vectors, fallback_section_size=20
    )["n3"] is None


def test_ids_tokenize_scientific_codes_and_cjk_deterministically():
    assert make_node_id("a", 1, "text") == make_node_id("a", 1, "text")
    assert make_node_id("a", 1, "text") != make_node_id("a", 2, "text")
    assert tokenize("IL-6 SARS-COV-2 α蛋白 東京") == [
        "il-6",
        "il",
        "6",
        "sars-cov-2",
        "sars",
        "cov",
        "2",
        "蛋",
        "白",
        "東",
        "京",
    ]


def test_disk_kv_store_batches_streams_and_readonly_guards(tmp_path):
    db = tmp_path / "kv.sqlite3"
    writable = DiskKVStore(str(db), "records; DROP TABLE x")
    assert writable.table_name == "recordsDROPTABLEx"
    writable.set_many({"k1": {"n": 1}, "k2": [2], "k3": None})
    assert list(writable.iter_keys(batch_size=2)) == ["k1", "k2", "k3"]
    assert list(writable.iter_items(batch_size=2)) == [
        ("k1", {"n": 1}),
        ("k2", [2]),
        ("k3", None),
    ]
    assert list(writable.iter_values(batch_size=1)) == [{"n": 1}, [2], None]
    assert "k2" in writable and "missing" not in writable
    assert writable["k1"] == {"n": 1}
    with pytest.raises(KeyError):
        _ = writable["missing"]
    assert writable.get("missing", "fallback") == "fallback"
    assert writable.pop("missing", 9) == 9
    assert writable.pop("k2") == [2]
    writable["k4"] = "four"
    assert list(writable) == ["k1", "k3", "k4"]
    assert len(writable) == 3

    writable.close()
    readonly = DiskKVStore(str(db), "recordsDROPTABLEx", read_only=True)
    assert readonly.get("k1") == {"n": 1}
    for action in (
        lambda: readonly.set("x", 1),
        lambda: readonly.set_many({"x": 1}),
        lambda: readonly.pop("k1"),
        lambda: readonly.__setitem__("x", 2),
        readonly.clear,
        readonly.vacuum,
    ):
        with pytest.raises(RuntimeError, match="read-only"):
            action()
    readonly.close()
    writable = DiskKVStore(str(db), "recordsDROPTABLEx")
    writable.clear()
    assert len(writable) == 0
    writable.close()


def test_disk_kv_store_backup_restore_noops_and_missing_sources(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    store = DiskKVStore(str(source), "things")
    store.set("one", 1)
    store.backup_to(str(source))
    store.backup_to(str(backup))
    store.close()

    clone = DiskKVStore(str(restored), "things")
    clone.restore_from(str(tmp_path / "not-there.db"))
    assert len(clone) == 0
    clone.restore_from(str(backup))
    assert clone.get("one") == 1
    clone.restore_from(str(restored))
    assert clone.get("one") == 1
    clone.close()


def test_topology_partition_preserves_llm_l3_and_experimental_edge_boundaries():
    chunks = [
        _chunk("a.txt", 0, "alpha seed", (1, 0, 0)),
        _chunk("a.txt", 1, "alpha child", (1, 0, 0)),
        _chunk("b.txt", 0, "alpha elsewhere", (1, 0, 0)),
        _chunk("c.txt", 0, "orthogonal", (0, 1, 0)),
    ]
    engine = BasinTopologyEngine(open_stores=False, section_size=4)
    engine.build_graph(chunks)
    engine.partition_into_basins()
    assert engine.graph.number_of_nodes() == 4
    assert not any(data.get("type") == "virtual-edge" for *_edge, data in engine.graph.edges(data=True))

    basin_id = engine.basin_id_of(chunks[0]["id"])
    assert basin_id == chunks[0]["id"]
    engine.basins[basin_id].rho_tree.nodes[basin_id]["l3_summary"] = "saved model summary"
    engine.basins[basin_id].rho_tree.nodes[basin_id]["l3_source"] = "llm"
    engine.partition_into_basins(
        preserve_l3={basin_id: {"l3_summary": "saved model summary", "l3_source": "llm"}}
    )
    assert engine.basins[basin_id].rho_tree.nodes[basin_id]["l3_summary"] == "saved model summary"
    assert engine.hops_of(chunks[1]["id"]) == 1
    assert engine.basin_id_of("missing") == ""

    engine.partition_into_basins(experimental_topology=True)
    virtual = [(u, v) for u, v, data in engine.graph.edges(data=True) if data.get("type") == "virtual-edge"]
    assert virtual
    degree: dict[str, int] = {}
    for u, v in virtual:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert max(degree.values()) <= engine.MAX_SEMANTIC_DEGREE
    assert all(engine.basin_id_of(u) != engine.basin_id_of(v) for u, v in virtual)


def test_topology_experimental_link_stops_without_eligible_edges():
    chunks = [
        _chunk("same.txt", 0, "one", (1, 0)),
        _chunk("same.txt", 1, "two", (1, 0)),
        _chunk("same.txt", 2, "three", (1, 0)),
    ]
    engine = _engine(chunks)
    engine.partition_into_basins(experimental_topology=True)
    assert not [
        (u, v)
        for u, v, data in engine.graph.edges(data=True)
        if data.get("type") == "virtual-edge"
    ]


def test_vector_index_empty_flat_and_hnsw_paths(monkeypatch):
    faiss = pytest.importorskip("faiss")
    assert build_ip_index(np.empty((0, 3), dtype=np.float32)) is None
    assert build_ip_index(np.empty((2, 0), dtype=np.float32)) is None

    flat = build_ip_index(np.asarray([[3, 4], [0, 2]], dtype=np.float32))
    assert isinstance(flat, faiss.IndexFlatIP)
    distances, ids = flat.search(np.asarray([[1, 0]], dtype=np.float32), 2)
    assert ids.shape == (1, 2)
    assert distances[0, 0] == pytest.approx(0.6)

    monkeypatch.setattr("basinrag.core.vector_index.HNSW_THRESHOLD", 1)
    hnsw = build_ip_index(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    assert isinstance(hnsw, faiss.IndexHNSWFlat)
    assert HNSW_THRESHOLD == 2048


def test_local_search_handles_empty_index_dense_results_and_token_cap():
    empty_engine = BasinTopologyEngine(open_stores=False)
    empty = TopologicalLocalSearch(empty_engine)
    assert empty.dense_hits(np.asarray([1, 0])) == []
    assert empty.search_nodes(np.asarray([1, 0])) == []

    chunks = [
        _chunk("doc.txt", 0, "first passage with enough content", (1, 0)),
        _chunk("doc.txt", 1, "second passage with more content", (0.8, 0.6)),
        _chunk("other.txt", 0, "unrelated astronomy", (0, 1)),
    ]
    engine = _engine(chunks)
    local = TopologicalLocalSearch(engine)
    query = np.asarray([1, 0], dtype=np.float32)
    dense = local.dense_hits(query, top_k=2)
    assert [item["id"] for item in dense] == [chunks[0]["id"], chunks[1]["id"]]
    assert dense[0]["text"] == chunks[0]["text"]
    assert dense[0]["score"] == pytest.approx(1.0)

    local._faiss_hits = lambda _q, _k: [(chunks[0]["id"], 0.9), (chunks[1]["id"], 0.8)]
    hits = local.search_nodes(query, top_k=4, max_tokens=1)
    assert len(hits) == 1  # first relevant result survives even if it exceeds budget
    assert hits[0]["id"] == chunks[0]["id"]
    assert local._neighbors_of_type("not-in-graph", "sequential") == []
    local._faiss_hits = lambda _q, _k: []
    assert local.search_nodes(query) == []
    local._faiss_hits = lambda _q, _k: [(chunks[0]["id"], 0.01)]
    assert local.search_nodes(query) == []


def test_local_ppr_degenerate_dangling_and_uniform_fallback():
    chunks = [
        _chunk("ppr.txt", 0, "center", (1, 0)),
        _chunk("ppr.txt", 1, "leaf", (0, 1)),
    ]
    engine = _engine(chunks)
    local = TopologicalLocalSearch(engine)
    assert local._compute_local_ppr(set(), {}) == {}
    assert local._compute_local_ppr({chunks[0]["id"]}, {}) == {chunks[0]["id"]: 1.0}

    # Use a disconnected pair to exercise dangling mass and uniform teleport.
    engine.graph.remove_edges_from(list(engine.graph.edges()))
    nodes = {chunks[0]["id"], chunks[1]["id"]}
    scores = local._compute_local_ppr(nodes, {}, max_iter=50)
    assert scores == pytest.approx({chunks[0]["id"]: 1.0, chunks[1]["id"]: 1.0})


def test_global_search_batches_and_keys_l3_cache_by_snapshot():
    chunks = [
        _chunk("alpha.txt", 0, "alpha source", (1, 0)),
        _chunk("beta.txt", 0, "beta source", (0, 1)),
    ]
    engine = _engine(chunks)
    engine.build_id = "build-a"
    alpha_id = engine.basin_id_of(chunks[0]["id"])
    beta_id = engine.basin_id_of(chunks[1]["id"])
    for basin_id, summary in ((alpha_id, "alpha summary"), (beta_id, "beta summary")):
        data = engine.basins[basin_id].rho_tree.nodes[basin_id]
        data["l3_source"] = "llm"
        data["l3_summary"] = summary

    encoder = AxisEncoder()
    search = TopologicalGlobalSearch(engine, encoder, encoder_revision="enc-a")
    query = np.asarray([1, 0], dtype=np.float32)
    results = search.search_structured("alpha query", query_embedding=query, max_nodes_per_basin=1)
    assert results[0]["basin_id"] == alpha_id
    assert results[0]["l3"] == "alpha summary"
    assert results[0]["node_ids"] == [chunks[0]["id"]]
    assert len(encoder.calls) == 1 and isinstance(encoder.calls[0], list)

    search.search_structured("alpha query", query_embedding=query)
    assert len(encoder.calls) == 1  # same build, encoder revision, and content
    engine.build_id = "build-b"
    search.search_structured("alpha query", query_embedding=query)
    assert len(encoder.calls) == 2


def test_global_search_fallbacks_empty_and_deterministic_ties():
    empty = TopologicalGlobalSearch(BasinTopologyEngine(open_stores=False))
    assert empty.search_structured("overview") == []

    chunks = [
        _chunk("b.txt", 0, "second equal passage", (1, 0)),
        _chunk("a.txt", 0, "first equal passage", (1, 0)),
    ]
    engine = _engine(chunks)
    search = TopologicalGlobalSearch(engine)
    results = search.search_structured("detail", query_embedding=np.asarray([1, 0]))
    assert len(results) == 2
    assert results[0]["basin_id"] == min(engine.basins)
    assert search.search_structured("detail", query_embedding=np.asarray([0, 1])) == []


def test_router_uses_per_instance_centroids_and_honors_precomputed_query_vector():
    encoder = AxisEncoder()
    router = IntelligentQueryRouter(encoder, query_prompt="instruction: ")
    assert router._global_centroid is not None
    assert router._hybrid_centroid is not None
    assert router._local_centroid is not None
    query = "Please explain this unusual experimental result in detail"
    assert router.route_with_embeddings(query, np.asarray([1, 0, 0])) == "global"
    assert router.route_with_embeddings(query, np.asarray([0, 1, 0])) == "local"
    assert router.route_with_embeddings(query, np.asarray([0, 0, 1])) == "hybrid"
    assert len(encoder.calls) == 3  # centroid batches only; query embeddings were supplied
    assert router.route_with_embeddings("overview of the main theme") == "global"


def test_router_gracefully_falls_back_when_encoder_cannot_batch():
    class SingleOnlyEncoder:
        def encode(self, text, **_kwargs):
            if isinstance(text, list):
                raise ValueError("batch unavailable")
            return np.asarray([1.0, 0.0])

    router = IntelligentQueryRouter(SingleOnlyEncoder())
    assert router._global_centroid is None
    assert router.route_with_embeddings("tell me something specific here") == "hybrid"


def test_briefing_budget_and_satellite_helpers_cover_priority_and_scripts():
    assert estimate_tokens("") == 0
    assert estimate_tokens("漢字かな") == 4
    assert estimate_tokens("abc") == 2
    assert cap_satellite("  short  ") == "short"
    assert cap_satellite("x" * 500) == "x" * 400

    packet = BriefingPacket(
        hubs=["high priority evidence"],
        neighbors=["secondary evidence"],
        satellites=["summary one", "summary two"],
        node_ids=["n1"],
    )
    assert packet.texts_for_rerank() == ["high priority evidence", "secondary evidence"]
    assert packet.as_context(max_tokens=2, token_counter=lambda text: len(text.split())) == ""
    context = packet.as_context(max_tokens=50, token_counter=lambda text: len(text.split()))
    assert context.startswith("[Hubs]")
    assert "[Neighbors]" in context and "[Satellites]" in context


def test_hybrid_dense_fallback_gate_and_graph_expansion():
    chunks = [
        _chunk("doc.txt", 0, "alpha seed", (1, 0)),
        _chunk("doc.txt", 1, "alpha sequential child", (0.8, 0.6)),
        _chunk("other.txt", 0, "alpha virtual sibling", (1, 0)),
    ]
    engine = _engine(chunks)
    local = TopologicalLocalSearch(engine)
    hybrid = HybridSearch(engine, local)
    # Exercise dense-only operation independently from BM25 ranking.
    hybrid._bm25 = None
    local._faiss_hits = lambda _q, k: [
        (chunks[0]["id"], 0.9),
        (chunks[1]["id"], 0.8),
    ][:k]
    dense = hybrid.search_nodes("alpha", np.asarray([1, 0]), top_k=2)
    assert [row["id"] for row in dense] == [chunks[0]["id"], chunks[1]["id"]]
    assert dense[0]["confidence"] == pytest.approx(0.9)
    local._faiss_hits = lambda _q, _k: [(chunks[0]["id"], 0.01)]
    assert hybrid.search_nodes("alpha", np.asarray([1, 0]), top_k=2) == []

    hybrid._bm25 = type(
        "BM25Stub",
        (),
        {"n": 0, "score": lambda self, _query, top_k: []},
    )()
    assert hybrid.search_nodes("alpha", np.asarray([1, 0]), top_k=2, use_confidence_gate=False)

    assert hybrid._expand_graph_candidates([], max_extra=3) == []
    assert hybrid._expand_graph_candidates(["absent"], max_extra=3) == []
    assert hybrid._expand_graph_candidates([chunks[0]["id"]], max_extra=0) == []
    expanded = hybrid._expand_graph_candidates([chunks[0]["id"]], max_extra=3, sequential_radius=1)
    assert chunks[1]["id"] in expanded
