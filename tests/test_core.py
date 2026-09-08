import numpy as np
import os
import pytest

from basinrag.core.ids import make_node_id, tokenize
from basinrag.core.functional_graph import (
    sequential_successor,
    detect_attractors,
    reverse_hops,
    rho_parent_child,
)
from basinrag.core.topology import BasinTopologyEngine
from basinrag.core.persistence import BasinPersistence
from basinrag.indexer.bm25 import BM25Index
from basinrag.indexer.condensation import node_layers
from basinrag.retriever.briefing import BriefingPacket
from basinrag.retriever.fusion import weighted_rrf, apply_hop_prior, ranked_ids
from basinrag.retriever.router import IntelligentQueryRouter
from basinrag.retriever.local_search import TopologicalLocalSearch


def _chunk(source, index, text, dim_slot, dim=8):
    layers = node_layers(text)
    e = np.zeros(dim, dtype=np.float32)
    e[dim_slot % dim] = 1.0
    return {
        "id": make_node_id(source, index, text),
        "text": text,
        "embedding": e,
        "source": source,
        "chunk_index": index,
        "l1": layers["l1"],
        "l2": layers["l2"],
        "metadata": {},
    }


def test_hybrid_search_imports():
    from basinrag.retriever.hybrid_search import HybridSearch
    assert HybridSearch is not None


def test_hop_prior_does_not_erase_deep_hits():
    from basinrag.retriever.fusion import apply_hop_prior
    scores = {"shallow": 1.0, "deep": 1.0}
    hops = {"shallow": 0, "deep": 19}
    out = apply_hop_prior(scores, hops)
    assert out["deep"] >= 0.7
    assert out["shallow"] > out["deep"]


def test_query_routing():
    assert IntelligentQueryRouter.route("qual é o tema principal deste texto?") == "global"
    assert IntelligentQueryRouter.route("summarize the main theme of this book") == "global"
    assert IntelligentQueryRouter.route("visão geral") == "global"
    assert IntelligentQueryRouter.route("qual é a idade do joão?") == "hybrid"
    assert IntelligentQueryRouter.route("Hawking") == "hybrid"
    assert IntelligentQueryRouter.route("quais são os experimentos de Millikan?") == "hybrid"


def test_missing_ingest_dir_does_not_create_folder(tmp_path):
    from basinrag.indexer.ingestor import BasinIngestor

    missing = tmp_path / "no-such-docs"
    dummy = type("Ingestor", (), {})()
    out = BasinIngestor.ingest_directory(dummy, str(missing))
    assert out == []
    assert not missing.exists()


def test_empty_ingest_does_not_wipe_index(tmp_path, monkeypatch):
    chunks = [_chunk("p.txt", 0, "nucleo atomico", 0)]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    store = BasinPersistence(str(tmp_path / "idx"))
    assert store.save_topology(engine)

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None

        def ingest_directory(self, path):
            return []

        def ingest(self, path):
            return []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("basinrag.factory.BasinIngestor", FakeIngestor)
    monkeypatch.setattr("basinrag.factory.UniversalLLM", FakeLLM)
    monkeypatch.setattr("basinrag.indexer.summarizer.UniversalLLM", FakeLLM)

    from basinrag.factory import BasinRAG, BasinRAGConfig

    rag = BasinRAG(BasinRAGConfig(storage_dir=str(tmp_path / "idx")))
    rag.engine = engine
    rag.persistence = store
    n = rag.ingest(str(tmp_path / "missing-docs"))
    assert n == 0
    assert engine.graph.number_of_nodes() == 1
    other = BasinTopologyEngine()
    assert store.load_topology(other)
    assert other.graph.number_of_nodes() == 1


def test_extractive_l3_does_not_look_like_llm():
    chunks = [
        _chunk("s.txt", 0, "primeiro paragrafo sobre atomos", 0),
        _chunk("s.txt", 1, "segundo paragrafo sobre atomos", 1),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    basin = next(iter(engine.basins.values()))
    data = basin.rho_tree.nodes[basin.attractor_id]
    assert data.get("l3_source") == "extractive"
    assert data.get("l3_label")
    assert not data.get("l3_summary")


def test_stable_ids_are_deterministic():
    a = make_node_id("doc.txt", 0, "hello")
    b = make_node_id("doc.txt", 0, "hello")
    c = make_node_id("doc.txt", 1, "hello")
    assert a == b
    assert a != c
    assert tokenize("Hello, mundo!") == ["hello", "mundo"]


def test_functional_graph_drains_to_section_start():
    by_source = {"a.txt": ["a0", "a1", "a2"]}
    succ = sequential_successor(by_source, section_size=20)
    assert succ["a0"] is None
    assert succ["a1"] == "a0"
    assert succ["a2"] == "a1"
    attr = detect_attractors(succ)
    assert attr["a0"] == attr["a1"] == attr["a2"] == "a0"
    hops = reverse_hops(succ, attr)
    assert hops["a0"] == 0
    assert hops["a2"] == 2
    edges = rho_parent_child(succ, ["a0", "a1", "a2"])
    assert ("a0", "a1") in edges
    assert ("a1", "a2") in edges


def test_build_graph_clears_and_does_not_duplicate():
    chunks = [
        _chunk("s.txt", 0, "primeiro paragrafo sobre atomos", 0),
        _chunk("s.txt", 1, "segundo paragrafo sobre atomos", 1),
        _chunk("t.txt", 0, "outro documento sobre estrelas", 3),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    n1 = engine.graph.number_of_nodes()
    assert n1 == 3
    assert len(engine.basins) == 2
    attr = chunks[0]["id"]
    assert engine.basins[attr].rho_tree.has_edge(chunks[0]["id"], chunks[1]["id"])

    engine.build_graph(chunks)
    engine.partition_into_basins()
    assert engine.graph.number_of_nodes() == n1
    assert set(engine.graph.nodes) == {c["id"] for c in chunks}


def test_bm25_and_weighted_rrf():
    index = BM25Index()
    index.build(["n1", "n2", "n3"], ["black hole hawking", "quantum superposition", "black hole entropy"])
    hits = index.score("black hole hawking", top_k=2)
    assert hits[0][0] in {"n1", "n3"}
    scores = weighted_rrf(["n1", "n2"], ["n2", "n3"], alpha=0.55, k=60)
    assert scores["n2"] > scores["n3"]
    equal = apply_hop_prior({"a": 1.0, "b": 1.0}, {"a": 0, "b": 4})
    assert equal["a"] > equal["b"]
    assert ranked_ids(scores, 2)[0] in scores


def test_persistence_roundtrip(tmp_path):
    chunks = [
        _chunk("p.txt", 0, "introducao ao tema", 0),
        _chunk("p.txt", 1, "desenvolvimento do tema", 1),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    index = BM25Index()
    ids = list(engine.graph.nodes)
    index.build(ids, [engine.graph.nodes[n]["text"] for n in ids])
    engine.bm25 = index

    store = BasinPersistence(str(tmp_path / "idx"))
    live_id = chunks[0]["id"]
    assert "embedding" in engine.graph.nodes[live_id]
    assert store.save_topology(engine)
    assert "embedding" in engine.graph.nodes[live_id]
    rho = next(iter(engine.basins.values())).rho_tree
    assert "embedding" in rho.nodes[live_id]
    other = BasinTopologyEngine()
    assert store.load_topology(other)
    assert other.graph.number_of_nodes() == 2
    assert len(other.basins) == 1
    nid = chunks[0]["id"]
    assert "embedding" in other.graph.nodes[nid]
    assert other.successor[chunks[1]["id"]] == nid
    assert other.bm25 is not None
    assert other.bm25.score("tema", top_k=1)[0][0] in ids
    tree = next(iter(other.basins.values())).rho_tree
    assert tree.number_of_edges() >= 1


def test_load_rejects_graph_without_embeddings(tmp_path):
    chunks = [_chunk("p.txt", 0, "so um chunk", 0)]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    store = BasinPersistence(str(tmp_path / "idx"))
    assert store.save_topology(engine)
    os.remove(os.path.join(store.storage_dir, "embeddings.npz"))
    other = BasinTopologyEngine()
    assert store.load_topology(other) is False


def test_local_search_stays_in_seed_basin():
    chunks = [
        _chunk("a.txt", 0, "nucleos atomicos e quanta", 0),
        _chunk("a.txt", 1, "continuidade sobre nucleos atomicos", 1),
        _chunk("b.txt", 0, "estrelas e galaxias distantes", 7),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    local = TopologicalLocalSearch(engine)
    hits = local.search_nodes(chunks[0]["embedding"], top_k=10, max_tokens=5000)
    ids = {h["id"] for h in hits}
    assert chunks[0]["id"] in ids
    assert chunks[2]["id"] not in ids


def test_briefing_excludes_satellites_from_rerank():
    packet = BriefingPacket(
        hubs=["hub text"],
        neighbors=["neighbor text"],
        satellites=["L3 summary must not be reranked"],
        node_ids=["a"],
    )
    assert "L3 summary must not be reranked" not in packet.texts_for_rerank()
    ctx = packet.as_context(max_tokens=500)
    assert "Hubs" in ctx
    assert "Satellites" in ctx
    long_sat = "x" * 800
    packet.satellites = [long_sat]
    ctx = packet.as_context(max_tokens=500)
    assert "x" * 400 in ctx
    assert "x" * 500 not in ctx


def test_local_search_includes_next_section_sequential_neighbor():
    chunks = [
        _chunk("s.txt", i, f"paragrafo numero {i} sobre o tema continuo", 0 if i < 20 else 1, dim=32)
        for i in range(25)
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    assert len(engine.basins) == 2
    local = TopologicalLocalSearch(engine)
    hits = local.search_nodes(chunks[0]["embedding"], top_k=30, max_tokens=50000)
    ids = {h["id"] for h in hits}
    assert chunks[0]["id"] in ids
    assert chunks[20]["id"] in ids


def test_local_search_includes_virtual_edge_not_orthogonal():
    chunks = [
        _chunk("a.txt", 0, "nucleos atomicos e quanta", 0),
        _chunk("b.txt", 0, "mesmo tema nucleos atomicos", 0),
        _chunk("c.txt", 0, "estrelas e galaxias distantes", 7),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    assert engine.graph.has_edge(chunks[0]["id"], chunks[1]["id"])
    local = TopologicalLocalSearch(engine)
    hits = local.search_nodes(chunks[0]["embedding"], top_k=10, max_tokens=5000)
    ids = {h["id"] for h in hits}
    assert chunks[0]["id"] in ids
    assert chunks[1]["id"] in ids
    assert chunks[2]["id"] not in ids


def test_local_search_rejects_low_similarity():
    chunks = [
        _chunk("a.txt", 0, "nucleos atomicos", 0),
        _chunk("a.txt", 1, "continuidade nucleos", 1),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    local = TopologicalLocalSearch(engine)
    query = np.zeros(8, dtype=np.float32)
    query[7] = 1.0
    assert local.search_nodes(query, top_k=10, max_tokens=5000) == []


def test_projector_ignores_short_stopwords():
    from basinrag.retriever.projector import seed_node_ids

    chunks = [_chunk("p.txt", 0, "nucleo atomico para estudo", 0)]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    assert seed_node_ids(engine, "para como") == []
    assert chunks[0]["id"] in seed_node_ids(engine, "atomico")


def test_save_leaves_no_tmp_and_rebuilds_missing_basins(tmp_path):
    import json
    import shutil

    chunks = [
        _chunk("p.txt", 0, "introducao ao tema", 0),
        _chunk("p.txt", 1, "desenvolvimento do tema", 1),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    index = BM25Index()
    ids = list(engine.graph.nodes)
    index.build(ids, [engine.graph.nodes[n]["text"] for n in ids])
    engine.bm25 = index
    store = BasinPersistence(str(tmp_path / "idx"))
    assert store.save_topology(engine)
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == []

    shutil.rmtree(store.basins_dir)
    other = BasinTopologyEngine()
    assert store.load_topology(other)
    assert len(other.basins) == 1
    tree = next(iter(other.basins.values())).rho_tree
    assert tree.number_of_nodes() == 2


def test_stale_bm25_is_discarded_on_load(tmp_path):
    import json

    chunks = [_chunk("p.txt", 0, "introducao ao tema", 0)]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    index = BM25Index()
    index.build(list(engine.graph.nodes), ["introducao ao tema"])
    engine.bm25 = index
    store = BasinPersistence(str(tmp_path / "idx"))
    assert store.save_topology(engine)
    path = os.path.join(store.storage_dir, "bm25.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["buildId"] = "stale"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    other = BasinTopologyEngine()
    assert store.load_topology(other)
    assert other.bm25 is None


def test_json_documents_get_distinct_sources(tmp_path):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from basinrag.indexer.ingestor import group_json_documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    records = [
        {"documents": ["alpha doc one"], "question": "q1", "answer": "a1"},
        {"documents": ["beta doc two"], "question": "q2", "answer": "a2"},
    ]
    groups = group_json_documents(str(tmp_path / "d.json"), records, splitter)
    assert len(groups) == 2
    assert groups[0][0] != groups[1][0]
    assert groups[0][0].endswith("#0")
    assert groups[1][0].endswith("#1")


def test_hnsw_path_when_corpus_exceeds_threshold(monkeypatch):
    monkeypatch.setattr("basinrag.core.vector_index.HNSW_THRESHOLD", 2)
    chunks = [_chunk("h.txt", i, f"texto unico {i} xyzabc", i) for i in range(3)]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    local = TopologicalLocalSearch(engine)
    import faiss
    assert isinstance(local._index, faiss.IndexHNSWFlat)
    hits = local.search_nodes(chunks[0]["embedding"], top_k=3, max_tokens=5000)
    assert hits
    assert hits[0]["id"] == chunks[0]["id"]


def test_global_search_empty_when_all_scores_zero():
    from basinrag.retriever.global_search import TopologicalGlobalSearch

    chunks = [
        _chunk("p.txt", 0, "introducao ao tema", 0),
        _chunk("p.txt", 1, "desenvolvimento do tema", 1),
    ]
    engine = BasinTopologyEngine()
    engine.build_graph(chunks)
    engine.partition_into_basins()
    gs = TopologicalGlobalSearch(engine, encoder=None)
    assert gs.search_structured("visao geral") == []
