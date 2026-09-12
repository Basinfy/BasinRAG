import json
import os

import numpy as np

from basinrag.core.ids import make_node_id
from basinrag.core.persistence import BasinPersistence, basin_filename
from basinrag.core.topology import BasinTopologyEngine, TopologicalBasin
from basinrag.indexer.condensation import node_layers
from basinrag.retriever.reranker import CrossEncoderReranker


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


def test_rerank_keeps_distinct_ids_for_duplicate_text():
    reranker = CrossEncoderReranker()
    reranker._model = "disabled"
    items = [("node-a", "same text"), ("node-b", "same text")]
    out = reranker.rerank_items("query", items, top_k=2)
    assert [nid for nid, _ in out] == ["node-a", "node-b"]


def test_save_with_open_stores_preserves_index(tmp_path):
    storage = str(tmp_path / "idx")
    engine = BasinTopologyEngine(storage_dir=storage)
    chunks = [
        _chunk("p.txt", 0, "introducao ao tema", 0),
        _chunk("p.txt", 1, "desenvolvimento do tema", 1),
    ]
    engine.build_graph(chunks)
    engine.partition_into_basins()
    store = BasinPersistence(storage)
    assert store.save_topology(engine)
    assert engine.graph.number_of_nodes() == 2
    assert os.path.exists(os.path.join(storage, "current.json"))
    other = BasinTopologyEngine(storage_dir=str(tmp_path / "load"))
    assert store.load_topology(other)
    assert other.graph.number_of_nodes() == 2
    assert set(other.graph.nodes) == set(engine.graph.nodes)


def test_ingest_second_file_preserves_first(tmp_path, monkeypatch):
    storage = tmp_path / "idx"
    first = [_chunk("a.txt", 0, "primeiro documento atomos", 0)]
    second = [_chunk("b.txt", 0, "segundo documento nucleos", 1)]
    engine = BasinTopologyEngine(storage_dir=str(storage))
    engine.build_graph(first)
    engine.partition_into_basins()
    store = BasinPersistence(str(storage))
    store.save_topology(engine)

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None

        def ingest(self, path):
            return second

        def ingest_directory(self, path):
            return second

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("basinrag.factory.BasinIngestor", FakeIngestor)
    monkeypatch.setattr("basinrag.factory.UniversalLLM", FakeLLM)
    monkeypatch.setattr("basinrag.indexer.summarizer.UniversalLLM", FakeLLM)

    from basinrag.factory import BasinRAG, BasinRAGConfig

    rag = BasinRAG(BasinRAGConfig(storage_dir=str(storage)))
    rag.engine = engine
    rag.persistence = store
    n = rag.ingest(str(tmp_path / "b.txt"))
    assert n == 1
    assert first[0]["id"] in rag.engine.graph
    assert second[0]["id"] in rag.engine.graph
    assert rag.engine.graph.number_of_nodes() == 2


def test_basin_id_cannot_escape_storage(tmp_path):
    storage = tmp_path / "idx"
    engine = BasinTopologyEngine(storage_dir=str(storage))
    engine.build_graph([_chunk("p.txt", 0, "texto seguro", 0)])
    engine.partition_into_basins()
    evil_id = "../escape"
    engine.basins[evil_id] = TopologicalBasin(evil_id)
    store = BasinPersistence(str(storage))
    assert store.save_topology(engine)
    assert not (tmp_path / "escape.json").exists()
    basins_dir = os.path.join(store.active_dir(), "basins")
    names = os.listdir(basins_dir)
    assert basin_filename(evil_id) in names
    assert all(".." not in name for name in names)
    other = BasinTopologyEngine(storage_dir=str(tmp_path / "load"))
    assert store.load_topology(other)
    assert evil_id in other.basins


def test_public_bind_requires_key(monkeypatch):
    from basinrag.api.server import require_api_key_for_public_bind
    import pytest

    monkeypatch.delenv("BASINRAG_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_api_key_for_public_bind("0.0.0.0")
    require_api_key_for_public_bind("127.0.0.1")
    monkeypatch.setenv("BASINRAG_API_KEY", "secret")
    require_api_key_for_public_bind("0.0.0.0")


def test_ingestor_honors_chunk_size(tmp_path, monkeypatch):
    import numpy as np

    class DummyST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 8), dtype=np.float32)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", DummyST)
    from basinrag.indexer.ingestor import BasinIngestor

    ingestor = BasinIngestor("dummy", chunk_size=50, chunk_overlap=0)
    assert ingestor.splitter._chunk_size == 50
    doc = tmp_path / "doc.txt"
    doc.write_text("palavra " * 80, encoding="utf-8")
    nodes = ingestor.ingest(str(doc))
    assert len(nodes) >= 3
    assert all(len(n["text"]) <= 60 for n in nodes)


def test_current_pointer_is_json(tmp_path):
    storage = str(tmp_path / "idx")
    engine = BasinTopologyEngine(storage_dir=storage)
    engine.build_graph([_chunk("p.txt", 0, "pointer check", 0)])
    engine.partition_into_basins()
    store = BasinPersistence(storage)
    assert store.save_topology(engine)
    with open(os.path.join(storage, "current.json"), encoding="utf-8") as f:
        pointer = json.load(f)
    assert pointer["build_id"]
    assert os.path.isdir(os.path.join(storage, "builds", pointer["build_id"]))
