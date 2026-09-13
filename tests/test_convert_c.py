import json
import hashlib
import os

import numpy as np

from basinrag.core.ids import make_node_id
from basinrag.core.persistence import BasinPersistence
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


def _v3_index_metadata(source_file_hashes=None):
    return {
        "format_version": 3,
        "encoder_model": "unconfigured",
        "encoder_revision": "unresolved",
        "tokenizer": "unconfigured",
        "tokenizer_revision": "unresolved",
        "reranker_revision": "disabled",
        "chunking_mode": "characters",
        "chunk_policy_version": 1,
        "chunk_size": 512,
        "chunk_overlap": 0,
        "chunk_size_tokens": None,
        "chunk_overlap_tokens": None,
        "sources": {},
        "source_file_hashes": source_file_hashes or {},
    }


def test_rerank_keeps_distinct_ids_for_duplicate_text():
    from types import SimpleNamespace

    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs, **kwargs: [-1.0, -2.0])
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
    source_b = tmp_path / "b.txt"
    source_b.write_text("second source", encoding="utf-8")
    first = [_chunk("a.txt", 0, "primeiro documento atomos", 0)]
    second = [_chunk(str(source_b), 0, "segundo documento nucleos", 1)]
    engine = BasinTopologyEngine(storage_dir=str(storage))
    engine.build_graph(first)
    engine.partition_into_basins()
    store = BasinPersistence(str(storage))
    store.save_topology(engine)

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None
            self.last_ingest_status = {}

        def ingest(self, path):
            self.last_ingest_status[os.path.abspath(path)] = "ok"
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

    rag = BasinRAG(BasinRAGConfig(
        storage_dir=str(storage), encoder_model="unconfigured", chunk_overlap=0,
        use_rerank=False,
    ))
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
    assert store.save_topology(engine) is False
    assert not (tmp_path / "escape.json").exists()
    assert not os.path.exists(os.path.join(storage, "current.json"))


def test_public_bind_requires_key(monkeypatch):
    from basinrag.api.server import require_api_key_for_public_bind
    import pytest

    monkeypatch.delenv("BASINRAG_API_KEY", raising=False)
    monkeypatch.setenv("BASINRAG_ENV", "local_dev")
    monkeypatch.setenv("BASINRAG_ALLOW_KEYLESS_LOCAL", "true")
    with pytest.raises(RuntimeError):
        require_api_key_for_public_bind("0.0.0.0")
    require_api_key_for_public_bind("127.0.0.1")
    with pytest.raises(RuntimeError):
        require_api_key_for_public_bind("203.0.113.1")
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
    # Legacy chunk_size continues to mean characters.
    doc.write_text("palavra " * 200, encoding="utf-8")
    nodes = ingestor.ingest(str(doc))
    assert len(nodes) >= 3
    assert all(len(n["text"]) <= 50 for n in nodes)


def test_ingestor_token_chunks_respect_encoder_limit_and_batching(tmp_path, monkeypatch):
    import numpy as np

    class CharTokenizer:
        def __call__(self, text, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
            result = {"input_ids": list(range(len(text)))}
            if return_offsets_mapping:
                result["offset_mapping"] = [[i, i + 1] for i in range(len(text))]
            return result

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class DummyST:
        def __init__(self, *args, **kwargs):
            self.tokenizer = CharTokenizer()
            self.max_seq_length = 8
            self.batch_sizes = []

        def encode(self, texts, **kwargs):
            self.batch_sizes.append(len(texts))
            return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", DummyST)
    from basinrag.indexer.ingestor import BasinIngestor

    ingestor = BasinIngestor(
        "dummy",
        chunk_size=100,
        chunk_overlap=0,
        chunk_size_tokens=50,
        chunk_overlap_tokens=2,
        embedding_batch_size=2,
    )
    doc = tmp_path / "cjk.txt"
    source_text = "你好世界你好世界你好世界"
    doc.write_text(source_text, encoding="utf-8")

    nodes = ingestor.ingest(str(doc))

    assert len(nodes) == 3
    assert all(ingestor.splitter.token_count(node["text"]) <= 6 for node in nodes)
    _, offsets = ingestor.splitter._tokenizer_result(source_text, offsets=True)
    assert len(offsets) == len(source_text)
    assert ingestor.encoder.batch_sizes == [2, 1]
    rebuilt = nodes[0]["text"] + "".join(node["text"][2:] for node in nodes[1:])
    assert rebuilt == source_text

    legacy_ingestor = BasinIngestor("dummy", chunk_size=100, chunk_overlap=0)
    legacy_nodes = legacy_ingestor.ingest(str(doc))
    assert len(legacy_nodes) == 2
    assert all(legacy_ingestor.splitter.token_count(node["text"]) <= 6 for node in legacy_nodes)


def test_ingestor_json_streams_records_and_batches_embeddings(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    class CharTokenizer:
        def __call__(self, text, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
            result = {"input_ids": list(range(len(text)))}
            if return_offsets_mapping:
                result["offset_mapping"] = [[i, i + 1] for i in range(len(text))]
            return result

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class DummyST:
        def __init__(self, *args, **kwargs):
            self.tokenizer = CharTokenizer()
            self.max_seq_length = 10
            self.batch_sizes = []

        def encode(self, texts, **kwargs):
            self.batch_sizes.append(len(texts))
            return np.ones((len(texts), 8), dtype=np.float32)

    calls = []
    records = [
        {"documents": ["abcdefghijk"], "question": "q1", "answer": "a1"},
        {"documents": ["uvwxyz"], "question": "q2", "answer": "a2"},
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append(kwargs)
        return iter(records)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", DummyST)
    from basinrag.indexer.ingestor import BasinIngestor

    ingestor = BasinIngestor(
        "dummy", chunk_size=100, chunk_overlap=0,
        chunk_size_tokens=6, chunk_overlap_tokens=0, embedding_batch_size=2,
    )
    json_path = tmp_path / "records.jsonl"
    json_path.write_text("{}\n{}\n", encoding="utf-8")
    stream = ingestor.iter_ingest_json_dataset(str(json_path))
    nodes = [next(stream)]
    assert ingestor.encoder.batch_sizes == [2]
    nodes.extend(stream)

    assert calls[0]["streaming"] is True
    assert [size for size in ingestor.encoder.batch_sizes if size > 0] == [2, 1]
    assert all(ingestor.splitter.token_count(node["text"]) <= 6 for node in nodes)
    assert len({node["source"] for node in nodes}) == 2
    assert isinstance(ingestor.ingest_json_dataset(str(json_path)), list)


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


def test_virtual_edge_degree_cap_applies_to_both_endpoints(tmp_path):
    engine = BasinTopologyEngine(storage_dir=str(tmp_path / "semantic-degree"))
    try:
        for i in range(10):
            engine.graph.add_node(
                f"node-{i}",
                embedding=np.ones(8, dtype=np.float32),
                basin_id=f"basin-{i}",
            )

        engine._link_semantic_neighbors()

        virtual_degrees = {
            node_id: sum(
                1 for _, _, data in engine.graph.edges(node_id, data=True)
                if data.get("type") == "virtual-edge"
            )
            for node_id in engine.graph.nodes
        }
        assert max(virtual_degrees.values()) <= engine.MAX_SEMANTIC_DEGREE
        assert min(virtual_degrees.values()) > 0
    finally:
        engine.close_stores()


def test_sync_replaces_changed_source_and_removes_deleted_source(tmp_path, monkeypatch):
    root = tmp_path / "sync-root"
    root.mkdir()
    source_a = os.path.abspath(root / "a.md")
    source_b = os.path.abspath(root / "b.md")
    (root / "a.md").write_text("old a", encoding="utf-8")
    (root / "b.md").write_text("old b", encoding="utf-8")
    old_a = _chunk(source_a, 0, "old a", 0)
    old_b = _chunk(source_b, 0, "old b", 1)
    storage = str(tmp_path / "sync-index")
    engine = BasinTopologyEngine(storage_dir=storage)
    engine.build_graph([old_a, old_b])
    engine.partition_into_basins()
    engine.index_metadata = _v3_index_metadata({
        os.path.normcase(source_a): hashlib.sha256((root / "a.md").read_bytes()).hexdigest(),
        os.path.normcase(source_b): hashlib.sha256((root / "b.md").read_bytes()).hexdigest(),
    })
    persistence = BasinPersistence(storage)
    assert persistence.save_topology(engine)

    (root / "a.md").write_text("changed a", encoding="utf-8")
    changed_a = _chunk(source_a, 0, "changed a", 2)

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None
            self.nodes = [changed_a]

        def ingest(self, path):
            return self.nodes if os.path.normcase(os.path.abspath(path)) == os.path.normcase(source_a) else []

        def ingest_directory(self, path):
            return self.nodes

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("basinrag.factory.BasinIngestor", FakeIngestor)
    monkeypatch.setattr("basinrag.factory.UniversalLLM", FakeLLM)
    monkeypatch.setattr("basinrag.indexer.summarizer.UniversalLLM", FakeLLM)
    from basinrag.factory import BasinRAG, BasinRAGConfig

    rag = BasinRAG(BasinRAGConfig(
        storage_dir=storage, encoder_model="unconfigured", chunk_overlap=0,
        use_rerank=False,
    ))
    rag.engine = engine
    rag.persistence = persistence

    assert rag.sync(str(root)) == 1
    assert old_a["id"] not in rag.engine.graph
    assert changed_a["id"] in rag.engine.graph
    assert old_b["id"] in rag.engine.graph

    (root / "b.md").unlink()
    assert rag.sync(str(root)) == 0
    assert old_b["id"] not in rag.engine.graph
    engine.close_stores()
    rag.engine.close_stores()


def test_sync_does_not_record_hash_for_source_that_failed_ingestion(tmp_path, monkeypatch):
    root = tmp_path / "sync-failed-source"
    root.mkdir()
    source_ok = os.path.abspath(root / "ok.md")
    source_bad = os.path.abspath(root / "broken.pdf")
    (root / "ok.md").write_text("new ok", encoding="utf-8")
    (root / "broken.pdf").write_bytes(b"new broken")
    old_ok = _chunk(source_ok, 0, "old ok", 0)
    old_bad = _chunk(source_bad, 0, "old broken", 1)
    new_ok = _chunk(source_ok, 0, "new ok", 2)

    storage = str(tmp_path / "sync-failed-index")
    engine = BasinTopologyEngine(storage_dir=storage)
    engine.build_graph([old_ok, old_bad])
    engine.partition_into_basins()
    persistence = BasinPersistence(storage)
    old_bad_hash = "hash-before-failed-parse"
    normalized_bad = os.path.normcase(os.path.abspath(source_bad))
    assert persistence.save_topology(engine)
    engine.index_metadata = _v3_index_metadata({
        os.path.normcase(os.path.abspath(source_ok)): "old-ok-hash",
        normalized_bad: old_bad_hash,
    })

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None
            self.last_ingest_status = {}
            self.calls = []

        def ingest_files(self, paths):
            normalized = {os.path.normcase(os.path.abspath(item)) for item in paths}
            self.calls.append(normalized)
            if normalized_bad in normalized:
                self.last_ingest_status = {source_bad: "error"}
            else:
                self.last_ingest_status = {}
            if os.path.normcase(os.path.abspath(source_ok)) in normalized:
                self.last_ingest_status[source_ok] = "ok"
                return [new_ok]
            return []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("basinrag.factory.BasinIngestor", FakeIngestor)
    monkeypatch.setattr("basinrag.factory.UniversalLLM", FakeLLM)
    monkeypatch.setattr("basinrag.indexer.summarizer.UniversalLLM", FakeLLM)
    from basinrag.factory import BasinRAG, BasinRAGConfig

    rag = BasinRAG(BasinRAGConfig(
        storage_dir=storage, encoder_model="unconfigured", chunk_overlap=0,
        use_rerank=False,
    ))
    rag.engine = engine
    rag.persistence = persistence

    assert rag.sync(str(root)) == 1
    hashes = rag.engine.index_metadata["source_file_hashes"]
    assert hashes[normalized_bad] == old_bad_hash
    assert old_bad["id"] in rag.engine.graph
    assert new_ok["id"] in rag.engine.graph

    assert rag.sync(str(root)) == 0
    assert normalized_bad in rag.ingestor.calls[-1]

    engine.close_stores()
    rag.engine.close_stores()


def test_reindex_aborts_on_source_error_and_preserves_active_snapshot(tmp_path, monkeypatch):
    import pytest

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    good_path = os.path.abspath(corpus / "good.md")
    bad_path = os.path.abspath(corpus / "broken.pdf")
    (corpus / "good.md").write_text("good source", encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"broken pdf")

    storage = str(tmp_path / "reindex-index")
    old_node = _chunk("previous-corpus.md", 0, "previous snapshot remains", 0)
    engine = BasinTopologyEngine(storage_dir=storage)
    engine.build_graph([old_node])
    engine.partition_into_basins()
    persistence = BasinPersistence(storage)
    assert persistence.save_topology(engine)
    active_build_id = persistence.current_build_id()

    new_node = _chunk(good_path, 0, "readable replacement source", 1)

    class FakeIngestor:
        def __init__(self, *args, **kwargs):
            self.encoder = None
            self.last_ingest_status = {}

        def ingest_directory(self, path):
            self.last_ingest_status = {
                good_path: "ok",
                bad_path: "error",
            }
            return [new_node]

    monkeypatch.setattr("basinrag.factory.BasinIngestor", FakeIngestor)
    from basinrag.factory import BasinRAG, BasinRAGConfig

    destination = str(tmp_path / "new-v3-destination")
    rag = BasinRAG(BasinRAGConfig(
        storage_dir=destination, encoder_model="unconfigured", chunk_overlap=0,
        use_rerank=False,
    ))

    try:
        with pytest.raises(RuntimeError, match="(?i)(falha|erro)"):
            rag.reindex(str(corpus))

        assert persistence.current_build_id() == active_build_id
        assert set(engine.graph.nodes) == {old_node["id"]}
        assert not os.path.exists(os.path.join(destination, "current.json"))

        reloaded = BasinTopologyEngine(storage_dir=str(tmp_path / "reloaded"))
        try:
            assert persistence.load_topology(reloaded)
            assert set(reloaded.graph.nodes) == {old_node["id"]}
        finally:
            reloaded.close_stores()
    finally:
        if rag.engine is not engine:
            rag.engine.close_stores()
        engine.close_stores()
