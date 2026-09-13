from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from basinrag.factory import BasinRAG, BasinRAGConfig
from basinrag.retriever.base import BasinRAGRetriever
from basinrag.retriever.global_search import TopologicalGlobalSearch
from basinrag.retriever.hybrid_search import HybridSearch
from basinrag.retriever.router import IntelligentQueryRouter


def _key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(path))


class _Persistence:
    def __init__(self, engine):
        self.engine = engine

    def current_build_id(self):
        return self.engine.build_id

    def write_transaction(self, expected_build_id=None):
        return nullcontext()


class _Ingestor:
    def __init__(self, *, errors=(), empty=(), after_read=None):
        self.errors = {_key(path) for path in errors}
        self.empty = {_key(path) for path in empty}
        self.after_read = after_read
        self.last_ingest_status = {}
        self.read_paths = []

    def iter_ingest_files(self, paths):
        for path in paths:
            absolute = os.path.abspath(path)
            source = _key(absolute)
            self.read_paths.append(source)
            if source in self.errors:
                self.last_ingest_status[source] = "error"
                continue
            text = Path(absolute).read_text(encoding="utf-8")
            self.last_ingest_status[source] = "ok"
            if self.after_read:
                self.after_read(absolute)
            if source in self.empty:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            yield {
                "id": f"{Path(absolute).name}:{digest}",
                "text": text,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "source": source,
                "chunk_index": 0,
                "metadata": {},
            }


def _rag(tmp_path, monkeypatch, *, ingestor=None, old_sources=(), hashes=None):
    rag = BasinRAG.__new__(BasinRAG)
    rag.config = BasinRAGConfig(
        storage_dir=str(tmp_path / "index"),
        encoder_model="unconfigured",
        use_rerank=False,
        chunk_overlap=0,
    )
    rag.engine = SimpleNamespace(
        build_id="",
        graph=nx.Graph(),
        index_metadata={"source_file_hashes": dict(hashes or {})},
    )
    for index, source in enumerate(old_sources):
        rag.engine.graph.add_node(f"old-{index}", source=_key(source), text="old passage")
    rag.persistence = _Persistence(rag.engine)
    rag._ingestor = ingestor or _Ingestor()
    rag.summarizer = SimpleNamespace(engine=rag.engine)
    rag.retriever = None
    rag._loaded = True
    rag._inference_semaphore = None
    rag._snapshot_lock = RLock()
    publications = []

    def capture(chunks, base_engine, **kwargs):
        publications.append((list(chunks), base_engine, kwargs))
        return len(publications[-1][0])

    monkeypatch.setattr(rag, "_publish_snapshot", capture)
    return rag, publications


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _key(path)


def test_directory_ingest_is_additive_and_only_reads_new_sources(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    existing = root / "existing.txt"
    new = root / "new.txt"
    old_digest = _write(existing, "unchanged source")
    _write(new, "new source")
    old_hash = hashlib.sha256(existing.read_bytes()).hexdigest()
    ingestor = _Ingestor()
    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=ingestor,
        old_sources=[existing],
        hashes={old_digest: old_hash},
    )

    count = rag.ingest(str(root))

    assert count == 1
    assert ingestor.read_paths == [_key(new)]
    assert [item["source"] for item in publications[0][0]] == [_key(new)]
    assert publications[0][2]["source_file_hashes"] == {
        _key(existing): old_hash,
        _key(new): hashlib.sha256(new.read_bytes()).hexdigest(),
    }


def test_directory_ingest_noop_when_all_sources_are_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "same.md"
    source_key = _write(source, "same content")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ingestor = _Ingestor()
    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=ingestor,
        old_sources=[source],
        hashes={source_key: digest},
    )

    assert rag.ingest(str(source.parent)) == 0
    assert ingestor.read_paths == []
    assert publications == []


def test_directory_ingest_rejects_changed_existing_source_before_partial_addition(
    tmp_path, monkeypatch
):
    root = tmp_path / "sources"
    existing = root / "existing.txt"
    _write(existing, "changed now")
    _write(root / "new.txt", "would be additive")
    old_digest = hashlib.sha256(b"previous content").hexdigest()
    ingestor = _Ingestor()
    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=ingestor,
        old_sources=[existing],
        hashes={_key(existing): old_digest},
    )

    with pytest.raises(ValueError, match=r"use sync\(\)"):
        rag.ingest(str(root))

    assert ingestor.read_paths == []
    assert publications == []


def test_directory_ingest_parse_failure_does_not_publish_new_sources(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    failed = root / "bad.txt"
    added = root / "good.txt"
    _write(failed, "bad document")
    _write(added, "good document")
    ingestor = _Ingestor(errors=[failed])
    rag, publications = _rag(tmp_path, monkeypatch, ingestor=ingestor)

    with pytest.raises(RuntimeError, match="nenhuma alteração foi publicada"):
        rag.ingest(str(root))

    assert set(ingestor.read_paths) == {_key(failed), _key(added)}
    assert publications == []


def test_ingest_aborts_if_source_changes_during_read(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "race.txt"
    _write(source, "version one")

    def mutate(path):
        Path(path).write_text("version two", encoding="utf-8")

    rag, publications = _rag(
        tmp_path, monkeypatch, ingestor=_Ingestor(after_read=mutate)
    )

    with pytest.raises(RuntimeError, match="mudaram durante ingest"):
        rag.ingest(str(source.parent))

    assert publications == []
    assert source.read_text(encoding="utf-8") == "version two"


def test_sync_preserves_failed_source_hash_while_replacing_successful_peer(
    tmp_path, monkeypatch
):
    root = tmp_path / "sources"
    failed = root / "failed.txt"
    successful = root / "successful.txt"
    _write(failed, "failed changed")
    _write(successful, "successful changed")
    old_failed_hash = hashlib.sha256(b"failed old content").hexdigest()
    old_success_hash = hashlib.sha256(b"successful old content").hexdigest()
    ingestor = _Ingestor(errors=[failed])
    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=ingestor,
        old_sources=[failed, successful],
        hashes={_key(failed): old_failed_hash, _key(successful): old_success_hash},
    )

    count = rag.sync(str(root))

    assert count == 1
    chunks, _engine, kwargs = publications[0]
    assert [chunk["source"] for chunk in chunks] == [_key(successful)]
    assert kwargs["replace_sources"] == {_key(successful)}
    assert kwargs["source_file_hashes"][_key(failed)] == old_failed_hash
    assert kwargs["source_file_hashes"][_key(successful)] == hashlib.sha256(
        successful.read_bytes()
    ).hexdigest()


def test_sync_empty_success_replaces_old_chunks_and_records_new_hash(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "emptied.txt"
    _write(source, "now intentionally empty")
    old_digest = hashlib.sha256(b"previous nonempty content").hexdigest()
    ingestor = _Ingestor(empty=[source])
    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=ingestor,
        old_sources=[source],
        hashes={_key(source): old_digest},
    )

    assert rag.sync(str(source)) == 0

    assert len(publications) == 1
    chunks, _engine, kwargs = publications[0]
    assert chunks == []
    assert kwargs["replace_sources"] == {_key(source)}
    assert kwargs["source_file_hashes"][_key(source)] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_sync_aborts_publication_when_a_source_changes_during_read(tmp_path, monkeypatch):
    source = tmp_path / "sources" / "racing.txt"
    _write(source, "before")
    old_digest = hashlib.sha256(b"older indexed value").hexdigest()

    def mutate(path):
        Path(path).write_text("after", encoding="utf-8")

    rag, publications = _rag(
        tmp_path,
        monkeypatch,
        ingestor=_Ingestor(after_read=mutate),
        old_sources=[source],
        hashes={_key(source): old_digest},
    )

    with pytest.raises(RuntimeError, match="mudaram durante sync"):
        rag.sync(str(source.parent))

    assert publications == []


def test_retriever_encodes_a_prompted_query_once_and_reuses_normalized_vector(
    monkeypatch,
):
    import basinrag.retriever.base as base_module

    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return np.array([3.0, 4.0], dtype=np.float32)

    class Router:
        def __init__(self, encoder, *, query_prompt):
            self.encoder = encoder
            self.query_prompt = query_prompt

    class Engine:
        graph = nx.Graph()
        basins = {}
        index_metadata = {}
        bm25 = None

    encoder = Encoder()
    monkeypatch.setattr(base_module, "IntelligentQueryRouter", Router)
    retriever = BasinRAGRetriever(
        engine=Engine(), encoder=encoder, query_prompt="instruction: ", use_rerank=False
    )

    first = retriever._encode_query("find evidence")
    second = retriever._encode_query("find evidence")

    assert encoder.calls == ["instruction: find evidence"]
    assert first is second
    assert first == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))


def test_retriever_reuses_one_embedding_for_router_hybrid_and_global_search(monkeypatch):
    import basinrag.retriever.base as base_module

    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return np.array([1.0, 0.0], dtype=np.float32)

    class Router:
        def __init__(self, _encoder, *, query_prompt):
            self.seen = []

        def route_with_embeddings(self, query, query_embedding):
            self.seen.append(query_embedding)
            return "global"

    class Hybrid:
        def __init__(self):
            self.seen = []

        def search_nodes(self, query, query_embedding, **kwargs):
            self.seen.append(query_embedding)
            return [{"id": "seed", "text": "seed evidence", "score": 1.0}]

    class Global:
        def __init__(self):
            self.seen = []

        def search_structured(self, query, *, top_k_basins, query_embedding):
            self.seen.append(query_embedding)
            return []

    class Engine:
        def __init__(self):
            self.graph = nx.Graph()
            self.graph.add_node("seed", text="seed evidence", embedding=np.array([1.0, 0.0]))
            self.basins = {}
            self.index_metadata = {}

    encoder = Encoder()
    router = Router(encoder, query_prompt="")
    hybrid = Hybrid()
    global_search = Global()

    def install_stubs(self):
        self._local = SimpleNamespace(search_nodes=lambda *_args, **_kwargs: [])
        self._global = global_search
        self._hybrid = hybrid
        self._reranker = None
        self._router = router

    monkeypatch.setattr(base_module, "IntelligentQueryRouter", Router)
    monkeypatch.setattr(BasinRAGRetriever, "_rebuild_indexes", install_stubs)
    retriever = BasinRAGRetriever(
        engine=Engine(), encoder=encoder, search_type="auto", use_rerank=False
    )

    packet = retriever.brief("a query routed globally", top_k=1)

    assert len(encoder.calls) == 1
    assert len(router.seen) == len(hybrid.seen) == len(global_search.seen) == 1
    assert router.seen[0] is hybrid.seen[0] is global_search.seen[0]
    assert packet.node_ids == ["seed"]


@pytest.mark.parametrize("top_k", [0, 51])
def test_retriever_rejects_out_of_range_top_k(top_k):
    retriever = BasinRAGRetriever.model_construct(top_k=5)
    with pytest.raises(ValueError, match="1 and 50"):
        retriever.configure(top_k=top_k)


def test_global_search_batches_and_invalidates_summary_embeddings_by_build_and_model():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **_kwargs):
            texts = list(texts)
            self.calls.append(texts)
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    class Basin:
        def __init__(self, basin_id, summary):
            self.rho_tree = nx.DiGraph()
            self.rho_tree.add_node(
                basin_id,
                l3_source="llm",
                l3_summary=summary,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
            self.rho_tree.add_node(
                f"chunk-{basin_id}",
                text=f"evidence for {basin_id}",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                hops=0,
            )

    engine = SimpleNamespace(
        build_id="build-1",
        basins={"b1": Basin("b1", "summary one"), "b2": Basin("b2", "summary two")},
    )
    encoder = Encoder()
    search = TopologicalGlobalSearch(
        engine, encoder, encoder_revision="encoder-sha-a", query_prompt="instruction: "
    )
    query = np.array([1.0, 0.0], dtype=np.float32)

    search.search_structured("question", query_embedding=query)
    search.search_structured("question", query_embedding=query)
    assert len(encoder.calls) == 1
    assert set(encoder.calls[0]) == {"summary one", "summary two"}

    engine.build_id = "build-2"
    search.search_structured("question", query_embedding=query)
    search.encoder_revision = "encoder-sha-b"
    search.search_structured("question", query_embedding=query)
    engine.basins["b1"].rho_tree.nodes["b1"]["l3_summary"] = "summary one changed"
    search.search_structured("question", query_embedding=query)

    assert set(encoder.calls[1]) == {"summary one", "summary two"}
    assert set(encoder.calls[2]) == {"summary one", "summary two"}
    assert encoder.calls[3] == ["summary one changed"]


def test_router_centroids_are_owned_by_the_encoder_instance():
    class GroupEncoder:
        def __init__(self, directions):
            self.directions = directions
            self.batch = 0

        def encode(self, values, **_kwargs):
            assert isinstance(values, list)
            direction = self.directions[self.batch]
            self.batch += 1
            return np.tile(np.asarray(direction, dtype=np.float32), (len(values), 1))

    global_router = IntelligentQueryRouter(
        GroupEncoder([(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)])
    )
    hybrid_router = IntelligentQueryRouter(
        GroupEncoder([(0.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
    )
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    query = "explain the mechanism discussed in this passage"

    assert global_router.route_with_embeddings(query, query_vector) == "global"
    assert hybrid_router.route_with_embeddings(query, query_vector) == "hybrid"


def test_hybrid_graph_expansion_obeys_candidate_cap_and_ignores_invalid_seed():
    graph = nx.Graph()
    graph.add_edge("seed", "seq-a", type="sequential")
    graph.add_edge("seed", "seq-b", type="sequential")
    graph.add_edge("seq-a", "seq-c", type="sequential")
    engine = SimpleNamespace(
        graph=graph,
        basins={},
        bm25=SimpleNamespace(n=0),
        basin_id_of=lambda _node_id: None,
    )

    class Local:
        def _neighbors_of_type(self, node_id, edge_type):
            return [
                neighbor
                for neighbor in graph.neighbors(node_id)
                if graph.edges[node_id, neighbor].get("type") == edge_type
            ]

    search = HybridSearch(engine, Local())

    candidates = search._expand_graph_candidates(
        ["missing", "seed"], max_extra=2, sequential_radius=2
    )

    assert len(candidates) == 2
    assert set(candidates) <= {"seq-a", "seq-b", "seq-c"}
    assert search._expand_graph_candidates(["missing"], max_extra=10) == []
    assert search._expand_graph_candidates(["seed"], max_extra=0) == []
