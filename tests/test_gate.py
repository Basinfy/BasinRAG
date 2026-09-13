from __future__ import annotations

import pytest

from basinrag.eval.gate import (
    GateSearcher,
    _ndcg,
    _provenance_valid,
    _shared_provenance,
    aggregate_metrics,
    basin_diagnostics,
    collapse_to_docs,
    decide,
    evaluate_system,
    join_title_text,
    node_doc_id,
    recall_at_k,
    write_json,
)


def _provenance(dataset: str, split: str) -> dict:
    return {
        "dataset": dataset,
        "split": split,
        "dataset_revision": "a" * 40,
        "complete": True,
        "sampled": False,
        "encoder": "org/encoder",
        "encoder_revision": "b" * 40,
        "reranker": "org/reranker",
        "reranker_revision": "c" * 40,
        "query_prompt": "search instruction",
        "protocol_version": "gate-v2",
        "chunk_policy": {"size": 512, "overlap": 128},
        "n_queries": 100,
        "n_qrels": 100,
    }


def _valid_inputs():
    scifact = {
        "hybrid_min": {"ndcg@10": 0.60},
        "hybrid_min_topo": {"ndcg@10": 0.601},
    }
    qasper = {
        "encoder_pure": {"ndcg@10": 0.40},
        "hybrid_min": {"ndcg@10": 0.41},
        "hybrid_min_topo": {"ndcg@10": 0.43},
        "basinrag_full": {"ndcg@10": 0.44},
    }
    basins = {"gate_geometry_ok": True, "median_members": 8.0, "singleton_frac": 0.1}
    return scifact, qasper, basins, _provenance("SciFact", "test"), _provenance("QASPER", "validation")


def _decide(scifact, qasper, basins, scifact_provenance, qasper_provenance, **kwargs):
    return decide(
        scifact,
        qasper,
        basins,
        scifact_complete=True,
        qasper_complete=True,
        scifact_provenance=scifact_provenance,
        qasper_provenance=qasper_provenance,
        **kwargs,
    )


def test_valid_complete_protocol_can_continue_b():
    out = _decide(*_valid_inputs())
    assert out["DECISION"] == "CONTINUE_B"


def test_valid_complete_protocol_can_convert_c_when_topology_loses():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    qasper["hybrid_min_topo"]["ndcg@10"] = 0.39
    qasper["basinrag_full"]["ndcg@10"] = 0.40
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "CONVERT_C"


def test_absent_or_partial_scifact_never_decides():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    out = decide(
        scifact, qasper, basins,
        scifact_complete=False, qasper_complete=True,
        scifact_provenance=scifact_prov, qasper_provenance=qasper_prov,
    )
    assert out["DECISION"] == "GATE_INVALID"


def test_missing_qasper_never_uses_another_long_document_dataset():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    out = decide(
        scifact, qasper, basins,
        scifact_complete=True, qasper_complete=False,
        scifact_provenance=scifact_prov, qasper_provenance=qasper_prov,
    )
    assert out["DECISION"] == "GATE_INVALID"
    assert "QASPER" in out["reason"]


def test_incompatible_or_sampled_provenance_invalidates_gate():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    qasper_prov["encoder_revision"] = "different-revision"
    assert _decide(scifact, qasper, basins, scifact_prov, qasper_prov)["DECISION"] == "GATE_INVALID"

    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    qasper_prov["sampled"] = True
    assert _decide(scifact, qasper, basins, scifact_prov, qasper_prov)["DECISION"] == "GATE_INVALID"

    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    scifact_prov["dataset"] = "not-scifact"
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "GATE_INVALID"
    assert "SciFact provenance" in out["reason"]


@pytest.mark.parametrize("topo_score", [0.606, 0.594])
def test_scifact_absolute_control_deviation_invalid_in_both_directions(topo_score):
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    scifact["hybrid_min_topo"]["ndcg@10"] = topo_score
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "GATE_INVALID"
    assert "deviation" in out["reason"]


def test_missing_control_and_invalid_geometry_are_invalid():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    del scifact["hybrid_min_topo"]
    assert _decide(scifact, qasper, basins, scifact_prov, qasper_prov)["DECISION"] == "GATE_INVALID"

    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    basins["gate_geometry_ok"] = False
    assert _decide(scifact, qasper, basins, scifact_prov, qasper_prov)["DECISION"] == "GATE_INVALID"


def test_arxiv_metrics_cannot_substitute_for_qasper():
    scifact, _qasper, basins, scifact_prov, _qasper_prov = _valid_inputs()
    arxiv = {
        "encoder_pure": {"ndcg@10": 0.4},
        "hybrid_min": {"ndcg@10": 0.5},
        "hybrid_min_topo": {"ndcg@10": 0.7},
    }
    out = decide(
        scifact, arxiv, basins,
        scifact_complete=True, qasper_complete=False,
        scifact_provenance=scifact_prov,
        qasper_provenance=_provenance("arxiv_longdoc", "test"),
    )
    assert out["DECISION"] == "GATE_INVALID"


def test_explicit_invalid_reason_short_circuits_decision():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov, invalid_reasons=["sampled run"])
    assert out["DECISION"] == "GATE_INVALID"


@pytest.mark.parametrize(
    ("provenance", "dataset", "split"),
    [
        (None, "SciFact", "test"),
        ([], "SciFact", "test"),
        ({"dataset": "other"}, "SciFact", "test"),
        ({"dataset": "SciFact", "split": "train"}, "SciFact", "test"),
    ],
)
def test_provenance_rejects_missing_or_wrong_identity(provenance, dataset, split):
    assert _provenance_valid(provenance, dataset=dataset, split=split) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", 1),
        ("sampled", True),
        ("encoder", ""),
        ("encoder_revision", "unknown"),
        ("dataset_revision", "not-a-commit"),
        ("dataset_revision", "g" * 40),
        ("chunk_policy", None),
        ("protocol_version", ""),
        ("n_queries", "many"),
        ("n_queries", 0),
        ("n_qrels", -1),
    ],
)
def test_provenance_rejects_invalid_contract_fields(field, value):
    provenance = _provenance("SciFact", "test")
    provenance[field] = value
    assert not _provenance_valid(provenance, dataset="SciFact", split="test")


def test_provenance_accepts_numeric_counts_and_shared_protocol_is_exact():
    provenance = _provenance("SciFact", "test")
    provenance["n_queries"] = "10"
    provenance["n_qrels"] = 11.0
    assert _provenance_valid(provenance, dataset="SciFact", split="test")
    assert _shared_provenance(provenance) == _shared_provenance(dict(provenance))
    altered = dict(provenance, query_prompt="different prompt")
    assert _shared_provenance(provenance) != _shared_provenance(altered)


def test_decision_rejects_cross_dataset_protocol_mismatch_and_missing_longdoc():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    qasper_prov["query_prompt"] = "different prompt"
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "GATE_INVALID"
    assert "provenance differs" in out["reason"]

    scifact, _qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    out = _decide(scifact, {}, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "GATE_INVALID"
    assert "long-doc corpus missing" in out["reason"]


def test_decision_rejects_incomplete_longdoc_ablation_and_uses_topology_fallback():
    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    del qasper["encoder_pure"]
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "GATE_INVALID"
    assert "ablation incomplete" in out["reason"]

    scifact, qasper, basins, scifact_prov, qasper_prov = _valid_inputs()
    qasper["basinrag_full"] = {"skipped": True, "ndcg@10": 0.99}
    out = _decide(scifact, qasper, basins, scifact_prov, qasper_prov)
    assert out["DECISION"] == "CONTINUE_B"
    assert out["longdoc_best_basin_ndcg@10"] == pytest.approx(0.43)


def test_metric_aggregation_and_empty_relevance_handling():
    assert recall_at_k(["a"], set()) == 0.0
    empty = aggregate_metrics({"q": set()}, {"q": ["a"]})
    assert empty == {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@10": 0.0, "hit@10": 0.0, "n_queries": 0}

    metrics = aggregate_metrics(
        {"q1": {"a"}, "q2": {"x"}, "ignored": set()},
        {"q1": ["a", "a"], "q2": ["miss"]},
        k=2,
    )
    assert metrics["n_queries"] == 2
    assert metrics["recall@10"] == pytest.approx(0.5)
    assert metrics["hit@10"] == pytest.approx(0.5)
    assert metrics["mrr@10"] == pytest.approx(0.5)


class _TinyGraph:
    def __init__(self):
        import networkx as nx

        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(
            [
                ("n1", {"metadata": {"doc_id": "doc-a"}, "source": "src-a", "text": "alpha"}),
                ("n2", {"metadata": {}, "source": "src-a", "text": "alpha duplicate"}),
                ("n3", {"metadata": {}, "source": "", "text": "gamma"}),
            ]
        )
        self.nodes = self._graph.nodes

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def __contains__(self, node):
        return node in self._graph

    def __getitem__(self, node):
        return self._graph[node]


class _Tree:
    def __init__(self, size):
        self.size = size

    def number_of_nodes(self):
        return self.size


class _Engine:
    def __init__(self):
        self.graph = _TinyGraph()
        self.basins = {"b1": type("Basin", (), {"rho_tree": _Tree(3)})(), "b2": type("Basin", (), {"rho_tree": _Tree(1)})()}


def test_doc_collapse_and_basin_diagnostics_cover_fallbacks():
    engine = _Engine()
    assert node_doc_id(engine, "n1") == "doc-a"
    assert node_doc_id(engine, "n2") == "src-a"
    assert node_doc_id(engine, "n3") == "n3"
    assert collapse_to_docs(engine, [{"id": "n1"}, {"id": "n1"}, {"id": "n2"}, {"id": "n3"}], 5) == ["doc-a", "src-a", "n3"]
    assert collapse_to_docs(engine, [{"id": "n1"}, {"id": "n2"}], 1) == ["doc-a"]

    diagnostics = basin_diagnostics(engine)
    assert diagnostics["n_nodes"] == 3
    assert diagnostics["n_basins"] == 2
    assert diagnostics["n_sources"] == 2
    assert diagnostics["median_members"] == 2.0
    assert diagnostics["singleton_frac"] == 0.5
    assert diagnostics["gate_geometry_ok"] is False

    class EmptyGraph:
        def __init__(self):
            import networkx as nx

            self._graph = nx.DiGraph()
            self.nodes = self._graph.nodes

        def __getattr__(self, name):
            return getattr(self._graph, name)

    empty = type("EmptyEngine", (), {"graph": EmptyGraph(), "basins": {}})()
    empty_diagnostics = basin_diagnostics(empty)
    assert empty_diagnostics["median_members"] == 0.0
    assert empty_diagnostics["mean_hops"] == 0.0
    assert empty_diagnostics["singleton_frac"] == 1.0
    assert empty_diagnostics["gate_geometry_ok"] is False


def test_ndcg_rejects_missing_skipped_and_null_rows():
    assert _ndcg({}, "hybrid_min") is None
    assert _ndcg({"hybrid_min": {"skipped": True, "ndcg@10": 0.5}}, "hybrid_min") is None
    assert _ndcg({"hybrid_min": {"ndcg@10": None}}, "hybrid_min") is None
    assert _ndcg({"hybrid_min": {"ndcg@10": "0.5"}}, "hybrid_min") == 0.5


def test_title_join_and_json_writer(tmp_path):
    assert join_title_text(" title ", " body ") == "title body"
    assert join_title_text(" title ", " ") == "title"
    assert join_title_text("", " body ") == "body"
    assert join_title_text("", "") == ""
    output = tmp_path / "nested" / "report.json"
    write_json(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == '{\n  "ok": true\n}'


class _Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [3.0, 4.0]


class _Local:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def dense_hits(self, embedding, top_k):
        self.calls.append((embedding, top_k))
        return self.hits


class _Hybrid:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_nodes(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.hits


class _BM25:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def score(self, query, top_k):
        self.calls.append((query, top_k))
        return self.rows


def _fake_rag(hits=None, bm25=None):
    from types import SimpleNamespace

    hits = hits or [{"id": "n1", "text": "alpha", "score": 1.0}]
    engine = _Engine()
    hybrid = _Hybrid(hits)
    local = _Local(hits)
    rag = SimpleNamespace(
        engine=engine,
        retriever=SimpleNamespace(_hybrid=hybrid, _local=local),
        ingestor=SimpleNamespace(encoder=_Encoder()),
        _ensure_retriever=lambda **kwargs: None,
    )
    engine.bm25 = bm25
    return rag, hybrid, local


def test_gate_searcher_caches_normalized_prompted_query_embedding():
    rag, _, _ = _fake_rag()
    searcher = GateSearcher(rag, query_prompt="P: ")
    first = searcher.encode("question")
    second = searcher.encode("question")
    assert first is second
    assert rag.ingestor.encoder.calls == ["P: question"]
    assert sum(first ** 2) == pytest.approx(1.0)

    class ZeroEncoder:
        def encode(self, text):
            assert text == "question"
            return [0.0, 0.0]

    rag.ingestor.encoder = ZeroEncoder()
    unprompted = GateSearcher(rag, query_prompt="")
    assert list(unprompted.encode("question")) == [0.0, 0.0]


def test_gate_searcher_dispatches_all_systems_and_filters_missing_nodes():
    hits = [{"id": "n1", "text": "alpha", "score": 1.0}]
    bm25 = _BM25([("n2", 2.0), ("missing", 1.0)])
    rag, hybrid, local = _fake_rag(hits, bm25)
    searcher = GateSearcher(rag)
    assert searcher.search("q", "encoder_pure", top_k=1) == ["doc-a"]
    assert local.calls[-1][1] == 50
    assert searcher.search("q", "bm25_pure", top_k=1) == ["src-a"]
    assert bm25.calls == [("q", 50)]
    assert searcher.search("q", "hybrid_min", top_k=1) == ["doc-a"]
    assert searcher.search("q", "hybrid_min_topo", top_k=1) == ["doc-a"]
    assert hybrid.calls[-1][1]["use_hop_prior"] is True
    assert hybrid.calls[-1][1]["expand_graph"] is False
    with pytest.raises(ValueError, match="unknown system"):
        searcher.search("q", "typo")


def test_gate_searcher_handles_no_bm25_and_reranks_negative_logits_monotonically():
    hits = [{"id": "n1", "text": "alpha", "score": 1.0}]
    rag, hybrid, _ = _fake_rag(hits, bm25=None)

    class Reranker:
        def predict_scores(self, query, passages):
            assert query == "q"
            return [-2.0 for _ in passages]

    searcher = GateSearcher(rag, reranker=Reranker())
    assert searcher.search("q", "bm25_pure") == []
    assert searcher.search("q", "basinrag_full") == ["doc-a"]
    assert hybrid.calls[-1][1]["use_hop_prior"] is True
    assert searcher._rerank("q", [], top_k=5) == []


def test_evaluate_system_skips_empty_qrels_and_reports_aggregate(capsys):
    class Searcher:
        def __init__(self):
            self.queries = []

        def search(self, query, system, top_k):
            self.queries.append((query, system, top_k))
            return ["relevant"]

    searcher = Searcher()
    metrics, results = evaluate_system(
        searcher,
        {"q1": "one", "q2": "two", "q3": "three", "q4": "four"},
        {"q1": {"relevant"}, "q2": {"relevant"}, "q3": {"relevant"}, "q4": set(), "absent": {"relevant"}},
        "hybrid_min",
        top_k=1,
    )
    assert metrics["n_queries"] == 3
    assert results == {"q1": ["relevant"], "q2": ["relevant"], "q3": ["relevant"]}
    assert searcher.queries == [
        ("one", "hybrid_min", 1),
        ("two", "hybrid_min", 1),
        ("three", "hybrid_min", 1),
    ]
    output = capsys.readouterr().out
    assert "[hybrid_min] 1/3" in output
    assert "[hybrid_min] 3/3" in output
