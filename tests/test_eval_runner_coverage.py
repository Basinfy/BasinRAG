from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from basinrag.eval import failure_audit, qasper_evidence, run_gate, run_mteb, swebench
from basinrag.eval.mteb_wrapper import BasinRAGMTEBWrapper, SkipLargeCorpus


class _FakeEngine:
    def __init__(self):
        self.graph = nx.Graph()
        self.basins = {}
        self.encoder_model = ""
        self.build_id = ""
        self.index_metadata = {}
        self.gate_cache_tag = ""

    def build_graph(self, nodes):
        self.graph.clear()
        for node in nodes:
            self.graph.add_node(node["id"], **node)

    def partition_into_basins(self):
        self.basins = {
            node_id: SimpleNamespace(rho_tree=nx.Graph([(node_id, node_id)]))
            for node_id in self.graph
        }


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
    def __init__(self, *, loaded=False, saved=True):
        self.loaded = loaded
        self.saved = saved
        self.save_calls = 0

    def load_topology(self, engine):
        return self.loaded

    def current_build_id(self):
        return "base-build"

    def save_topology(self, engine):
        self.save_calls += 1
        return self.saved


class _FakeRag:
    def __init__(self, *, loaded=False, saved=True):
        self.engine = _FakeEngine()
        self.persistence = _FakePersistence(loaded=loaded, saved=saved)
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
            embedding_batch_size=2,
        )
        self.retriever = None
        self.attach_calls = 0

    def _attach_bm25(self):
        self.attach_calls += 1

    @staticmethod
    def _chunk_policy_metadata(config):
        return {
            "chunking_mode": "characters",
            "chunk_policy_version": 1,
            "chunk_size": 10,
            "chunk_overlap": 0,
            "chunk_size_tokens": None,
            "chunk_overlap_tokens": None,
        }

    @staticmethod
    def _source_manifest(nodes):
        return {str(node["source"]): str(node["id"]) for node in nodes}


def test_qasper_text_helpers_cover_empty_containment_and_sampling():
    assert qasper_evidence.tokenize("A an Alpha, BETA!") == {"alpha", "beta"}
    assert not qasper_evidence.evidence_overlap("a an", ["useful evidence"])
    assert not qasper_evidence.evidence_overlap("alpha beta", ["gamma delta"])
    assert qasper_evidence.evidence_overlap(
        "prefix alpha beta gamma suffix", ["alpha beta gamma"], min_jaccard=0.99
    )
    assert qasper_evidence._mid_excerpts("") == []
    assert qasper_evidence._mid_excerpts("short", span=10) == ["short"]
    excerpts = qasper_evidence._mid_excerpts("0123456789 " * 400, n=3, span=120)
    assert len(excerpts) == 3
    assert all(len(excerpt) > 80 for excerpt in excerpts)


def test_qasper_loaders_parse_fake_datasets_and_fallback(monkeypatch):
    calls = []
    qasper_rows = [
        {
            "id": "paper-1",
            "title": "Title",
            "abstract": "Abstract",
            "full_text": {
                "section_name": ["Methods", ""],
                "paragraphs": [["first", "second"], ["body"]],
            },
            "qas": {
                "question": ["What happened?", "No evidence"],
                "answers": [
                    {
                        "answer": [
                            {
                                "evidence": ["first evidence", ""],
                                "highlighted_evidence": ["highlight"],
                            },
                            "ignored",
                        ]
                    },
                    {},
                ],
            },
        }
    ]

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError("trust flag required")
        return qasper_rows

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    corpus, queries, evidence = qasper_evidence.load_qasper_evidence(None, None)
    assert "## Methods\nfirst\nsecond" in corpus["paper-1"]
    assert queries == {"paper-1::q0": "What happened?"}
    assert evidence["paper-1::q0"] == ["first evidence", "highlight"]
    assert calls[-1]["trust_remote_code"] is True

    expected = ({"fallback": "doc"}, {"q": "query"}, {"q": ["evidence"]})
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(qasper_evidence, "load_arxiv_passage_evidence", lambda *args: expected)
    assert qasper_evidence.load_qasper_evidence(1, 1) == expected


def test_arxiv_passage_loader_filters_caps_and_enforces_minimum(monkeypatch):
    rows = [
        {"id": "bad", "article": "tiny", "abstract": "short"},
        *[
            {
                "id": f"paper-{index}",
                "article": ("body evidence words " * 2500) if index == 0 else ("body evidence words " * 150),
                "abstract": "abstract query terms " * 8,
            }
            for index in range(12)
        ],
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: rows)
    corpus, queries, evidence = qasper_evidence.load_arxiv_passage_evidence(12, 10)
    assert len(corpus) == 12
    assert len(queries) == len(evidence) == 10
    assert len(corpus["paper-0"]) == 40000

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: rows[:2])
    with pytest.raises(RuntimeError, match="too small"):
        qasper_evidence.load_arxiv_passage_evidence(None, None)


def test_evaluate_passage_recall_uses_requested_graph_mode_and_skips_missing_gold():
    class Hybrid:
        def __init__(self):
            self.calls = []

        def search_nodes(self, query, embedding, **kwargs):
            self.calls.append((query, kwargs))
            if query == "match":
                return [{"text": "alpha beta gamma and context"}, {"text": "noise only"}]
            return [{"text": "unrelated material"}]

    hybrid = Hybrid()
    searcher = SimpleNamespace(encode=lambda query: np.ones(2), hybrid=hybrid)
    result = qasper_evidence.evaluate_passage_recall(
        searcher,
        {"q0": "ignored", "q1": "match", "q2": "miss"},
        {"q1": ["alpha beta gamma", "other evidence"], "q2": ["target phrase"]},
        top_k=2,
        expand_graph=False,
    )
    assert result == {"hit@10": 0.5, "evidence_recall@10": 0.25, "n_queries": 2}
    assert all(call[1]["expand_graph"] is False for call in hybrid.calls)
    assert qasper_evidence.evaluate_passage_recall(searcher, {"q": "x"}, {}, top_k=1)["n_queries"] == 0


def test_failure_audit_traces_all_stages_and_aggregates(monkeypatch):
    engine = _FakeEngine()
    for node_id, doc_id in (("n1", "doc-1"), ("n2", "doc-2"), ("n3", "doc-3")):
        engine.graph.add_node(node_id, text=node_id, source=doc_id, metadata={"doc_id": doc_id})
    engine.graph.add_edges_from([("n1", "n2"), ("n2", "n3")])

    class Searcher:
        bm25 = SimpleNamespace(score=lambda query, top_k: [("n1", 2.0), ("n2", 1.0)])
        local = SimpleNamespace(dense_hits=lambda emb, top_k: [{"id": "n1"}, {"id": "n3"}])
        hybrid = object()
        reranker = object()

        def __init__(self):
            self.engine = engine
            self.reranked = False

        @staticmethod
        def encode(query):
            return np.ones(2)

        def _rerank(self, query, hits, top_k):
            self.reranked = True
            return [{"id": "n2", "text": "n2"}]

    searcher = Searcher()
    row = failure_audit.audit_query_stages(
        searcher, "query", {"doc-1"}, top_k=3, candidate_k=3, use_rerank=True
    )
    assert searcher.reranked
    assert row["stage_hit"]["bm25"] is True
    assert row["stage_hit"]["ce"] is False
    assert row["drop_at"] == "ce"
    assert row["n_relevant"] == 1
    summary = failure_audit.aggregate_audit([row, row])
    assert summary["n_queries"] == 2
    assert summary["drop_heatmap"] == {"ce": 2}
    assert failure_audit.aggregate_audit([]) == {"n_queries": 0}
    assert failure_audit._hits_relevant(["a", "b"], {"b"}, 2)
    assert failure_audit.compare_hnsw_vs_flat(
        SimpleNamespace(local=SimpleNamespace(_index=None, _node_ids=None)), {}, {}
    )["error"] == "no FAISS index"


def test_run_corpus_audit_isolated_with_fakes(monkeypatch, tmp_path):
    fake_rag = _FakeRag()
    monkeypatch.setattr(failure_audit, "BasinRAGConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(failure_audit, "BasinRAG", lambda config: fake_rag)
    indexed = []
    monkeypatch.setattr(failure_audit, "index_flat_docs", lambda rag, corpus: indexed.append("flat"))
    monkeypatch.setattr(failure_audit, "index_long_docs", lambda *args: indexed.append("long"))
    monkeypatch.setattr(failure_audit, "maybe_reranker", lambda *args: "rr")
    monkeypatch.setattr(failure_audit, "GateSearcher", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        failure_audit,
        "audit_query_stages",
        lambda *args, **kwargs: {
            "stage_hit": {stage: True for stage in failure_audit.STAGES},
            "stage_recall": {stage: 1.0 for stage in failure_audit.STAGES},
            "drop_at": None,
            "n_relevant": 1,
        },
    )
    monkeypatch.setattr(failure_audit, "compare_hnsw_vs_flat", lambda *args, **kwargs: {"n_queries": 1})
    result = failure_audit.run_corpus_audit(
        "tiny",
        {"d": "text"},
        {"q": "query", "unused": "x"},
        {"q": {"d"}},
        str(tmp_path),
        "encoder",
        "reranker",
        True,
        10,
        1,
        5,
        1,
        True,
        True,
    )
    assert indexed == ["flat"]
    assert result["summary"]["n_queries"] == 1
    assert result["hnsw_vs_flat"] == {"n_queries": 1}


def test_run_gate_dataset_loaders_use_fake_rows(monkeypatch):
    datasets = {
        "corpus": [{"_id": "d1", "title": "Title", "text": "Body"}],
        "queries": [{"_id": "q1", "text": "Question"}],
        "test": [
            {"query-id": "q1", "corpus-id": "d1", "score": 1},
            {"query-id": "q1", "corpus-id": "ignored", "score": 0},
        ],
    }

    def fake_load(name, subset=None, *, split, **kwargs):
        return datasets[split]

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    with pytest.raises(RuntimeError, match="scifact-revision"):
        run_gate.load_scifact()
    corpus, queries, qrels = run_gate.load_scifact("a" * 40)
    assert corpus == {"d1": "Title Body"}
    assert queries == {"q1": "Question"}
    assert qrels == {"q1": {"d1"}}

    qasper_rows = [
        {
            "id": "paper",
            "title": "T",
            "abstract": "A",
            "full_text": {"section_name": ["S"], "paragraphs": [["P"]]},
            "qas": {"question": ["", "Q"], "question_id": [None, "qid"]},
        }
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: qasper_rows)
    corpus, queries, qrels = run_gate.load_qasper(None, 1, revision="b" * 40)
    assert corpus["paper"].endswith("## S\nP")
    assert queries == {"qid": "Q"}
    assert qrels == {"qid": {"paper"}}


def test_run_gate_arxiv_loader_limit_and_query_filter(monkeypatch):
    rows = [
        {
            "id": f"p{index}",
            "article": "article body " * 400,
            "abstract": "abstract query " * 10,
        }
        for index in range(35)
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: rows)
    with pytest.raises(RuntimeError, match="arxiv-revision"):
        run_gate.load_arxiv_longdoc(None, None)
    corpus, queries, qrels = run_gate.load_arxiv_longdoc(30, 30, revision="c" * 40)
    assert len(corpus) == len(queries) == len(qrels) == 30
    limited_queries, limited_qrels = run_gate._limit_queries(
        {"q1": "one", "missing": "no rel", "q2": "two"},
        {"q1": {"d1"}, "q2": {"d2"}},
        1,
    )
    assert limited_queries == {"q1": "one"}
    assert limited_qrels == {"q1": {"d1"}}
    original_queries = {"q": "x"}
    original_qrels = {"q": {"d"}}
    assert run_gate._limit_queries(original_queries, original_qrels, None) == (original_queries, original_qrels)


def test_run_gate_index_helpers_build_and_surface_save_failures(monkeypatch):
    monkeypatch.setattr(run_gate, "_set_gate_manifest", lambda *args, **kwargs: None)
    rag = _FakeRag()
    run_gate.index_flat_docs(rag, {"d1": "alpha", "d2": "beta"})
    assert set(rag.engine.graph) == {"d1", "d2"}
    assert rag.engine.build_id == "base-build"
    assert rag.attach_calls == 1
    assert rag.persistence.save_calls == 1

    failing = _FakeRag(saved=False)
    with pytest.raises(RuntimeError, match="SciFact"):
        run_gate.index_flat_docs(failing, {"d": "text"})

    long_rag = _FakeRag()
    run_gate.index_long_docs(long_rag, {"doc": "alpha beta gamma delta"}, 8, 1)
    assert len(long_rag.engine.graph) > 1
    assert all(data["source"] == "doc" for _, data in long_rag.engine.graph.nodes(data=True))


def test_mteb_wrapper_indexes_dict_cache_and_limits():
    rag = _FakeRag()
    wrapper = BasinRAGMTEBWrapper(rag, force_reindex=True, title_boost=2, cache_tag="tag")
    wrapper.index(
        {
            "d1": {"title": "Title", "text": "Body"},
            "d2": {"title": "", "text": "Second"},
        }
    )
    assert rag.engine.graph.nodes["d1"]["text"] == "Title Title Body"
    assert rag.engine.gate_cache_tag == "tag"
    assert wrapper._is_flat_index is True
    assert rag.persistence.save_calls == 1
    from basinrag.core.persistence import BasinPersistence
    BasinPersistence._validate_index_metadata(rag.engine.index_metadata)
    assert rag.engine.index_metadata["format_version"] == 3
    assert rag.engine.index_metadata["reranker_revision"] == "disabled"

    limited = BasinRAGMTEBWrapper(_FakeRag(), force_reindex=True, max_corpus_docs=1)
    with pytest.raises(SkipLargeCorpus, match="2 docs"):
        limited.index({"d1": {"text": "one"}, "d2": {"text": "two"}})

    cached_rag = _FakeRag(loaded=True)
    cached_rag.engine.graph.add_node("d", chunk_index=0)
    cached_rag.engine.basins = {"d": object()}
    cached_rag.engine.gate_cache_tag = "Task|test|default"
    cached = BasinRAGMTEBWrapper(cached_rag)
    cached.index({}, task_metadata=SimpleNamespace(name="Task"))
    assert cached_rag.attach_calls == 1
    assert cached_rag.persistence.save_calls == 0


def test_mteb_wrapper_search_covers_ce_tail_plain_local_and_empty():
    class Reranker:
        _model = object()

        @staticmethod
        def predict_scores(query, passages):
            return [0.2, 0.9][: len(passages)]

    ranked = [
        {"id": "n1", "text": "one", "score": 0.8, "metadata": {"doc_id": "d1"}},
        {"id": "n2", "text": "two", "score": 0.7, "metadata": {"doc_id": "d2"}},
        {"id": "n3", "text": "three", "score": 0.5, "metadata": {"doc_id": "d3"}},
    ]

    class Retriever:
        _reranker = Reranker()
        _local = SimpleNamespace(search_nodes=lambda emb, top_k: ranked)
        _hybrid = SimpleNamespace(search_nodes=lambda *args, **kwargs: ranked)

        @staticmethod
        def _encode_query(query):
            return np.ones(2)

    rag = _FakeRag()
    rag.retriever = Retriever()
    rag._ensure_retriever = lambda **kwargs: None
    wrapper = BasinRAGMTEBWrapper(rag, rerank_top_k=2, max_queries=2, use_rerank=True)
    scores = wrapper.search({"q1": "first", "q2": "second", "q3": "ignored"}, top_k=3)
    assert list(scores) == ["q1", "q2"]
    assert list(scores["q1"])[:2] == ["d2", "d1"]
    assert scores["q1"]["d3"] < scores["q1"]["d1"]

    wrapper.search_type = "local"
    wrapper.use_rerank = False
    plain = wrapper.search({"q": "local"}, top_k=2)
    assert list(plain["q"]) == ["d1", "d2"]

    Retriever._hybrid = SimpleNamespace(search_nodes=lambda *args, **kwargs: [])
    wrapper.search_type = "hybrid"
    assert wrapper.search({"q": "empty"}, top_k=2) == {"q": {}}


def test_run_mteb_metadata_and_source_error_branches(monkeypatch):
    assert run_mteb._encoder_tag("org/model-name") == "org_model_name"
    assert run_mteb._task_metadata(SimpleNamespace(), "Fallback") == {
        "corpus": "Fallback",
        "expected_splits": None,
    }
    metadata = SimpleNamespace(dataset={"path": "org/data"}, eval_splits="test")
    assert run_mteb._task_metadata(SimpleNamespace(metadata=metadata), "fallback") == {
        "corpus": "org/data",
        "expected_splits": ["test"],
    }
    metadata.eval_splits = 7
    assert run_mteb._task_metadata(SimpleNamespace(metadata=metadata), "fallback")["expected_splits"] is None

    monkeypatch.setattr(run_mteb.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no git")))
    assert run_mteb._source_revision() == {"commit": None, "dirty": None, "revision": "unknown"}


def test_swebench_record_workspace_and_dataset_helpers(monkeypatch, tmp_path):
    record = {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "base_commit": "abc",
        "problem_statement": "bug",
        "patch": "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n",
        "FAIL_TO_PASS": '["test_a"]',
        "PASS_TO_PASS": "not-json",
    }
    instance = swebench.SWEBenchInstance.from_hf_record(record)
    assert instance.fail_to_pass == ["test_a"]
    assert instance.pass_to_pass == ["not-json"]
    assert swebench._normalise_repo_path(" ./src\\a.py ") == "src/a.py"
    assert swebench._matches_gold_path("./src/a.py", instance.ground_truth_files)
    assert not swebench._matches_gold_path("a.py", instance.ground_truth_files)
    assert len(swebench._checkpoint_subset_id([instance])) == 64

    commands = []
    monkeypatch.setattr(swebench.subprocess, "run", lambda cmd, **kwargs: commands.append((cmd, kwargs)))
    manager = swebench.RepoWorkspaceManager(str(tmp_path / "repos"))
    repo_dir = manager.prepare_repo("org/repo", "abc")
    assert repo_dir == (tmp_path / "repos" / "org__repo").resolve()
    assert [call[0][1] for call in commands] == ["clone", "reset", "clean", "checkout"]

    rows = [
        record,
        {**record, "instance_id": "other-1", "repo": "other/repo"},
        {**record, "instance_id": "org__repo-2"},
    ]
    monkeypatch.setattr(swebench, "load_dataset", lambda *args, **kwargs: rows)
    evaluator = swebench.SWEBenchEvaluator(str(tmp_path / "workspaces"))
    selected = evaluator.load_dataset(repo_filter="org/repo", limit=2, limit_per_repo=1)
    assert [item.instance_id for item in selected] == ["org__repo-1"]


def test_code_repo_ingestor_skips_tests_and_builds_nodes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "app.py").write_text("def useful():\n    return 1\n" * 80, encoding="utf-8")
    (repo / "src" / "test_hidden.py").write_text("secret", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("secret", encoding="utf-8")
    rag = _FakeRag()
    count = swebench.CodeRepoIngestor.ingest_codebase(rag, repo)
    assert count > 0
    assert rag.attach_calls == 1
    assert {data["source"] for _, data in rag.engine.graph.nodes(data=True)} == {"src/app.py"}
    assert rag.retriever is None
