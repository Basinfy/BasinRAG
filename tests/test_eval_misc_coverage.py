"""Offline coverage for evaluation utilities that must not need benchmark data."""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basinrag.eval import beir, datasets, metrics
from basinrag.eval import generate_mteb_metadata as mteb_metadata
from basinrag.eval import swebench_protocol


def _import_optional_spacy_module(name: str):
    """Import the baseline without making spaCy a test-suite dependency."""
    existing = sys.modules.get("spacy")
    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = lambda _name: None
    sys.modules["spacy"] = fake_spacy
    try:
        return importlib.import_module(name)
    finally:
        if existing is None:
            sys.modules.pop("spacy", None)
        else:
            sys.modules["spacy"] = existing


def test_beir_cache_extract_load_and_zip_slip_guard(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    archive_path = cache / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("tiny/corpus.jsonl", '{"_id":"d1","title":"Title","text":"Body"}\n')
        archive.writestr("tiny/queries.jsonl", '{"_id":"q1","text":"Question"}\n')
        archive.writestr("tiny/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td0\t0\ninvalid\n")

    extracted = Path(beir.download_and_unzip_beir_dataset("tiny", str(cache)))
    assert (extracted / "corpus.jsonl").is_file()
    corpus, queries, qrels = beir.load_beir_dataset(str(extracted))
    assert corpus == {"d1": "Title Body"}
    assert queries == {"q1": "Question"}
    assert qrels == {"q1": {"d1"}}

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with zipfile.ZipFile(bad_dir / "unsafe.zip", "w") as archive:
        archive.writestr("../outside.txt", "not allowed")
    with pytest.raises(ValueError, match="zip-slip"):
        beir.download_and_unzip_beir_dataset("unsafe", str(bad_dir))
    assert not (tmp_path / "outside.txt").exists()


def test_beir_adapter_deduplicates_and_ingest_builds_fake_graph(monkeypatch):
    class Rag:
        def __init__(self):
            self.calls = []
            self.ingestor = SimpleNamespace(encoder=SimpleNamespace(encode=self.encode))
            self.config = SimpleNamespace(encoder_model="fake/encoder")
            self.engine = SimpleNamespace(build_graph=self.build_graph, partition_into_basins=lambda: None)
            self.nodes = []
            self.attached = False
            self.retriever = object()

        def query(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return [
                SimpleNamespace(metadata={"doc_id": "d1"}, score=0.0),
                SimpleNamespace(metadata={"source": "d1"}, score=0.8),
                SimpleNamespace(metadata={"node_id": "d2"}, score=0.4),
                SimpleNamespace(metadata={}, score=0.3),
            ]

        @staticmethod
        def encode(texts, **kwargs):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        def build_graph(self, nodes):
            self.nodes = nodes

        def _attach_bm25(self):
            self.attached = True

    rag = Rag()
    adapter = beir.BasinRAGBEIRAdapter(rag)
    ranked = adapter.search_beir({"q": "question"}, top_k=2)
    assert ranked == {"q": {"d1": 1.0, "d2": 0.4}}
    assert rag.calls[-1][1]["top_k"] == 4
    assert adapter.search({"q": "question"}, {"d1": "one"}, top_k=2) == {"q": ["d1"]}

    monkeypatch.setattr("basinrag.indexer.condensation.node_layers", lambda text: {"l1": "L1", "l2": "L2"})
    count = beir.ingest_beir_corpus(rag, {"d1": "first", "d2": "second"}, batch_size=3)
    assert count == 2
    assert [node["id"] for node in rag.nodes] == ["d1", "d2"]
    assert rag.nodes[0]["metadata"] == {"doc_id": "d1"}
    assert rag.attached and rag.retriever is None
    assert beir.ingest_beir_corpus(rag, {}) == 0


def test_compare_beir_pipeline_uses_fakes_and_writes_summary(monkeypatch, tmp_path):
    compare_beir = _import_optional_spacy_module("basinrag.eval.compare_beir")
    monkeypatch.setattr(compare_beir, "download_and_unzip_beir_dataset", lambda *args, **kwargs: "fake-dataset")
    monkeypatch.setattr(
        compare_beir,
        "load_beir_dataset",
        lambda *args, **kwargs: (
            {"d1": "doc"},
            {"q1": "valid", "q2": "no labels"},
            {"q1": {"d1"}},
        ),
    )

    class FakeRag:
        def query(self, query, **kwargs):
            return [SimpleNamespace(metadata={"doc_id": "d1"}, score=0.75)]

    monkeypatch.setattr(compare_beir.BasinRAG, "create", lambda **kwargs: FakeRag())
    monkeypatch.setattr(compare_beir, "ingest_beir_corpus", lambda rag, corpus: len(corpus))

    class FakeGraphRAG:
        def build_graph(self, corpus):
            self.count = len(corpus)

        def search_beir(self, queries, top_k):
            return {qid: {"d1": 0.5} for qid in queries}

    monkeypatch.setattr(compare_beir, "GraphRAGBaseline", FakeGraphRAG)

    class FakeEvaluator:
        @staticmethod
        def evaluate(qrels, results, ks):
            assert set(qrels) == set(results) == {"q1"}
            return (
                {f"NDCG@{k}": 0.5 for k in ks},
                {f"MAP@{k}": 0.4 for k in ks},
                {f"Recall@{k}": 0.6 for k in ks},
                {f"P@{k}": 0.3 for k in ks},
            )

        @staticmethod
        def evaluate_custom(qrels, results, ks, metric):
            assert metric == "mrr"
            return {f"MRR@{k}": 0.7 for k in ks}

    monkeypatch.setattr(compare_beir, "EvaluateRetrieval", FakeEvaluator)
    output = tmp_path / "nested" / "summary.json"
    summary = compare_beir.BEIRComparativeBenchmarker(str(tmp_path / "cache")).run_comparison(
        num_queries=10, output_file=str(output)
    )
    assert summary["num_docs"] == 1
    assert summary["num_queries"] == 1
    assert summary["basinrag"]["ndcg"]["NDCG@10"] == 0.5
    assert json.loads(output.read_text(encoding="utf-8"))["graphrag"]["mrr"]["MRR@10"] == 0.7


class _FakeNLPDoc:
    def __init__(self, text):
        self.ents = [
            SimpleNamespace(label_="ORG", lemma_="Acme Systems"),
            SimpleNamespace(label_="DATE", lemma_="ignored date"),
        ] if "NamedProbe" in text else []
        self.noun_chunks = [
            SimpleNamespace(root=SimpleNamespace(pos_="PRON"), lemma_="ignored pronoun"),
            SimpleNamespace(root=SimpleNamespace(pos_="NOUN"), lemma_="useful concept"),
        ] if "NamedProbe" in text else []


def test_graphrag_baseline_entity_graph_and_search(monkeypatch):
    graphrag_baseline = _import_optional_spacy_module("basinrag.eval.graphrag_baseline")
    monkeypatch.setattr(graphrag_baseline.spacy, "load", lambda _name: _FakeNLPDoc)
    model = graphrag_baseline.GraphRAGBaseline()
    entities = model._extract_entities("NamedProbe UsefulIdentifier DateProbe")
    assert "acme systems" in entities
    assert "useful concept" in entities
    assert "ignored date" not in entities
    assert "ignored pronoun" not in entities

    model.build_graph(
        {
            "direct": "QuasarAlpha SharedBridge",
            "neighbor": "SharedBridge NebulaBeta",
            "third": "UnrelatedGamma",
            "fourth": "AnotherDelta",
        }
    )
    assert model.entity_df["sharedbridge"] == 2
    assert model._idf("missing") == 0.0
    ranked = model.search_single("QuasarAlpha", top_k=3)
    assert ranked[0][0] == "direct"
    assert {doc_id for doc_id, _ in ranked} == {"direct", "neighbor"}
    assert model.search_single("no_known_term") == []
    assert model.search_beir({"q": "QuasarAlpha"}, top_k=1)["q"]
    assert model.search({"q": "QuasarAlpha"}, {}, top_k=1)["q"] == ["direct"]
    model.build_graph({})
    assert model.search_single("QuasarAlpha") == []


def test_mteb_metadata_parser_yaml_and_cli(tmp_path, monkeypatch):
    result_path = tmp_path / "SciFact.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "SciFact",
                "dataset_revision": "rev-1",
                "scores": {
                    "validation": [
                        {
                            "ndcg_at_10": 0.123456789,
                            "accuracy": 0.5,
                            "custom_metric": 0.25,
                            "nauc_at_10": 0.9,
                            "label": "ignored",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    parsed = mteb_metadata.parse_task_results(result_path)
    assert parsed["task"] == {"type": "Retrieval"}
    assert parsed["dataset"]["split"] == "validation"
    assert parsed["dataset"]["revision"] == "rev-1"
    assert {metric["type"]: metric["value"] for metric in parsed["metrics"]} == {
        "ndcg_at_10": 12.34568,
        "accuracy": 50.0,
        "custom_metric": 25.0,
    }
    yaml = mteb_metadata.generate_metadata_yaml("org/model", [parsed])
    assert "- name: org/model" in yaml
    assert "revision: rev-1" in yaml

    (tmp_path / "model_meta.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "metadata.md"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_mteb_metadata", str(tmp_path), "--model-name", "org/model", "--output", str(output)],
    )
    mteb_metadata.main()
    assert output.read_text(encoding="utf-8").startswith("---\npipeline_tag: sentence-similarity")

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr("sys.argv", ["generate_mteb_metadata", str(empty)])
    with pytest.raises(FileNotFoundError, match="Nenhum arquivo JSON"):
        mteb_metadata.main()


def test_faquad_cached_loader_and_extractors(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "faquad_dev.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Article",
                        "paragraphs": [
                            {
                                "context": "Paragraph context",
                                "qas": [
                                    {"id": "q1", "question": "Question?", "answers": [{"text": "Answer"}]},
                                    {"id": "q2", "question": "No answer", "answers": []},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    loaded = datasets.load_faquad(str(cache))
    corpus, queries = datasets.extract_faquad_corpus_and_queries(loaded)
    assert corpus == [{"id": "Article_0", "title": "Article", "text": "Paragraph context"}]
    assert [item["answers"] for item in queries] == [["Answer"], []]
    assert len(datasets.get_mmarco_subset()[0]) == 8
    assert len(datasets.get_mmarco_subset()[1]) == 5


def test_metrics_edge_cases_and_aggregation():
    assert metrics.mrr_at_k(["x", "gold"], {"gold"}, k=2) == 0.5
    assert metrics.ndcg_at_k(["gold", "other"], {"gold"}, k=2) == 1.0
    assert metrics.ndcg_at_k(["gold"], set(), k=2) == 0.0
    assert metrics.hit_rate_at_k(["other", "gold"], {"gold"}, k=2) == 1.0
    assert metrics.recall_at_k(["gold", "gold"], {"gold", "second"}, k=2) == 0.5
    assert metrics.recall_at_k([], set()) == 0.0
    assert metrics.evaluate_retrieval({}, {}) == {"mrr": 0.0, "ndcg": 0.0, "hit_rate": 0.0, "recall": 0.0}
    result = metrics.evaluate_retrieval(
        {"q1": {"gold"}, "q2": {"absent"}},
        {"q1": ["gold"], "q2": []},
        k=1,
    )
    assert result == {"mrr": 0.5, "ndcg": 0.5, "hit_rate": 0.5, "recall": 0.5}


def test_swebench_protocol_validation_io_and_runtime_branches(monkeypatch, tmp_path):
    assert swebench_protocol.dataset_alias("custom/dataset") == "custom/dataset"
    with pytest.raises(ValueError, match="empty"):
        swebench_protocol.resolve_dataset("  ")
    assert swebench_protocol.gold_files_from_patch("--- a/x.py\n--- a/dev/null\n--- a//dev/null\n") == {"x.py"}
    assert swebench_protocol.drop_leading_readme(["src/README.md", "README_CODE.py", "a.py"]) == ["README_CODE.py", "a.py"]
    assert swebench_protocol.official_retrieval_scores([], set()) == {
        "recall": 0.0,
        "any_recall": 0.0,
        "all_recall": 0.0,
    }
    assert not swebench_protocol.hit_at_k(["x"], set(), 1)
    assert swebench_protocol.mrr_at_k(["x"], {"gold"}, 1) == 0.0
    assert swebench_protocol.estimate_tokens("") == 0
    assert swebench_protocol.pack_files_to_budget(["missing", "a.py"], {"a.py": "one"}, 0) == ["a.py"]
    assert swebench_protocol.extract_model_patch("```patch\n--- a/a.py\n+++ b/a.py\n```") == "--- a/a.py\n+++ b/a.py\n"
    assert swebench_protocol.prediction_record("i", "m", "") == {
        "instance_id": "i",
        "model_name_or_path": "m",
        "model_patch": "",
    }
    with pytest.raises(ValueError, match="prediction missing"):
        swebench_protocol.write_predictions_jsonl(tmp_path / "bad.jsonl", [{"instance_id": "i"}])

    assert swebench_protocol.OfficialReport.from_harness(
        {"Instances submitted": 4, "Instances resolved": 2}, dataset="lite", run_id="r"
    ).resolved_rate == 0.5
    assert swebench_protocol.OfficialReport.from_harness({}, dataset="lite", run_id="r").resolved_rate == 0.0

    monkeypatch.setattr(swebench_protocol.shutil, "which", lambda name: None)
    assert not swebench_protocol.docker_available()
    monkeypatch.setattr(swebench_protocol.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(swebench_protocol.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert swebench_protocol.docker_available()
    monkeypatch.setattr(
        swebench_protocol.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(swebench_protocol.subprocess.TimeoutExpired("docker", 20)),
    )
    assert not swebench_protocol.docker_available()
    monkeypatch.setattr(swebench_protocol.shutil, "which", lambda name: "swebench" if name == "swebench" else None)
    command = swebench_protocol.build_eval_command(
        "lite", gold=True, run_id="run", timeout=10, modal=True, workers=2
    )
    assert command[0] == "swebench" and "--modal" in command and "-t" in command
    assert re.fullmatch(r"test-\d{8}T\d{6}Z", swebench_protocol.new_run_id("test"))


def test_compare_graphrag_checkpoint_resume_summarizes_without_work(monkeypatch, tmp_path):
    compare_graphrag = _import_optional_spacy_module("basinrag.eval.compare_graphrag")
    class FakeEvaluator:
        def __init__(self, _workspace):
            pass

        @staticmethod
        def load_dataset(**kwargs):
            assert kwargs["split"] == "test"
            return [SimpleNamespace(instance_id="issue-1")]

    monkeypatch.setattr(compare_graphrag, "SWEBenchEvaluator", FakeEvaluator)
    checkpoint = tmp_path / "checkpoint.jsonl"
    record = {
        "instance_id": "issue-1",
        "basinrag": {"hit@1": True, "hit@5": True, "hit@10": True, "recall": 0.5, "mrr": 1.0, "latency_ms": 4.0},
        "graphrag": {"hit@1": False, "hit@5": True, "hit@10": True, "recall": 1.0, "mrr": 0.5, "latency_ms": 8.0},
    }
    checkpoint.write_text(json.dumps(record) + "\n", encoding="utf-8")
    bench = compare_graphrag.ComparativeBenchmarker(str(tmp_path / "workspaces"))
    summary = bench.run_comparison(checkpoint_path=str(checkpoint))
    assert summary["instances_evaluated"] == 1
    assert summary["basinrag"]["mrr"] == 1.0
    assert summary["graphrag"]["recall@10"] == 1.0
    assert summary["basinrag"]["avg_build_time_s"] == 0.0
