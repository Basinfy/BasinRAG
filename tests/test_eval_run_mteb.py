import json
import sys
from pathlib import Path
from types import SimpleNamespace

from basinrag.eval import run_mteb
from basinrag.eval.run_mteb import (
    _build_provenance,
    _build_run_manifest,
    _model_identity,
    _read_current_task_scores,
    _summarize_run,
)
from basinrag.eval.mteb_wrapper import BasinRAGMTEBWrapper


def _write_result(root: Path, task: str, ndcg: float) -> None:
    path = root / "results" / "Basinfy__BasinRAG" / "1.0.4" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_name": task,
                "dataset_revision": f"revision-{task}",
                "mteb_version": "2.20.5",
                "scores": {
                    "test": [{"ndcg_at_10": ndcg, "recall_at_10": ndcg / 2}]
                },
            }
        ),
        encoding="utf-8",
    )


def test_mteb_run_aggregation_filters_tasks_and_uses_relative_manifest_paths(tmp_path):
    _write_result(tmp_path, "SciFact", 0.73)
    _write_result(tmp_path, "NFCorpus", 0.37)  # stale result from an earlier invocation

    run_log = [
        {"task": "SciFact", "status": "ok", "corpus": "mteb/scifact"},
        {"task": "NFCorpus", "status": "error", "error": "failed this run"},
        {"task": "QuoraRetrieval", "status": "skipped_large"},
    ]
    scores = _read_current_task_scores(tmp_path, run_log)

    assert list(scores) == ["SciFact"]
    assert scores["SciFact"]["ndcg@10"] == 0.73
    assert scores["SciFact"]["corpus"] == "mteb/scifact"
    assert scores["SciFact"]["split"] == "test"
    assert scores["SciFact"]["dataset_revision"] == "revision-SciFact"
    assert scores["SciFact"]["mteb_version"] == "2.20.5"
    assert scores["SciFact"]["path"] == (
        "results/Basinfy__BasinRAG/1.0.4/SciFact.json"
    )
    assert not Path(scores["SciFact"]["path"]).is_absolute()

    manifest = _build_run_manifest(
        ["SciFact", "NFCorpus", "QuoraRetrieval"], run_log, scores
    )
    assert manifest["summary_path"] == "arena_summary.json"
    assert manifest["complete"] is False
    assert manifest["requested_tasks"] == ["SciFact", "NFCorpus", "QuoraRetrieval"]
    assert manifest["completed_tasks"] == ["SciFact"]
    assert manifest["failed_tasks"] == ["NFCorpus"]
    assert manifest["skipped_tasks"] == ["QuoraRetrieval"]
    assert manifest["tasks"] == [
        {
            "task": "SciFact",
            "status": "ok",
            "corpus": "mteb/scifact",
            "expected_splits": None,
            "outcome": "completed",
            "result_path": "results/Basinfy__BasinRAG/1.0.4/SciFact.json",
            "split": "test",
            "dataset_revision": "revision-SciFact",
            "mteb_version": "2.20.5",
        },
        {
            "task": "NFCorpus",
            "status": "error",
            "error": "failed this run",
            "outcome": "failed",
            "result_path": None,
            "corpus": None,
            "split": None,
            "expected_splits": None,
            "dataset_revision": None,
            "mteb_version": None,
        },
        {
            "task": "QuoraRetrieval",
            "status": "skipped_large",
            "outcome": "skipped",
            "result_path": None,
            "corpus": None,
            "split": None,
            "expected_splits": None,
            "dataset_revision": None,
            "mteb_version": None,
        },
    ]


def test_mteb_partial_run_never_publishes_a_partial_mean(tmp_path):
    _write_result(tmp_path, "SciFact", 0.73)
    _write_result(tmp_path, "NFCorpus", 0.37)  # must not mask the current failure
    run_log = [
        {"task": "SciFact", "status": "ok"},
        {"task": "NFCorpus", "status": "error"},
    ]

    outcome = _summarize_run(
        ["SciFact", "NFCorpus"], run_log, _read_current_task_scores(tmp_path, run_log)
    )

    assert outcome["complete"] is False
    assert outcome["aggregate_status"] == "partial"
    assert outcome["completed_tasks"] == ["SciFact"]
    assert outcome["failed_tasks"] == ["NFCorpus"]
    assert outcome["mean_ndcg@10"] is None


def test_mteb_subset_is_complete_but_has_no_official_arena_mean(tmp_path):
    _write_result(tmp_path, "SciFact", 0.7)
    _write_result(tmp_path, "NFCorpus", 0.5)
    run_log = [
        {"task": "SciFact", "status": "ok"},
        {"task": "NFCorpus", "status": "ok"},
    ]

    outcome = _summarize_run(
        ["SciFact", "NFCorpus"], run_log, _read_current_task_scores(tmp_path, run_log)
    )

    assert outcome["complete"] is True
    assert outcome["aggregate_status"] == "complete"
    assert outcome["completed_tasks"] == ["SciFact", "NFCorpus"]
    assert outcome["failed_tasks"] == []
    assert outcome["skipped_tasks"] == []
    assert outcome["mean_ndcg@10"] is None


def test_mteb_provenance_contains_source_runtime_model_parameters_and_corpus(monkeypatch):
    monkeypatch.setattr(
        run_mteb,
        "_source_revision",
        lambda: {
            "commit": "abc123",
            "dirty": True,
            "revision": "abc123-dirty",
        },
    )
    monkeypatch.setattr(
        run_mteb, "_dependency_versions", lambda: {"mteb": "2.20.5", "torch": "2.4.1"}
    )
    rag = SimpleNamespace(
        ingestor=SimpleNamespace(
            splitter=SimpleNamespace(
                tokenizer=SimpleNamespace(
                    name_or_path="org/encoder-tokenizer",
                    init_kwargs={"_commit_hash": "tokrev"},
                )
            ),
            encoder=SimpleNamespace(
                _modules={
                    "0": SimpleNamespace(
                        auto_model=SimpleNamespace(
                            config=SimpleNamespace(_commit_hash="enc-rev")
                        )
                    )
                }
            ),
        )
    )
    model = _model_identity("org/encoder", rag)
    parameters = {
        "tasks": ["SciFact"],
        "max_queries": 50,
        "max_corpus_docs": 200_000,
        "force_reindex": False,
        "effective_rerank": None,
    }
    provenance = _build_provenance(
        parameters,
        ["SciFact"],
        {
            "SciFact": {
                "corpus": "mteb/scifact",
                "split": "test",
                "expected_splits": ["test"],
                "dataset_revision": "dataset-rev",
                "mteb_version": "2.20.5",
            }
        },
        model,
    )

    assert provenance["source"] == {
        "commit": "abc123",
        "dirty": True,
        "revision": "abc123-dirty",
    }
    assert provenance["python"]["version"]
    assert provenance["python"]["implementation"]
    assert provenance["dependencies"] == {"mteb": "2.20.5", "torch": "2.4.1"}
    assert provenance["package"]["source_version"] == run_mteb._BASINRAG_SOURCE_VERSION
    assert provenance["package"]["distribution_metadata_version"] is None
    assert provenance["model"] == {
        "encoder": {"name": "org/encoder", "revision": "enc-rev"},
        "tokenizer": {"name": "org/encoder-tokenizer", "revision": "tokrev"},
    }
    assert provenance["parameters"] == parameters
    assert provenance["corpus_splits"]["SciFact"] == {
        "corpus": "mteb/scifact",
        "split": "test",
        "expected_splits": ["test"],
        "dataset_revision": "dataset-rev",
        "mteb_version": "2.20.5",
    }


def test_mteb_main_writes_provenance_and_partial_summary_without_stale_scores(
    monkeypatch, tmp_path, capsys
):
    def fake_run_one_task(task_name, **kwargs):
        _write_result(kwargs["output_dir"], task_name, 0.9)
        if task_name == "SciFact":
            return {
                "task": task_name,
                "status": "ok",
                "corpus": "mteb/scifact",
                "expected_splits": ["test"],
                "model": _model_identity(kwargs["encoder_model"]),
                "effective_ranking": {
                    "ranking_mode": kwargs["ranking_mode"],
                    "use_hop_prior": kwargs["use_hop_prior"],
                    "use_multi_signal_drf": (
                        kwargs["ranking_mode"] == "experimental_topology"
                    ),
                    "expand_graph": False,
                },
            }
        return {
            "task": task_name,
            "status": "error",
            "error": "simulated evaluation failure",
            "corpus": "mteb/nfcorpus",
            "expected_splits": ["test"],
            "model": _model_identity(kwargs["encoder_model"]),
            "effective_ranking": {
                "ranking_mode": kwargs["ranking_mode"],
                "use_hop_prior": kwargs["use_hop_prior"],
                "use_multi_signal_drf": (
                    kwargs["ranking_mode"] == "experimental_topology"
                ),
                "expand_graph": False,
            },
        }

    monkeypatch.setattr(sys, "argv", [
        "run_mteb", "--tasks", "SciFact,NFCorpus", "--output", str(tmp_path)
    ])
    monkeypatch.setattr(run_mteb, "run_one_task", fake_run_one_task)

    run_mteb.main()
    capsys.readouterr()
    summary = json.loads((tmp_path / "arena_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "arena_manifest.json").read_text(encoding="utf-8"))

    assert summary["complete"] is False
    assert summary["aggregate_status"] == "partial"
    assert summary["requested_tasks"] == ["SciFact", "NFCorpus"]
    assert summary["completed_tasks"] == ["SciFact"]
    assert summary["failed_tasks"] == ["NFCorpus"]
    assert summary["mean_ndcg@10"] is None
    assert summary["scores"].keys() == {"SciFact"}
    assert summary["manifest_path"] == "arena_manifest.json"
    assert summary["parameters"]["tasks"] == ["SciFact", "NFCorpus"]
    assert summary["parameters"]["max_corpus_docs"] is None
    assert summary["parameters"]["ranking_mode"] == "hybrid_rrf"
    assert summary["parameters"]["use_hop_prior"] is False
    assert summary["parameters"]["effective_ranking_by_task"]["SciFact"] == {
        "ranking_mode": "hybrid_rrf",
        "use_hop_prior": False,
        "use_multi_signal_drf": False,
        "expand_graph": False,
    }
    assert summary["provenance"]["source"]["revision"]
    assert summary["provenance"]["python"]["version"]
    assert "mteb" in summary["provenance"]["dependencies"]
    assert summary["provenance"]["model"]["encoder"]["name"] == "BAAI/bge-base-en-v1.5"
    assert summary["provenance"]["corpus_splits"]["SciFact"]["split"] == "test"
    assert manifest["complete"] is False
    assert "mean_ndcg@10" not in manifest
    assert manifest["provenance"] == summary["provenance"]
    assert manifest["parameters"] == summary["parameters"]
    assert manifest["tasks"][1]["corpus"] == "mteb/nfcorpus"


def test_mteb_wrapper_ranking_defaults_to_topology_free_rrf():
    wrapper = BasinRAGMTEBWrapper(SimpleNamespace())

    assert wrapper.effective_ranking_parameters == {
        "ranking_mode": "hybrid_rrf",
        "use_hop_prior": False,
        "use_multi_signal_drf": False,
        "expand_graph": False,
    }


def test_mteb_wrapper_experimental_topology_options_are_explicit():
    wrapper = BasinRAGMTEBWrapper(
        SimpleNamespace(), ranking_mode="experimental_topology", use_hop_prior=True
    )
    assert wrapper.effective_ranking_parameters == {
        "ranking_mode": "experimental_topology",
        "use_hop_prior": True,
        "use_multi_signal_drf": True,
        "expand_graph": True,
    }

    wrapper._is_flat_index = True
    assert wrapper.effective_ranking_parameters["expand_graph"] is False
