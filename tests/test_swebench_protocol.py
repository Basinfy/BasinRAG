import json
from types import SimpleNamespace

import pytest

from basinrag.eval.swebench import SWEBenchInstance, run_official_eval
from basinrag.eval import swebench as swebench_module
from basinrag.eval.swebench_protocol import (
    OfficialReport,
    build_eval_command,
    dataset_alias,
    drop_leading_readme,
    extract_model_patch,
    gold_files_from_patch,
    hit_at_k,
    is_official_test_path,
    official_retrieval_scores,
    pack_files_to_budget,
    prediction_record,
    resolve_dataset,
    write_predictions_jsonl,
)
from basinrag.eval.swebench import (
    SWEBenchEvaluator,
    _checkpoint_protocol_id,
    _checkpoint_subset_id,
    _matches_gold_path,
)


def test_dataset_aliases_match_official_harness():
    assert resolve_dataset("lite") == "SWE-bench/SWE-bench_Lite"
    assert resolve_dataset("verified") == "SWE-bench/SWE-bench_Verified"
    assert resolve_dataset("full") == "SWE-bench/SWE-bench"
    assert resolve_dataset("SWE-bench/SWE-bench_Lite") == "SWE-bench/SWE-bench_Lite"
    assert dataset_alias("SWE-bench/SWE-bench_Verified") == "verified"


def test_gold_files_follow_official_eval_retrieval_regex():
    patch = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/dev/null b/src/created.py\n"
        "--- a/dev/null\n"
        "+++ b/src/created.py\n"
    )
    assert gold_files_from_patch(patch) == {"src/foo.py"}
    assert SWEBenchInstance.extract_patch_files(patch) == {"src/foo.py"}


def test_official_test_path_filter():
    assert is_official_test_path("tests/test_core.py")
    assert is_official_test_path("django/test/utils.py")
    assert is_official_test_path("pkg/test_utils.py")
    assert not is_official_test_path("pkg/contest.py")
    assert not is_official_test_path("src/core.py")


def test_official_retrieval_metrics_and_readme_skip():
    gold = {"src/core.py", "src/util.py"}
    retrieved = ["README.md", "src/core.py", "docs/guide.py"]
    scores = official_retrieval_scores(retrieved, gold)
    assert scores["recall"] == 0.5
    assert scores["any_recall"] == 1.0
    assert scores["all_recall"] == 0.0
    assert drop_leading_readme(["README.rst", "a.py"]) == ["a.py"]


def test_hit_at_k_is_exact_path_only():
    gold = {"pkg/utils.py"}
    assert hit_at_k(["pkg/utils.py"], gold, 1)
    assert not hit_at_k(["utils.py"], gold, 1)
    assert not hit_at_k(["src/pkg/utils.py"], gold, 5)


def test_pack_stops_at_token_budget():
    ranked = ["a.py", "b.py", "c.py"]
    contents = {"a.py": "one two", "b.py": "three four five six", "c.py": "seven"}
    packed = pack_files_to_budget(ranked, contents, token_budget=5)
    assert packed == ["a.py"]


def test_extract_model_patch_strips_fences():
    raw = "Sure.\n```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n```\n"
    patch = extract_model_patch(raw)
    assert patch.startswith("diff --git a/a.py")
    assert extract_model_patch("no patch here") == ""


def test_prediction_jsonl_has_official_fields(tmp_path):
    rec = prediction_record("django__django-123", "basinrag+ollama/qwen2.5", "diff --git a/a.py b/a.py\n")
    path = write_predictions_jsonl(tmp_path / "preds.jsonl", [rec])
    line = path.read_text(encoding="utf-8").strip()
    assert '"instance_id": "django__django-123"' in line
    assert '"model_name_or_path":' in line
    assert '"model_patch":' in line


def test_eval_command_is_official_v5_cli():
    gold = build_eval_command("lite", gold=True, run_id="validate-gold", instance_ids=["sympy__sympy-20590"])
    assert gold[1:4] == ["eval", "SWE-bench/SWE-bench_Lite", "--run-id"]
    assert "--gold" in gold
    assert gold[gold.index("-i") + 1] == "sympy__sympy-20590"

    pred = build_eval_command("verified", predictions="preds.jsonl", run_id="basinrag-1", workers=8)
    assert pred[1:3] == ["eval", "SWE-bench/SWE-bench_Verified"]
    assert pred[pred.index("-p") + 1] == "preds.jsonl"
    assert pred[pred.index("-j") + 1] == "8"


def test_eval_command_rejects_gold_and_predictions_together():
    try:
        build_eval_command("lite", gold=True, predictions="x.jsonl", run_id="r")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_official_eval_refuses_without_docker(monkeypatch):
    monkeypatch.setattr("basinrag.eval.swebench.docker_available", lambda: False)
    try:
        run_official_eval("lite", gold=True, run_id="no-docker")
    except RuntimeError as exc:
        assert "Docker" in str(exc)
        return
    raise AssertionError("expected RuntimeError")


def test_official_report_uses_dataset_n_not_submitted():
    report = OfficialReport.from_harness(
        {
            "total_instances": 300,
            "submitted_instances": 20,
            "resolved_instances": 10,
        },
        dataset="lite",
        run_id="r1",
    )
    assert report.n_resolved == 10
    assert report.resolved_rate == 10 / 300


def _swebench_instance(instance_id, gold_file):
    return SWEBenchInstance(
        instance_id=instance_id,
        repo="psf/requests",
        base_commit="abc123",
        problem_statement="Handle this request correctly.",
        patch="",
        test_patch="",
        fail_to_pass=[],
        pass_to_pass=[],
        version="2.0",
        ground_truth_files={gold_file},
    )


def _checkpoint_kwargs():
    return {
        "split": "test",
        "top_k": 10,
        "search_type": "hybrid",
        "encoder_model": "BAAI/bge-base-en-v1.5",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "use_rerank": True,
        "query_prompt": "Represent this sentence for searching relevant passages: ",
        "ranking_mode": "hybrid_rrf",
    }


def test_checkpoint_reuses_only_matching_subset_and_protocol(tmp_path, monkeypatch):
    selected = _swebench_instance("requests__requests-1", "src/requests.py")
    subset_id = _checkpoint_subset_id([selected])
    protocol_id = _checkpoint_protocol_id(**_checkpoint_kwargs())

    matching = {
        "instance_id": selected.instance_id,
        "subset_id": subset_id,
        "protocol_id": protocol_id,
        "hit@1": True,
        "hit@5": True,
        "hit@10": True,
        "recall": 1.0,
        "mrr": 1.0,
    }
    outside_subset = {**matching, "instance_id": "requests__requests-2"}
    other_protocol = {**matching, "protocol_id": "different-protocol"}
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(json.dumps(row) for row in (matching, outside_subset, other_protocol)) + "\n",
        encoding="utf-8",
    )

    evaluator = SWEBenchEvaluator(workspace_root=str(tmp_path / "workspace"))
    monkeypatch.setattr(
        swebench_module.BasinRAG,
        "create",
        staticmethod(lambda **kwargs: pytest.fail("matching checkpoint should avoid rebuilding")),
    )
    metrics = evaluator.evaluate_retrieval(
        [selected], checkpoint_path=str(checkpoint), **_checkpoint_kwargs()
    )

    assert metrics["instances_evaluated"] == 1
    assert metrics["hit@1"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert [row["instance_id"] for row in metrics["details"]] == [selected.instance_id]


def test_checkpoint_protocol_changes_with_retrieval_settings():
    default = _checkpoint_protocol_id(**_checkpoint_kwargs())
    variants = (
        {"top_k": 5},
        {"candidate_k": 30},
        {"ranking_mode": "experimental_topology"},
        {"chunk_size": 900},
        {"chunk_overlap": 100},
        {"chunker_revision": "recursive-character-v2"},
        {"tokenizer_policy": "pinned-tokenizer-revision-v2"},
    )
    for changes in variants:
        changed = _checkpoint_protocol_id(**{**_checkpoint_kwargs(), **changes})
        assert default != changed, changes


def test_legacy_checkpoint_protocol_is_recomputed(tmp_path, monkeypatch):
    import hashlib

    selected = _swebench_instance("requests__requests-legacy", "src/requests.py")
    legacy_kwargs = _checkpoint_kwargs()
    legacy_protocol = {
        key: legacy_kwargs[key]
        for key in (
            "split",
            "top_k",
            "search_type",
            "encoder_model",
            "reranker_model",
            "use_rerank",
            "query_prompt",
        )
    }
    legacy_protocol.update(
        dataset=swebench_module.DATASET_ID,
        metric_version=swebench_module.CHECKPOINT_METRIC_VERSION,
    )
    legacy_canonical = json.dumps(
        legacy_protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    legacy_protocol_id = hashlib.sha256(legacy_canonical.encode("utf-8")).hexdigest()
    assert _checkpoint_protocol_id(**_checkpoint_kwargs()) != legacy_protocol_id

    checkpoint = tmp_path / "legacy-checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "instance_id": selected.instance_id,
                "subset_id": _checkpoint_subset_id([selected]),
                "protocol_id": legacy_protocol_id,
                "hit@1": False,
                "hit@5": False,
                "hit@10": False,
                "recall": 0.0,
                "mrr": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rag = SimpleNamespace(
        query=lambda *args, **kwargs: [SimpleNamespace(metadata={"file_path": "src/requests.py"})]
    )
    created_with_modes = []
    evaluator = SWEBenchEvaluator(workspace_root=str(tmp_path / "workspace"))
    evaluator.workspace.prepare_repo = lambda repo, commit: tmp_path
    monkeypatch.setattr(
        swebench_module.BasinRAG,
        "create",
        staticmethod(lambda **kwargs: (created_with_modes.append(kwargs["ranking_mode"]) or rag)),
    )
    monkeypatch.setattr(
        swebench_module.CodeRepoIngestor,
        "ingest_codebase",
        staticmethod(lambda rag, repo_path: 1),
    )

    metrics = evaluator.evaluate_retrieval(
        [selected], checkpoint_path=str(checkpoint), **_checkpoint_kwargs()
    )

    assert created_with_modes == ["hybrid_rrf"]
    assert metrics["hit@1"] == 1.0
    refreshed = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    assert len(refreshed) == 2
    assert refreshed[-1]["protocol_id"] != legacy_protocol_id
    assert refreshed[-1]["protocol_revision"] == swebench_module.CHECKPOINT_PROTOCOL_REVISION
    assert refreshed[-1]["ranking_mode"] == "hybrid_rrf"


def test_evaluator_requires_exact_normalized_file_paths(tmp_path, monkeypatch):
    instance = _swebench_instance("requests__requests-3", "src/utils.py")
    fake_rag = SimpleNamespace(
        query=lambda *args, **kwargs: [
            SimpleNamespace(metadata={"file_path": "utils.py"})
        ]
    )
    evaluator = SWEBenchEvaluator(workspace_root=str(tmp_path / "workspace"))
    evaluator.workspace.prepare_repo = lambda repo, commit: tmp_path
    monkeypatch.setattr(
        swebench_module.BasinRAG,
        "create",
        staticmethod(lambda **kwargs: fake_rag),
    )
    monkeypatch.setattr(
        swebench_module.CodeRepoIngestor,
        "ingest_codebase",
        staticmethod(lambda rag, repo_path: 1),
    )

    metrics = evaluator.evaluate_retrieval(
        [instance], checkpoint_path=None, **_checkpoint_kwargs()
    )

    assert metrics["instances_evaluated"] == 1
    assert metrics["hit@1"] == metrics["hit@5"] == metrics["hit@10"] == 0.0
    assert metrics["recall@10"] == metrics["mrr"] == 0.0
    assert _matches_gold_path("src\\utils.py", {"src/utils.py"})
    assert not _matches_gold_path("utils.py", {"src/utils.py"})
