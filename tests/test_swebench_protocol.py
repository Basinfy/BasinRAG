from basinrag.eval.swebench import SWEBenchInstance, run_official_eval
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
