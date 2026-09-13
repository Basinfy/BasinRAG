"""
Official SWE-bench protocol constants and helpers.

Source of truth:
  https://github.com/SWE-bench/SWE-bench  (harness v5)
  https://www.swebench.com/

Two tracks that must never be mixed:

  resolve  — % Resolved via Docker harness (the public leaderboard metric)
  retrieve — Avg / Any / All Recall from swebench.inference.make_datasets.eval_retrieval

BasinRAG is a retriever. A Hit@k number is not a SWE-bench leaderboard score.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Official CLI aliases plus Lite (CLI v5 passes unknown names through).
# https://github.com/SWE-bench/SWE-bench/blob/main/swebench/cli/_datasets.py
DATASET_ALIASES: Dict[str, str] = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
    "multimodal": "SWE-bench/SWE-bench_Multimodal",
}

# Split `test` counts from the HuggingFace parquet (2026-09-12).
DATASET_TEST_N: Dict[str, int] = {
    "lite": 300,
    "verified": 500,
    "full": 2294,
    "multilingual": 300,
    "multimodal": 480,
}

LEADERBOARD_METRIC = "resolved_rate"
RETRIEVAL_METRICS = ("avg_recall", "any_recall", "all_recall")

# Official retrieval gold files: paths on the `--- a/` side of the gold patch.
# swebench/inference/make_datasets/eval_retrieval.py
GOLD_PATCH_FILE_RE = re.compile(r"^--- a/(.+)$", re.MULTILINE)

# Official RAG corpus: *.py and drop test paths (include_tests=False).
# swebench/inference/make_datasets/utils.py :: is_test / list_files
TEST_PATH_WORDS = frozenset({"test", "tests", "testing"})
TEST_PATH_SPLIT_RE = re.compile(r"[ _/.]")

# Official eval_retrieval drops a leading README hit.
README_NAME_RE = re.compile(r"readme", re.IGNORECASE)

PREDICTION_FIELDS = ("instance_id", "model_name_or_path", "model_patch")

# Paper RAG context budgets (tokenized files packed until this cap).
TOKEN_BUDGETS = (13_000, 27_000, 40_000)
DEFAULT_TOKEN_BUDGET = 27_000


def resolve_dataset(name: str) -> str:
    """Expand lite/verified/full/...; pass HuggingFace ids and paths through."""
    key = (name or "").strip()
    if not key:
        raise ValueError("dataset name is empty")
    return DATASET_ALIASES.get(key.lower(), key)


def dataset_alias(name: str) -> str:
    """Return the short alias when known, else the resolved HuggingFace id."""
    resolved = resolve_dataset(name)
    for alias, hf_id in DATASET_ALIASES.items():
        if hf_id == resolved or alias == name.strip().lower():
            return alias
    return resolved


def is_official_test_path(relative_path: str) -> bool:
    """Match swebench.inference.make_datasets.utils.is_test."""
    words = set(TEST_PATH_SPLIT_RE.split(relative_path.lower()))
    return any(word in TEST_PATH_WORDS for word in words)


def gold_files_from_patch(patch: str) -> Set[str]:
    """Official retrieval gold set: `--- a/<path>` minus /dev/null."""
    files: Set[str] = set()
    if not patch:
        return files
    for raw in GOLD_PATCH_FILE_RE.findall(patch):
        path = raw.strip()
        if not path or path in {"/dev/null", "dev/null"}:
            continue
        if path.startswith("a/"):
            path = path[2:]
        files.add(path)
    return files


def drop_leading_readme(retrieved: Sequence[str]) -> List[str]:
    """Official eval_retrieval skips the first hit when it is a README."""
    files = list(retrieved)
    if files and README_NAME_RE.search(files[0]):
        return files[1:]
    return files


def official_retrieval_scores(
    retrieved: Sequence[str],
    gold_files: Set[str],
) -> Dict[str, float]:
    """
    Metrics from swebench.inference.make_datasets.eval_retrieval:

      recall      = |retrieved ∩ gold| / |gold|
      any_recall  = 1 if recall > 0 else 0
      all_recall  = 1 if recall == 1 else 0
    """
    if not gold_files:
        return {"recall": 0.0, "any_recall": 0.0, "all_recall": 0.0}
    retrieved_set = set(drop_leading_readme(retrieved))
    recall = len(retrieved_set & gold_files) / float(len(gold_files))
    return {
        "recall": recall,
        "any_recall": 1.0 if recall > 0 else 0.0,
        "all_recall": 1.0 if recall == 1.0 else 0.0,
    }


def hit_at_k(retrieved: Sequence[str], gold_files: Set[str], k: int) -> bool:
    """Community file-localization Hit@k. Exact path match only."""
    if not gold_files:
        return False
    return any(path in gold_files for path in retrieved[:k])


def mrr_at_k(retrieved: Sequence[str], gold_files: Set[str], k: int) -> float:
    for rank, path in enumerate(retrieved[:k], 1):
        if path in gold_files:
            return 1.0 / rank
    return 0.0


def estimate_tokens(text: str) -> int:
    """Whitespace token count when the official tokenizer is unavailable."""
    return len(text.split()) if text else 0


def pack_files_to_budget(
    ranked_files: Sequence[str],
    file_contents: Dict[str, str],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> List[str]:
    """Keep official retrieval order; stop when the packed context would overflow."""
    packed: List[str] = []
    used = 0
    for path in ranked_files:
        content = file_contents.get(path)
        if content is None:
            continue
        cost = estimate_tokens(f"{path}\n{content}")
        if packed and used + cost > token_budget:
            break
        packed.append(path)
        used += cost
    return packed


def extract_model_patch(raw: str) -> str:
    """Strip chat wrappers and keep a git-apply-able diff if one is present."""
    text = (raw or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for marker in ("diff --git ", "--- a/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:].strip() + ("\n" if not text.endswith("\n") else "")
    return ""


def prediction_record(
    instance_id: str,
    model_name_or_path: str,
    model_patch: str,
) -> Dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
    }


def write_predictions_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            missing = [field for field in PREDICTION_FIELDS if field not in record]
            if missing:
                raise ValueError(f"prediction missing {missing}: {record!r}")
            handle.write(
                json.dumps(
                    {
                        "instance_id": record["instance_id"],
                        "model_name_or_path": record["model_name_or_path"],
                        "model_patch": record.get("model_patch") or "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out


def new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def swebench_cli() -> Optional[str]:
    return shutil.which("swebench")


def build_eval_command(
    dataset: str,
    *,
    predictions: Optional[str] = None,
    gold: bool = False,
    run_id: str,
    workers: int = 4,
    instance_ids: Optional[Sequence[str]] = None,
    timeout: int = 1800,
    modal: bool = False,
) -> List[str]:
    """Exact SWE-bench v5 CLI. Never reuse run_id across different patches."""
    if gold == bool(predictions):
        raise ValueError("pass exactly one of gold=True or predictions=<path>")
    exe = swebench_cli() or "swebench"
    cmd = [exe, "eval", resolve_dataset(dataset), "--run-id", run_id, "-j", str(workers)]
    if gold:
        cmd.append("--gold")
    else:
        cmd.extend(["-p", str(predictions)])
    if instance_ids:
        for instance_id in instance_ids:
            cmd.extend(["-i", instance_id])
    if timeout != 1800:
        cmd.extend(["-t", str(timeout)])
    if modal:
        cmd.append("--modal")
    return cmd


@dataclass(frozen=True)
class OfficialReport:
    dataset: str
    split: str
    n_dataset: int
    n_submitted: int
    n_resolved: int
    resolved_rate: float
    run_id: str
    results_path: Optional[str] = None

    @classmethod
    def from_harness(cls, payload: Dict[str, Any], *, dataset: str, run_id: str) -> "OfficialReport":
        total = int(payload.get("total_instances") or payload.get("Total instances") or 0)
        submitted = int(payload.get("submitted_instances") or payload.get("Instances submitted") or 0)
        resolved = int(payload.get("resolved_instances") or payload.get("Instances resolved") or 0)
        # Leaderboard: resolved / N of the split, not resolved / submitted.
        denom = total or submitted or 1
        return cls(
            dataset=dataset,
            split=str(payload.get("split") or "test"),
            n_dataset=total,
            n_submitted=submitted,
            n_resolved=resolved,
            resolved_rate=resolved / float(denom),
            run_id=run_id,
        )
