"""
Runner oficial do MTEB para BasinRAG.
Executa tarefas de Retrieval padronizadas e gera artefatos no formato MTEB.

Cada task usa storage isolado (evita contaminação de corpus entre SciFact/NFCorpus/…).
Corpora acima de --max-corpus-docs são pulados (Wikipedia-scale estoura RAM/tempo).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .. import __version__ as _BASINRAG_SOURCE_VERSION

# Patch PyTorch DTensor antes de carregar mteb
try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

from ..logging_config import setup_logging

logger = setup_logging()

# BEIR-EN arena padrão (média nDCG@10 comparável ao board de Retrieval).
DEFAULT_BEIR_ARENA = (
    "SciFact,NFCorpus,FiQA2018,ArguAna,QuoraRetrieval,"
    "TRECCOVID,Touche2020,ClimateFEVER,DBPedia,FEVER,HotpotQA,NQ"
)

_PROVENANCE_DISTRIBUTIONS = (
    "basinrag",
    "mteb",
    "torch",
    "sentence-transformers",
    "transformers",
    "numpy",
    "faiss-cpu",
    "datasets",
    "beir",
    "langchain",
    "langchain-community",
    "langchain-core",
    "networkx",
    "pypdf",
    "langchain-text-splitters",
    "filelock",
    "python-dotenv",
)


def _source_revision() -> Dict[str, Any]:
    """Return the repository revision and whether the evaluated tree is dirty."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        dirty = bool(status.strip())
        return {
            "commit": head,
            "dirty": dirty,
            "revision": f"{head}-dirty" if dirty else head,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "revision": "unknown"}


def _dependency_versions() -> Dict[str, Optional[str]]:
    """Read installed distributions without importing optional evaluation stacks."""
    versions: Dict[str, Optional[str]] = {}
    for distribution in _PROVENANCE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_identity(encoder_model: str, rag: Any = None) -> Dict[str, Any]:
    """Describe the requested encoder and the tokenizer actually selected by it."""
    tokenizer = None
    encoder_revision = None
    tokenizer_revision = None
    if rag is not None:
        ingestor = getattr(rag, "ingestor", None)
        splitter = getattr(ingestor, "splitter", None)
        tokenizer = getattr(splitter, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(getattr(ingestor, "encoder", None), "tokenizer", None)
        tokenizer_revision = getattr(tokenizer, "revision", None)
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            tokenizer_revision = tokenizer_revision or init_kwargs.get("_commit_hash")

        encoder = getattr(ingestor, "encoder", None)
        modules = getattr(encoder, "_modules", {})
        module_values = modules.values() if isinstance(modules, dict) else []
        for module in module_values:
            auto_model = getattr(module, "auto_model", None)
            config = getattr(auto_model, "config", None)
            encoder_revision = getattr(config, "_commit_hash", None)
            if encoder_revision:
                break

    tokenizer_name = getattr(tokenizer, "name_or_path", None) or encoder_model
    return {
        "encoder": {"name": encoder_model, "revision": encoder_revision},
        "tokenizer": {"name": tokenizer_name, "revision": tokenizer_revision},
    }


def _task_metadata(task: Any, fallback_name: str) -> Dict[str, Any]:
    """Extract compact corpus/split metadata from an MTEB task when available."""
    metadata = getattr(task, "metadata", None)
    dataset = getattr(metadata, "dataset", None)
    if isinstance(dataset, dict):
        corpus = dataset.get("path") or dataset.get("name") or fallback_name
    elif isinstance(dataset, str) and dataset:
        corpus = dataset
    else:
        corpus = fallback_name
    eval_splits = getattr(metadata, "eval_splits", None)
    if isinstance(eval_splits, str):
        eval_splits = [eval_splits]
    elif eval_splits is not None:
        try:
            eval_splits = list(eval_splits)
        except TypeError:
            eval_splits = None
    return {"corpus": corpus, "expected_splits": eval_splits}


def _run_outcome(
    task_names: List[str], run_log: List[Dict[str, Any]], scores: Dict[str, Any]
) -> Dict[str, Any]:
    """Classify outcomes and only call the run complete when every score exists."""
    completed: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []
    task_rows: List[Dict[str, Any]] = []
    for row in run_log:
        task = str(row.get("task") or "")
        score = scores.get(task)
        metric = score.get("ndcg@10") if isinstance(score, dict) else None
        try:
            has_metric = metric is not None and math.isfinite(float(metric))
        except (TypeError, ValueError):
            has_metric = False
        status = row.get("status")
        if status == "ok" and has_metric:
            outcome = "completed"
            completed.append(task)
        elif isinstance(status, str) and status.startswith("skipped"):
            outcome = "skipped"
            skipped.append(task)
        else:
            outcome = "failed"
            failed.append(task)
        task_rows.append({**row, "outcome": outcome})

    complete = (
        bool(task_names)
        and len(run_log) == len(task_names)
        and len(completed) == len(task_names)
        and not skipped
        and not failed
    )
    return {
        "requested_tasks": list(task_names),
        "completed_tasks": completed,
        "skipped_tasks": skipped,
        "failed_tasks": failed,
        "complete": complete,
        "aggregate_status": "complete" if complete else (
            "partial" if completed or skipped else "failed"
        ),
        "task_rows": task_rows,
    }


def _summarize_run(
    task_names: List[str], run_log: List[Dict[str, Any]], scores: Dict[str, Any], *, sampled: bool = False
) -> Dict[str, Any]:
    """Expose a mean only for a complete run; partial means are not comparable."""
    outcome = _run_outcome(task_names, run_log, scores)
    task_ndcgs = [
        float(scores[task]["ndcg@10"])
        for task in outcome["completed_tasks"]
        if task in scores and scores[task].get("ndcg@10") is not None
    ]
    official_arena = set(task_names) == set(DEFAULT_BEIR_ARENA.split(","))
    if sampled:
        outcome["complete"] = False
        outcome["aggregate_status"] = "sampled"
    outcome["sampled"] = sampled
    outcome["mean_ndcg@10"] = (
        sum(task_ndcgs) / len(task_ndcgs)
        if outcome["complete"] and official_arena and not sampled and len(task_ndcgs) == len(task_names)
        else None
    )
    return outcome


def _build_provenance(
    parameters: Dict[str, Any],
    task_names: List[str],
    task_scores: Dict[str, Any],
    model_identity: Dict[str, Any],
    run_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Capture source, runtime, model, corpus, and split identity for one run."""
    run_log = run_log or []
    task_metadata = {str(row.get("task")): row for row in run_log}
    dependencies = _dependency_versions()
    return {
        "source": _source_revision(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "dependencies": dependencies,
        "package": {
            "source_version": _BASINRAG_SOURCE_VERSION,
            "distribution_metadata_version": dependencies.get("basinrag"),
        },
        "model": model_identity,
        "parameters": parameters,
        "corpus_splits": {
            task: {
                "corpus": (
                    task_scores.get(task, {}).get("corpus")
                    or task_metadata.get(task, {}).get("corpus")
                    or task
                ),
                "split": (
                    task_scores.get(task, {}).get("split")
                    or task_metadata.get(task, {}).get("split")
                ),
                "expected_splits": (
                    task_scores.get(task, {}).get("expected_splits")
                    or task_metadata.get(task, {}).get("expected_splits")
                ),
                "dataset_revision": task_scores.get(task, {}).get("dataset_revision"),
                "mteb_version": task_scores.get(task, {}).get("mteb_version"),
            }
            for task in task_names
        },
    }


def _encoder_tag(encoder_model: str) -> str:
    return encoder_model.replace("/", "_").replace("-", "_")


def _read_task_main_score(
    output_dir: Path,
    model_name: str = "Basinfy__BasinRAG",
    *,
    task_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Read scores for selected tasks, keeping result paths relative to the run directory."""
    out: Dict[str, Any] = {}
    output_root = output_dir.resolve()
    root = output_root / "results" / model_name
    if not root.exists():
        return out
    allowed_tasks: Optional[Set[str]] = set(task_names) if task_names is not None else None
    for path in root.rglob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task = payload.get("task_name") or path.stem
        if allowed_tasks is not None and task not in allowed_tasks:
            continue
        scores = payload.get("scores") or {}
        split_name = next(
            (name for name in ("test", "dev") if scores.get(name)),
            None,
        )
        split_scores = scores.get(split_name) if split_name else None
        if not split_scores:
            continue
        row = split_scores[0] if isinstance(split_scores, list) else split_scores
        metric_fields = {
            "ndcg@10": row.get("ndcg_at_10"),
            "recall@10": row.get("recall_at_10"),
            "mrr@10": row.get("mrr_at_10"),
            "map@10": row.get("map_at_10"),
            "precision@10": row.get("precision_at_10"),
            "main_score": row.get("main_score"),
            "main_score_name": row.get("main_score_name") or payload.get("main_score_name"),
        }
        out[task] = {
            **{key: value for key, value in metric_fields.items() if value is not None},
            "corpus": payload.get("corpus") or payload.get("dataset_name") or task,
            "split": split_name,
            "dataset_revision": payload.get("dataset_revision"),
            "mteb_version": payload.get("mteb_version"),
            "path": path.relative_to(output_root).as_posix(),
        }
    return out


def _read_current_task_scores(output_dir: Path, run_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate only successful tasks requested and run in this invocation."""
    completed_tasks = {
        str(row["task"])
        for row in run_log
        if row.get("status") == "ok" and row.get("task")
    }
    scores = _read_task_main_score(output_dir, task_names=completed_tasks)
    for row in run_log:
        task = row.get("task")
        if task in scores:
            if row.get("corpus"):
                scores[task]["corpus"] = row["corpus"]
            if row.get("expected_splits"):
                scores[task]["expected_splits"] = row["expected_splits"]
    return scores


def _build_run_manifest(
    task_names: List[str],
    run_log: List[Dict[str, Any]],
    scores: Dict[str, Any],
    *,
    parameters: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    sampled: bool = False,
) -> Dict[str, Any]:
    """Describe this invocation only; every artifact reference is relative."""
    outcome = _run_outcome(task_names, run_log, scores)
    if sampled:
        outcome["complete"] = False
        outcome["aggregate_status"] = "sampled"
    return {
        **{key: outcome[key] for key in (
            "requested_tasks", "completed_tasks", "skipped_tasks", "failed_tasks",
            "complete", "aggregate_status",
        )},
        "sampled": sampled,
        "tasks": [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"path", "model"}
                },
                "status": row.get("status"),
                "result_path": scores.get(row.get("task"), {}).get("path"),
                **{
                    key: scores.get(row.get("task"), {}).get(key)
                    or row.get(key)
                    for key in (
                        "corpus", "split", "expected_splits", "dataset_revision", "mteb_version"
                    )
                },
            }
            for row in outcome["task_rows"]
        ],
        "parameters": parameters or {},
        "provenance": provenance or {},
        "summary_path": "arena_summary.json",
    }


def run_one_task(
    task_name: str,
    *,
    encoder_model: str,
    encoder_revision: Optional[str] = None,
    reranker_model: str,
    reranker_revision: Optional[str] = None,
    search_type: str,
    rerank_top_k: int,
    prompt: str,
    rerank_flag: Optional[bool],
    force_reindex: bool,
    max_queries: Optional[int],
    max_corpus_docs: Optional[int],
    output_dir: Path,
    ranking_mode: str = "hybrid_rrf",
    use_hop_prior: bool = False,
) -> Dict[str, Any]:
    import mteb
    from mteb.cache import ResultCache

    from ..factory import BasinRAG
    from .mteb_wrapper import BasinRAGMTEBWrapper, SkipLargeCorpus

    cache_tag = _encoder_tag(encoder_model)
    task_tag = task_name.replace("/", "_")
    storage_dir = output_dir / "indices" / f"mteb_cache_{cache_tag}_{task_tag}"

    print(f"\n{'='*70}")
    print(f"  TASK: {task_name}")
    print(f"  storage: {storage_dir}")
    print(f"{'='*70}\n")

    rag = BasinRAG.create(
        storage_dir=storage_dir,
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        reranker_model=reranker_model,
        reranker_revision=reranker_revision,
        use_rerank=bool(rerank_flag) if rerank_flag is not None else False,
        ranking_mode=ranking_mode,
        query_prompt=prompt,
    )
    model_identity = _model_identity(encoder_model, rag)
    wrapper = BasinRAGMTEBWrapper(
        rag,
        search_type=search_type,
        rerank_top_k=rerank_top_k,
        query_prompt=prompt,
        force_reindex=force_reindex,
        ranking_mode=ranking_mode,
        use_hop_prior=use_hop_prior,
        cache_tag=f"{task_tag}|{encoder_model}|flat",
        use_rerank=rerank_flag,
        max_corpus_docs=max_corpus_docs,
    )
    if max_queries:
        wrapper.max_queries = max_queries

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        return {
            "task": task_name,
            "status": "missing",
            "error": "task not found in mteb",
            "model": model_identity,
            "effective_ranking": wrapper.effective_ranking_parameters,
        }
    task_identity = _task_metadata(tasks[0], task_name)

    cache = ResultCache(cache_path=output_dir)
    try:
        mteb.evaluate(
            model=wrapper,
            tasks=tasks,
            cache=cache,
            overwrite_strategy="always",
            prediction_folder=output_dir,
            show_progress_bar=True,
        )
    except SkipLargeCorpus as exc:
        print(f"[skip] {task_name}: {exc}")
        return {
            "task": task_name,
            "status": "skipped_large",
            "error": str(exc),
            "model": model_identity,
            "effective_ranking": wrapper.effective_ranking_parameters,
            **task_identity,
        }
    except Exception as exc:
        print(f"[error] {task_name}: {exc}")
        return {
            "task": task_name,
            "status": "error",
            "error": str(exc),
            "model": model_identity,
            "effective_ranking": wrapper.effective_ranking_parameters,
            **task_identity,
        }

    return {
        "task": task_name,
        "status": "ok",
        "model": model_identity,
        "effective_ranking": wrapper.effective_ranking_parameters,
        **task_identity,
    }


def main():
    parser = argparse.ArgumentParser(description="Official MTEB Runner for BasinRAG")
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_BEIR_ARENA,
        help="Tarefas MTEB separadas por vírgula (default: BEIR-EN arena)",
    )
    parser.add_argument("--search-type", type=str, default="hybrid")
    parser.add_argument(
        "--ranking-mode",
        choices=("hybrid_rrf", "experimental_topology"),
        default="hybrid_rrf",
        help="RRF de referência (padrão) ou ranking topológico experimental",
    )
    parser.add_argument(
        "--use-hop-prior",
        action="store_true",
        help="Habilita explicitamente o prior de hops topológicos (padrão: desabilitado)",
    )
    parser.add_argument("--encoder-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--encoder-revision", type=str, default=None)
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-revision", type=str, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument(
        "--query-prompt",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument(
        "--max-corpus-docs",
        type=int,
        default=None,
        help="Limita o tamanho do corpus (qualquer limite marca a execução como sampled; 0 = sem limite)",
    )
    parser.add_argument("--output", type=str, default="results/mteb_beir_arena")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prompt = "" if args.no_prompt else args.query_prompt
    if args.use_rerank and args.no_rerank:
        parser.error("Use apenas --use-rerank ou --no-rerank")
    rerank_flag = True if args.use_rerank else (False if args.no_rerank else None)
    max_docs = None if args.max_corpus_docs in (None, 0) else args.max_corpus_docs

    print(f"\n{'='*70}")
    print("      MTEB / BEIR ARENA")
    print(f"      Tarefas ({len(task_names)}): {task_names}")
    print(f"      Encoder: {args.encoder_model} | Busca: {args.search_type}")
    print(f"      Ranking: {args.ranking_mode} | hop prior: {args.use_hop_prior}")
    print(f"      Rerank: {'auto(flat=off)' if rerank_flag is None else rerank_flag}")
    print(f"      max_corpus_docs: {max_docs if max_docs is not None else 'unlimited'}")
    print(f"{'='*70}\n")

    # 4. Executa Avaliação Oficial
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"O output precisa ser novo/vazio; retomada exige identidade integral validada: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = args.max_queries is not None or max_docs is not None

    run_log: List[Dict[str, Any]] = []
    for name in task_names:
        try:
            row = run_one_task(
                name,
                encoder_model=args.encoder_model,
                encoder_revision=args.encoder_revision,
                reranker_model=args.reranker_model,
                reranker_revision=args.reranker_revision,
                search_type=args.search_type,
                rerank_top_k=args.rerank_top_k,
                prompt=prompt,
                rerank_flag=rerank_flag,
                force_reindex=args.force_reindex,
                max_queries=args.max_queries,
                max_corpus_docs=max_docs,
                output_dir=output_dir,
                ranking_mode=args.ranking_mode,
                use_hop_prior=args.use_hop_prior,
            )
        except Exception as exc:
            logger.exception("MTEB task %s failed before evaluation completed", name)
            row = {
                "task": name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        run_log.append(row)
        (output_dir / "arena_progress.json").write_text(
            json.dumps(run_log, indent=2), encoding="utf-8"
        )

    scores = _read_current_task_scores(output_dir, run_log)
    parameters = {
        "tasks": task_names,
        "search_type": args.search_type,
        "ranking_mode": args.ranking_mode,
        "use_hop_prior": args.use_hop_prior,
        "effective_ranking_by_task": {
            row["task"]: row["effective_ranking"]
            for row in run_log
            if row.get("task") and row.get("effective_ranking")
        },
        "encoder_model": args.encoder_model,
        "encoder_revision": args.encoder_revision,
        "reranker_model": args.reranker_model,
        "reranker_revision": args.reranker_revision,
        "rerank_top_k": args.rerank_top_k,
        "query_prompt": prompt,
        "no_prompt": args.no_prompt,
        "use_rerank": args.use_rerank,
        "no_rerank": args.no_rerank,
        "effective_rerank": rerank_flag,
        "max_queries": args.max_queries,
        "force_reindex": args.force_reindex,
        "max_corpus_docs": max_docs,
        "sampled": sampled,
        "output": args.output,
        "overwrite_strategy": "always",
        "show_progress_bar": True,
    }
    protocol = {
        "encoder": args.encoder_model,
        "rerank": rerank_flag,
        "search_type": args.search_type,
        "ranking_mode": args.ranking_mode,
        "use_hop_prior": args.use_hop_prior,
        "max_corpus_docs": max_docs,
        "query_prompt": prompt,
    }
    model_identity = next(
        (row["model"] for row in run_log if row.get("model")),
        _model_identity(args.encoder_model),
    )
    provenance = _build_provenance(
        parameters, task_names, scores, model_identity, run_log=run_log
    )
    outcome = _summarize_run(task_names, run_log, scores, sampled=sampled)
    summary = {
        "protocol": protocol,
        "parameters": parameters,
        "provenance": provenance,
        "requested_tasks": outcome["requested_tasks"],
        "completed_tasks": outcome["completed_tasks"],
        "skipped_tasks": outcome["skipped_tasks"],
        "failed_tasks": outcome["failed_tasks"],
        "complete": outcome["complete"],
        "aggregate_status": outcome["aggregate_status"],
        "run_log": run_log,
        "scores": scores,
        "n_completed": len(outcome["completed_tasks"]),
        "n_skipped": len(outcome["skipped_tasks"]),
        "n_skipped_large": sum(1 for r in run_log if r.get("status") == "skipped_large"),
        "n_errors": len(outcome["failed_tasks"]),
        "mean_ndcg@10": outcome["mean_ndcg@10"],
    }
    manifest = {
        "protocol": protocol,
        **_build_run_manifest(
            task_names,
            run_log,
            scores,
            parameters=parameters,
            provenance=provenance,
            sampled=sampled,
        ),
    }
    (output_dir / "arena_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary["manifest_path"] = "arena_manifest.json"
    (output_dir / "arena_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("      BEIR ARENA — RESUMO")
    for r in outcome["task_rows"]:
        print(f"  [{r.get('status'):14s}] {r.get('task')}"
              + (f" — {r.get('error')}" if r.get("error") else ""))
    if summary["mean_ndcg@10"] is not None:
        print(f"\n  mean nDCG@10 ({len(task_names)} tasks) = {summary['mean_ndcg@10']:.4f}")
    else:
        print(f"\n  aggregate nDCG@10 unavailable ({outcome['aggregate_status']})")
    print(f"  Artefatos: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
