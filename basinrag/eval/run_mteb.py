"""
Runner oficial do MTEB para BasinRAG.
Executa tarefas de Retrieval padronizadas e gera artefatos no formato MTEB.

Cada task usa storage isolado (evita contaminação de corpus entre SciFact/NFCorpus/…).
Corpora acima de --max-corpus-docs são pulados (Wikipedia-scale estoura RAM/tempo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Patch PyTorch DTensor antes de carregar mteb
try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

import mteb
from mteb.cache import ResultCache

from ..factory import BasinRAG
from ..logging_config import setup_logging
from .mteb_wrapper import BasinRAGMTEBWrapper, SkipLargeCorpus

logger = setup_logging()

# BEIR-EN arena padrão (média nDCG@10 comparável ao board de Retrieval).
DEFAULT_BEIR_ARENA = (
    "SciFact,NFCorpus,FiQA2018,ArguAna,QuoraRetrieval,"
    "TRECCOVID,Touche2020,ClimateFEVER,DBPedia,FEVER,HotpotQA,NQ"
)


def _encoder_tag(encoder_model: str) -> str:
    return encoder_model.replace("/", "_").replace("-", "_")


def _read_task_main_score(output_dir: Path, model_name: str = "Basinfy__BasinRAG") -> Dict[str, Any]:
    """Collect nDCG@10 / recall@10 from written MTEB result JSONs."""
    out: Dict[str, Any] = {}
    root = output_dir / "results"
    if not root.exists():
        return out
    for path in root.rglob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task = payload.get("task_name") or path.stem
        scores = payload.get("scores") or {}
        test = scores.get("test") or scores.get("dev") or []
        if not test:
            continue
        row = test[0] if isinstance(test, list) else test
        out[task] = {
            "ndcg@10": row.get("ndcg_at_10") or row.get("main_score"),
            "recall@10": row.get("recall_at_10"),
            "mrr@10": row.get("mrr_at_10"),
            "main_score": row.get("main_score"),
            "path": str(path),
        }
    return out


def run_one_task(
    task_name: str,
    *,
    encoder_model: str,
    reranker_model: str,
    search_type: str,
    rerank_top_k: int,
    prompt: str,
    rerank_flag: Optional[bool],
    force_reindex: bool,
    max_queries: Optional[int],
    max_corpus_docs: Optional[int],
    output_dir: Path,
) -> Dict[str, Any]:
    cache_tag = _encoder_tag(encoder_model)
    task_tag = task_name.replace("/", "_")
    storage_dir = f".basinrag/mteb_cache_{cache_tag}_{task_tag}"

    print(f"\n{'='*70}")
    print(f"  TASK: {task_name}")
    print(f"  storage: {storage_dir}")
    print(f"{'='*70}\n")

    rag = BasinRAG.create(
        storage_dir=storage_dir,
        encoder_model=encoder_model,
        reranker_model=reranker_model,
        use_rerank=bool(rerank_flag) if rerank_flag is not None else False,
        query_prompt=prompt,
    )
    wrapper = BasinRAGMTEBWrapper(
        rag,
        search_type=search_type,
        rerank_top_k=rerank_top_k,
        query_prompt=prompt,
        force_reindex=force_reindex,
        cache_tag=f"{task_tag}|{encoder_model}|flat",
        use_rerank=rerank_flag,
        max_corpus_docs=max_corpus_docs,
    )
    if max_queries:
        wrapper.max_queries = max_queries

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        return {"task": task_name, "status": "missing", "error": "task not found in mteb"}

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
        return {"task": task_name, "status": "skipped_large", "error": str(exc)}
    except Exception as exc:
        print(f"[error] {task_name}: {exc}")
        return {"task": task_name, "status": "error", "error": str(exc)}

    return {"task": task_name, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Official MTEB Runner for BasinRAG")
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_BEIR_ARENA,
        help="Tarefas MTEB separadas por vírgula (default: BEIR-EN arena)",
    )
    parser.add_argument("--search-type", type=str, default="hybrid")
    parser.add_argument("--encoder-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
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
        default=200_000,
        help="Pula task se o corpus tiver mais docs que isso (0 = sem limite)",
    )
    parser.add_argument("--output", type=str, default="results/mteb_beir_arena")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prompt = "" if args.no_prompt else args.query_prompt
    if args.use_rerank and args.no_rerank:
        parser.error("Use apenas --use-rerank ou --no-rerank")
    rerank_flag = True if args.use_rerank else (False if args.no_rerank else None)
    max_docs = None if args.max_corpus_docs == 0 else args.max_corpus_docs

    print(f"\n{'='*70}")
    print("      MTEB / BEIR ARENA")
    print(f"      Tarefas ({len(task_names)}): {task_names}")
    print(f"      Encoder: {args.encoder_model} | Busca: {args.search_type}")
    print(f"      Rerank: {'auto(flat=off)' if rerank_flag is None else rerank_flag}")
    print(f"      max_corpus_docs: {max_docs if max_docs is not None else 'unlimited'}")
    print(f"{'='*70}\n")

    # 4. Executa Avaliação Oficial
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log: List[Dict[str, Any]] = []
    for name in task_names:
        row = run_one_task(
            name,
            encoder_model=args.encoder_model,
            reranker_model=args.reranker_model,
            search_type=args.search_type,
            rerank_top_k=args.rerank_top_k,
            prompt=prompt,
            rerank_flag=rerank_flag,
            force_reindex=args.force_reindex,
            max_queries=args.max_queries,
            max_corpus_docs=max_docs,
            output_dir=output_dir,
        )
        run_log.append(row)
        (output_dir / "arena_progress.json").write_text(
            json.dumps(run_log, indent=2), encoding="utf-8"
        )

    scores = _read_task_main_score(output_dir)
    ndcgs = [float(v["ndcg@10"]) for v in scores.values() if v.get("ndcg@10") is not None]
    summary = {
        "protocol": {
            "encoder": args.encoder_model,
            "rerank": rerank_flag,
            "search_type": args.search_type,
            "max_corpus_docs": max_docs,
            "query_prompt": prompt,
        },
        "run_log": run_log,
        "scores": scores,
        "n_completed": sum(1 for r in run_log if r.get("status") == "ok"),
        "n_skipped_large": sum(1 for r in run_log if r.get("status") == "skipped_large"),
        "n_errors": sum(1 for r in run_log if r.get("status") == "error"),
        "mean_ndcg@10": (sum(ndcgs) / len(ndcgs)) if ndcgs else None,
    }
    (output_dir / "arena_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("      BEIR ARENA — RESUMO")
    for r in run_log:
        print(f"  [{r.get('status'):14s}] {r.get('task')}"
              + (f" — {r.get('error')}" if r.get("error") else ""))
    if summary["mean_ndcg@10"] is not None:
        print(f"\n  mean nDCG@10 ({len(ndcgs)} tasks) = {summary['mean_ndcg@10']:.4f}")
    print(f"  Artefatos: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
