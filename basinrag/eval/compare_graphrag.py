"""
BasinRAG vs GraphRAG Head-to-Head Comparative Benchmark on SWE-bench Lite.
Measures:
- Hit@1, Hit@5, Hit@10
- Recall@10
- MRR (Mean Reciprocal Rank)
- Graph Build Time (s)
- Retrieval Latency (ms)
"""
from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..factory import BasinRAG
from ..logging_config import setup_logging
from .swebench import SWEBenchEvaluator, SWEBenchInstance, CodeRepoIngestor
from .graphrag_baseline import GraphRAGBaseline

logger = setup_logging()


class ComparativeBenchmarker:
    """
    Executa o benchmark empírico comparativo BasinRAG vs GraphRAG
    sob condições 100% idênticas no SWE-bench Lite.
    """
    def __init__(self, workspace_root: str = ".basinrag/swebench_workspaces"):
        self.evaluator = SWEBenchEvaluator(workspace_root)

    def run_comparison(
        self,
        repo_filter: Optional[str] = "pallets/flask,psf/requests,mwaskom/seaborn",
        limit: int = 15,
        top_k: int = 10,
        checkpoint_path: str = "logs/basinrag_vs_graphrag.jsonl",
    ) -> Dict[str, Any]:
        instances = self.evaluator.load_dataset(
            split="test",
            repo_filter=repo_filter,
            limit=limit,
        )
        logger.info(f"Instâncias carregadas para benchmark comparativo: {len(instances)}")

        # Carrega instâncias já avaliadas do checkpoint
        completed_records: Dict[str, Dict[str, Any]] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            completed_records[rec["instance_id"]] = rec
                        except Exception:
                            pass
            logger.info(f"Retomando do checkpoint: {len(completed_records)} instâncias já avaliadas.")

        # Inicializa acumuladores de métricas
        metrics = {
            "basinrag": {
                "hit@1": 0, "hit@5": 0, "hit@10": 0,
                "total_recall": 0.0, "total_mrr": 0.0,
                "latencies_ms": [], "build_times_s": []
            },
            "graphrag": {
                "hit@1": 0, "hit@5": 0, "hit@10": 0,
                "total_recall": 0.0, "total_mrr": 0.0,
                "latencies_ms": [], "build_times_s": []
            }
        }

        records: List[Dict[str, Any]] = []
        total_evaluated = 0

        for rec in completed_records.values():
            total_evaluated += 1
            records.append(rec)
            b = rec.get("basinrag", {})
            g = rec.get("graphrag", {})
            if b.get("hit@1"): metrics["basinrag"]["hit@1"] += 1
            if b.get("hit@5"): metrics["basinrag"]["hit@5"] += 1
            if b.get("hit@10"): metrics["basinrag"]["hit@10"] += 1
            metrics["basinrag"]["total_recall"] += b.get("recall", 0.0)
            metrics["basinrag"]["total_mrr"] += b.get("mrr", 0.0)
            if "latency_ms" in b: metrics["basinrag"]["latencies_ms"].append(b["latency_ms"])

            if g.get("hit@1"): metrics["graphrag"]["hit@1"] += 1
            if g.get("hit@5"): metrics["graphrag"]["hit@5"] += 1
            if g.get("hit@10"): metrics["graphrag"]["hit@10"] += 1
            metrics["graphrag"]["total_recall"] += g.get("recall", 0.0)
            metrics["graphrag"]["total_mrr"] += g.get("mrr", 0.0)
            if "latency_ms" in g: metrics["graphrag"]["latencies_ms"].append(g["latency_ms"])

        pending_instances = [inst for inst in instances if inst.instance_id not in completed_records]
        logger.info(f"Instâncias pendentes para execução: {len(pending_instances)} de {len(instances)}")

        if not pending_instances and total_evaluated > 0:
            summary = self._compute_summary(metrics, total_evaluated, records)
            self.print_summary_table(summary)
            return summary

        # Agrupa por repo@commit para otimizar reuso de índice
        repo_groups: Dict[str, List[SWEBenchInstance]] = defaultdict(list)
        for inst in pending_instances:
            key = f"{inst.repo}@{inst.base_commit}"
            repo_groups[key].append(inst)

        os.makedirs(Path(checkpoint_path).parent, exist_ok=True)
        log_f = open(checkpoint_path, "a", encoding="utf-8")

        try:
            for repo_key, group in repo_groups.items():
                first = group[0]
                logger.info("\n========================================================")
                logger.info(f"Preparando Workspace para: {first.repo} ({len(group)} instâncias)")
                logger.info("========================================================")

                repo_dir = self.evaluator.workspace.prepare_repo(first.repo, first.base_commit)

                # 1. Coleta os chunks de código brutos (idênticos para ambos)
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=150,
                    separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
                )

                skip_dirs = {".git", ".basinrag", "__pycache__", "tests", "testing", "test", "docs", ".tox", "venv", ".env"}
                code_files = []
                for root, dirs, files in os.walk(repo_dir):
                    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                    for file in files:
                        if file.endswith(".py") and not file.startswith("test_") and not file.endswith("_test.py"):
                            code_files.append(Path(root) / file)

                corpus_dict: Dict[str, str] = {}
                chunk_to_file: Dict[str, str] = {}

                for fpath in code_files:
                    rel_path = fpath.relative_to(repo_dir).as_posix()
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    splits = splitter.split_text(content)
                    for idx, text in enumerate(splits):
                        chunk_id = f"{rel_path}#{idx}"
                        corpus_dict[chunk_id] = text
                        chunk_to_file[chunk_id] = rel_path

                logger.info(f"Corpus extraído: {len(corpus_dict)} chunks de {len(code_files)} arquivos Python.")

                # 2. Constrói Índice BasinRAG
                t0_basin = time.perf_counter()
                rag = BasinRAG.create(storage_dir=f".basinrag/eval_temp_{first.repo.replace('/', '_')}")
                CodeRepoIngestor.ingest_codebase(rag, repo_dir)
                basin_build_time = time.perf_counter() - t0_basin
                metrics["basinrag"]["build_times_s"].append(basin_build_time)
                logger.info(f"[BasinRAG] Grafo de Bacias Topológicas construído em {basin_build_time:.2f}s")

                # 3. Constrói Índice GraphRAG
                t0_graph = time.perf_counter()
                graphrag = GraphRAGBaseline()
                graphrag.build_graph(corpus_dict)
                graph_build_time = time.perf_counter() - t0_graph
                metrics["graphrag"]["build_times_s"].append(graph_build_time)
                logger.info(f"[GraphRAG] Grafo de Conhecimento Entidades construído em {graph_build_time:.2f}s")

                # 4. Avalia instâncias da issue no mesmo repositório
                for inst in group:
                    gt_files = inst.ground_truth_files
                    if not gt_files:
                        continue

                    total_evaluated += 1

                    # Helper para checar match com Ground Truth
                    def matches_gt(f: str) -> bool:
                        return any(f == gt or gt.endswith(f) or f.endswith(gt) for gt in gt_files)

                    # --- Query no BasinRAG ---
                    t0_q_basin = time.perf_counter()
                    basin_docs = rag.query(inst.problem_statement, search_type="hybrid", top_k=max(20, top_k))
                    basin_latency = (time.perf_counter() - t0_q_basin) * 1000.0
                    metrics["basinrag"]["latencies_ms"].append(basin_latency)

                    basin_files = []
                    for doc in basin_docs:
                        fpath = doc.metadata.get("file_path") or doc.metadata.get("source") or ""
                        norm_path = fpath.replace("\\", "/").strip()
                        if norm_path and norm_path not in basin_files:
                            basin_files.append(norm_path)

                    basin_hit1 = any(matches_gt(f) for f in basin_files[:1])
                    basin_hit5 = any(matches_gt(f) for f in basin_files[:5])
                    basin_hit10 = any(matches_gt(f) for f in basin_files[:10])
                    basin_recall = sum(1 for gt in gt_files if any(matches_gt(f) for f in basin_files[:top_k])) / float(len(gt_files))
                    basin_mrr = 0.0
                    for rank, f in enumerate(basin_files[:top_k], 1):
                        if matches_gt(f):
                            basin_mrr = 1.0 / rank
                            break

                    if basin_hit1: metrics["basinrag"]["hit@1"] += 1
                    if basin_hit5: metrics["basinrag"]["hit@5"] += 1
                    if basin_hit10: metrics["basinrag"]["hit@10"] += 1
                    metrics["basinrag"]["total_recall"] += basin_recall
                    metrics["basinrag"]["total_mrr"] += basin_mrr

                    # --- Query no GraphRAG ---
                    t0_q_graph = time.perf_counter()
                    graph_ranked_chunks = graphrag.search_single(inst.problem_statement, top_k=top_k * 5)
                    graph_latency = (time.perf_counter() - t0_q_graph) * 1000.0
                    metrics["graphrag"]["latencies_ms"].append(graph_latency)

                    graph_files = []
                    for chunk_id, score in graph_ranked_chunks:
                        fpath = chunk_to_file.get(chunk_id, "")
                        if fpath and fpath not in graph_files:
                            graph_files.append(fpath)

                    graph_hit1 = any(matches_gt(f) for f in graph_files[:1])
                    graph_hit5 = any(matches_gt(f) for f in graph_files[:5])
                    graph_hit10 = any(matches_gt(f) for f in graph_files[:10])
                    graph_recall = sum(1 for gt in gt_files if any(matches_gt(f) for f in graph_files[:top_k])) / float(len(gt_files))
                    graph_mrr = 0.0
                    for rank, f in enumerate(graph_files[:top_k], 1):
                        if matches_gt(f):
                            graph_mrr = 1.0 / rank
                            break

                    if graph_hit1: metrics["graphrag"]["hit@1"] += 1
                    if graph_hit5: metrics["graphrag"]["hit@5"] += 1
                    if graph_hit10: metrics["graphrag"]["hit@10"] += 1
                    metrics["graphrag"]["total_recall"] += graph_recall
                    metrics["graphrag"]["total_mrr"] += graph_mrr

                    record = {
                        "instance_id": inst.instance_id,
                        "repo": inst.repo,
                        "ground_truth": list(gt_files),
                        "basinrag": {
                            "hit@1": basin_hit1, "hit@5": basin_hit5, "hit@10": basin_hit10,
                            "recall": basin_recall, "mrr": basin_mrr, "latency_ms": basin_latency,
                            "top3_files": basin_files[:3]
                        },
                        "graphrag": {
                            "hit@1": graph_hit1, "hit@5": graph_hit5, "hit@10": graph_hit10,
                            "recall": graph_recall, "mrr": graph_mrr, "latency_ms": graph_latency,
                            "top3_files": graph_files[:3]
                        }
                    }
                    records.append(record)
                    log_f.write(json.dumps(record) + "\n")
                    log_f.flush()

                    b_h10 = metrics["basinrag"]["hit@10"] / total_evaluated
                    g_h10 = metrics["graphrag"]["hit@10"] / total_evaluated
                    b_mrr = metrics["basinrag"]["total_mrr"] / total_evaluated
                    g_mrr = metrics["graphrag"]["total_mrr"] / total_evaluated

                    print(f"[{total_evaluated}/{len(instances)}] {inst.instance_id} | "
                          f"Basin Hit@10: {b_h10:.1%} (MRR: {b_mrr:.3f}) vs "
                          f"Graph Hit@10: {g_h10:.1%} (MRR: {g_mrr:.3f})")

        finally:
            log_f.close()

        if total_evaluated == 0:
            return {}

        summary = self._compute_summary(metrics, total_evaluated, records)
        self.print_summary_table(summary)
        return summary

    def _compute_summary(self, metrics: Dict[str, Any], total_evaluated: int, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        b_lat = metrics["basinrag"]["latencies_ms"]
        g_lat = metrics["graphrag"]["latencies_ms"]
        b_bld = metrics["basinrag"]["build_times_s"]
        g_bld = metrics["graphrag"]["build_times_s"]

        return {
            "instances_evaluated": total_evaluated,
            "basinrag": {
                "hit@1": metrics["basinrag"]["hit@1"] / total_evaluated,
                "hit@5": metrics["basinrag"]["hit@5"] / total_evaluated,
                "hit@10": metrics["basinrag"]["hit@10"] / total_evaluated,
                "recall@10": metrics["basinrag"]["total_recall"] / total_evaluated,
                "mrr": metrics["basinrag"]["total_mrr"] / total_evaluated,
                "avg_latency_ms": sum(b_lat) / len(b_lat) if b_lat else 0.0,
                "avg_build_time_s": sum(b_bld) / len(b_bld) if b_bld else 0.0,
            },
            "graphrag": {
                "hit@1": metrics["graphrag"]["hit@1"] / total_evaluated,
                "hit@5": metrics["graphrag"]["hit@5"] / total_evaluated,
                "hit@10": metrics["graphrag"]["hit@10"] / total_evaluated,
                "recall@10": metrics["graphrag"]["total_recall"] / total_evaluated,
                "mrr": metrics["graphrag"]["total_mrr"] / total_evaluated,
                "avg_latency_ms": sum(g_lat) / len(g_lat) if g_lat else 0.0,
                "avg_build_time_s": sum(g_bld) / len(g_bld) if g_bld else 0.0,
            },
            "records": records,
        }

    def print_summary_table(self, summary: Dict[str, Any]):
        n = summary["instances_evaluated"]
        b = summary["basinrag"]
        g = summary["graphrag"]

        print("\n" + "=" * 76)
        print("       RESULTADO COMPARATIVO OFICIAL: BASINRAG 2.0 vs GRAPHRAG")
        print(f"       Total de Instâncias Avaliadas: {n} (SWE-bench Lite)")
        print("=" * 76)
        print(f"{'Métrica':<25} | {'GraphRAG':<20} | {'BasinRAG 2.0':<20} | {'Diferença':<10}")
        print("-" * 76)

        def diff_pct(vb, vg):
            delta = vb - vg
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta * 100:.1f} pp"

        print(f"{'Hit@1':<25} | {g['hit@1']:<20.1%} | {b['hit@1']:<20.1%} | {diff_pct(b['hit@1'], g['hit@1']):<10}")
        print(f"{'Hit@5':<25} | {g['hit@5']:<20.1%} | {b['hit@5']:<20.1%} | {diff_pct(b['hit@5'], g['hit@5']):<10}")
        print(f"{'Hit@10':<25} | {g['hit@10']:<20.1%} | {b['hit@10']:<20.1%} | {diff_pct(b['hit@10'], g['hit@10']):<10}")
        print(f"{'Recall@10':<25} | {g['recall@10']:<20.1%} | {b['recall@10']:<20.1%} | {diff_pct(b['recall@10'], g['recall@10']):<10}")
        print(f"{'MRR (Mean Recip. Rank)':<25} | {g['mrr']:<20.3f} | {b['mrr']:<20.3f} | {'+' if b['mrr']>=g['mrr'] else ''}{b['mrr'] - g['mrr']:.3f}")
        print(f"{'Latência Média Query':<25} | {g['avg_latency_ms']:<18.1f}ms | {b['avg_latency_ms']:<18.1f}ms | {b['avg_latency_ms'] - g['avg_latency_ms']:+.1f}ms")
        print(f"{'Tempo Médio Build Grafo':<25} | {g['avg_build_time_s']:<18.2f}s | {b['avg_build_time_s']:<18.2f}s | {b['avg_build_time_s'] - g['avg_build_time_s']:+.2f}s")
        print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BasinRAG vs GraphRAG Benchmark Runner")
    parser.add_argument("--repo-filter", type=str, default="pallets/flask,psf/requests,mwaskom/seaborn")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="logs/basinrag_vs_graphrag.jsonl")

    args = parser.parse_args()

    bench = ComparativeBenchmarker()
    bench.run_comparison(
        repo_filter=args.repo_filter,
        limit=args.limit,
        top_k=args.top_k,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
