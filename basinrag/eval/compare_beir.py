"""
Official BEIR Comparative Benchmark: BasinRAG 2.0 vs GraphRAG.
Evaluates using the official BEIR framework (EvaluateRetrieval) on SciFact.
Metrics:
- NDCG@{1, 5, 10}
- MAP@{1, 5, 10}
- Recall@{1, 5, 10}
- Precision@{1, 5, 10}
- MRR@{1, 5, 10}
- Graph Build Time & Retrieval Latency
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from beir.retrieval.evaluation import EvaluateRetrieval

from ..factory import BasinRAG
from ..logging_config import setup_logging
from .beir import download_and_unzip_beir_dataset, load_beir_dataset, BasinRAGBEIRAdapter, ingest_beir_corpus
from .graphrag_baseline import GraphRAGBaseline

logger = setup_logging()


class BEIRComparativeBenchmarker:
    """
    Executa o benchmark empírico comparativo oficial do BEIR
    (BasinRAG 2.0 vs GraphRAG) sob condições 100% idênticas e imparciais.
    """
    def __init__(self, cache_dir: str = ".basinrag/eval_cache"):
        self.cache_dir = cache_dir

    def run_comparison(
        self,
        dataset_name: str = "scifact",
        split: str = "test",
        num_queries: Optional[int] = 50,
        top_k: int = 10,
        output_file: str = "logs/beir_comparison_scifact.json",
    ) -> Dict[str, Any]:
        logger.info(f"Carregando dataset BEIR '{dataset_name}' ({split})...")
        dataset_dir = download_and_unzip_beir_dataset(dataset_name, out_dir=self.cache_dir)
        corpus, all_queries, qrels_raw = load_beir_dataset(dataset_dir, split=split)

        # Converte qrels para o formato oficial BEIR: Dict[str, Dict[str, int]]
        qrels: Dict[str, Dict[str, int]] = {}
        for qid, doc_set in qrels_raw.items():
            qrels[qid] = {doc_id: 1 for doc_id in doc_set}

        # Filtra queries que possuem ground truth no qrels
        valid_qids = [qid for qid in all_queries if qid in qrels and len(qrels[qid]) > 0]
        if num_queries and len(valid_qids) > num_queries:
            valid_qids = valid_qids[:num_queries]

        eval_queries = {qid: all_queries[qid] for qid in valid_qids}
        eval_qrels = {qid: qrels[qid] for qid in valid_qids}

        logger.info(f"Dataset carregado: {len(corpus)} documentos no corpus | {len(eval_queries)} queries a avaliar.")

        # ---------------------------------------------------------
        # 1. Construção do Índice BasinRAG 2.0
        # ---------------------------------------------------------
        logger.info("\n[1/4] Construindo Índice BasinRAG 2.0 (Topological Basins + FAISS + BM25)...")
        rag = BasinRAG.create(storage_dir=f".basinrag/eval_beir_{dataset_name}")
        t0_basin_build = time.perf_counter()
        n_nodes = ingest_beir_corpus(rag, corpus)
        basin_build_time = time.perf_counter() - t0_basin_build
        logger.info(f"[BasinRAG] Grafo de Bacias Topológicas construído com {n_nodes} nós em {basin_build_time:.2f}s")

        # ---------------------------------------------------------
        # 2. Construção do Índice GraphRAG Baseline
        # ---------------------------------------------------------
        logger.info("\n[2/4] Construindo Índice GraphRAG Baseline (SpaCy NER + Knowledge Graph)...")
        graphrag = GraphRAGBaseline()
        t0_graph_build = time.perf_counter()
        graphrag.build_graph(corpus)
        graph_build_time = time.perf_counter() - t0_graph_build
        logger.info(f"[GraphRAG] Grafo Bipartido de Conhecimento construído em {graph_build_time:.2f}s")

        # ---------------------------------------------------------
        # 3. Execução de Busca BasinRAG
        # ---------------------------------------------------------
        logger.info("\n[3/4] Executando queries no BasinRAG 2.0...")
        adapter = BasinRAGBEIRAdapter(rag)
        t0_basin_search = time.perf_counter()
        basin_results = adapter.search_beir(eval_queries, top_k=top_k, search_type="hybrid")
        basin_search_time = time.perf_counter() - t0_basin_search
        basin_avg_latency_ms = (basin_search_time / len(eval_queries)) * 1000.0

        # ---------------------------------------------------------
        # 4. Execução de Busca GraphRAG
        # ---------------------------------------------------------
        logger.info("\n[4/4] Executando queries no GraphRAG Baseline...")
        t0_graph_search = time.perf_counter()
        graph_results = graphrag.search_beir(eval_queries, top_k=top_k)
        graph_search_time = time.perf_counter() - t0_graph_search
        graph_avg_latency_ms = (graph_search_time / len(eval_queries)) * 1000.0

        # ---------------------------------------------------------
        # 5. Avaliação Oficial BEIR (EvaluateRetrieval)
        # ---------------------------------------------------------
        logger.info("\nCalculando métricas oficiais do BEIR (NDCG, MAP, Recall, Precision, MRR)...")
        evaluator = EvaluateRetrieval()
        k_values = [1, 5, 10]

        b_ndcg, b_map, b_recall, b_precision = evaluator.evaluate(eval_qrels, basin_results, k_values)
        b_mrr = evaluator.evaluate_custom(eval_qrels, basin_results, k_values, metric="mrr")

        g_ndcg, g_map, g_recall, g_precision = evaluator.evaluate(eval_qrels, graph_results, k_values)
        g_mrr = evaluator.evaluate_custom(eval_qrels, graph_results, k_values, metric="mrr")

        summary = {
            "dataset": dataset_name,
            "num_docs": len(corpus),
            "num_queries": len(eval_queries),
            "basinrag": {
                "ndcg": b_ndcg,
                "map": b_map,
                "recall": b_recall,
                "precision": b_precision,
                "mrr": b_mrr,
                "build_time_s": basin_build_time,
                "avg_latency_ms": basin_avg_latency_ms,
            },
            "graphrag": {
                "ndcg": g_ndcg,
                "map": g_map,
                "recall": g_recall,
                "precision": g_precision,
                "mrr": g_mrr,
                "build_time_s": graph_build_time,
                "avg_latency_ms": graph_avg_latency_ms,
            }
        }

        # Salva resultados no log
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Resultados exportados para: {output_file}")

        # Exibe Tabela Comparativa Formatada
        self.print_summary_table(summary)
        return summary

    def print_summary_table(self, summary: Dict[str, Any]):
        b = summary["basinrag"]
        g = summary["graphrag"]
        ds = summary["dataset"]
        nq = summary["num_queries"]
        nd = summary["num_docs"]

        print("\n" + "=" * 78)
        print(f"      BENCHMARK OFICIAL BEIR: BASINRAG 2.0 vs GRAPHRAG ({ds.upper()})")
        print(f"      Corpus: {nd} documentos | Queries Avaliadas: {nq} | Evaluator: BEIR")
        print("=" * 78)
        print(f"{'Métrica BEIR Oficial':<26} | {'GraphRAG':<18} | {'BasinRAG 2.0':<18} | {'Diferença':<10}")
        print("-" * 78)

        def diff_pct(vb, vg):
            delta = vb - vg
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta * 100:.1f} pp"

        def diff_abs(vb, vg):
            delta = vb - vg
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.3f}"

        print(f"{'NDCG@10 (Score Oficial)':<26} | {g['ndcg']['NDCG@10']:<18.3f} | {b['ndcg']['NDCG@10']:<18.3f} | {diff_abs(b['ndcg']['NDCG@10'], g['ndcg']['NDCG@10']):<10}")
        print(f"{'NDCG@5':<26} | {g['ndcg']['NDCG@5']:<18.3f} | {b['ndcg']['NDCG@5']:<18.3f} | {diff_abs(b['ndcg']['NDCG@5'], g['ndcg']['NDCG@5']):<10}")
        print(f"{'Recall@10':<26} | {g['recall']['Recall@10']:<18.1%} | {b['recall']['Recall@10']:<18.1%} | {diff_pct(b['recall']['Recall@10'], g['recall']['Recall@10']):<10}")
        print(f"{'Recall@5':<26} | {g['recall']['Recall@5']:<18.1%} | {b['recall']['Recall@5']:<18.1%} | {diff_pct(b['recall']['Recall@5'], g['recall']['Recall@5']):<10}")
        print(f"{'MRR@10':<26} | {g['mrr']['MRR@10']:<18.3f} | {b['mrr']['MRR@10']:<18.3f} | {diff_abs(b['mrr']['MRR@10'], g['mrr']['MRR@10']):<10}")
        print(f"{'MAP@10':<26} | {g['map']['MAP@10']:<18.3f} | {b['map']['MAP@10']:<18.3f} | {diff_abs(b['map']['MAP@10'], g['map']['MAP@10']):<10}")
        print(f"{'Precision@10':<26} | {g['precision']['P@10']:<18.1%} | {b['precision']['P@10']:<18.1%} | {diff_pct(b['precision']['P@10'], g['precision']['P@10']):<10}")
        print(f"{'Tempo de Build Grafo':<26} | {g['build_time_s']:<16.2f}s | {b['build_time_s']:<16.2f}s | {b['build_time_s'] - g['build_time_s']:+.2f}s")
        print(f"{'Latência Média Query':<26} | {g['avg_latency_ms']:<16.1f}ms | {b['avg_latency_ms']:<16.1f}ms | {b['avg_latency_ms'] - g['avg_latency_ms']:+.1f}ms")
        print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Official BEIR Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=str, default="logs/beir_comparison_scifact.json")

    args = parser.parse_args()

    bench = BEIRComparativeBenchmarker()
    bench.run_comparison(
        dataset_name=args.dataset,
        split=args.split,
        num_queries=args.num_queries,
        top_k=args.top_k,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
