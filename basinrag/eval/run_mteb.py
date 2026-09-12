"""
Runner oficial do MTEB para BasinRAG.
Executa tarefas de Retrieval padronizadas (ex: SciFact) e gera
os arquivos oficiais de submissão para o Hugging Face MTEB Leaderboard.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

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
from .mteb_wrapper import BasinRAGMTEBWrapper

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Official MTEB Runner for BasinRAG")
    parser.add_argument("--tasks", type=str, default="SciFact", help="Tarefas do MTEB separadas por vírgula (ex: SciFact)")
    parser.add_argument("--search-type", type=str, default="hybrid", help="Estratégia de busca do BasinRAG (hybrid, local, global)")
    parser.add_argument("--encoder-model", type=str, default="BAAI/bge-base-en-v1.5", help="Modelo de embedding (ex: BAAI/bge-base-en-v1.5)")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3", help="Modelo Cross-Encoder de reranking (ex: BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--rerank-top-k", type=int, default=25, help="Top-K candidatos submetidos ao Cross-Encoder (padrão: 25)")
    parser.add_argument("--query-prompt", type=str, default="Represent this sentence for searching relevant passages: ", help="Instrução/prompt de busca canônico para o encoder")
    parser.add_argument("--no-prompt", action="store_true", help="Desativa query prompting")
    parser.add_argument("--max-queries", type=int, default=None, help="Limita o número de queries avaliadas (para testes rápidos)")
    parser.add_argument("--force-reindex", action="store_true", help="Força a reindexação do corpus ignorando cache persistente")
    parser.add_argument("--output", type=str, default="results/mteb", help="Diretório de saída dos resultados")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prompt = "" if args.no_prompt else args.query_prompt
    print(f"\n{'='*70}")
    print("      EXECUÇÃO OFICIAL MTEB (HUGGING FACE LEADERBOARD)")
    print(f"      Tarefas: {task_names} | Busca: {args.search_type}")
    print(f"      Encoder: {args.encoder_model} | Reranker: {args.reranker_model} (Top-{args.rerank_top_k})")
    if prompt:
        print(f"      Prompt: '{prompt}'")
    if args.max_queries:
        print(f"      Amostra rápida: {args.max_queries} queries")
    if args.force_reindex:
        print("      Forçando reindexação do corpus (title+text, sem boost)...")
    print(f"{'='*70}\n")

    # 1. Instancia BasinRAG com cache isolado por encoder + primeira tarefa
    cache_tag = args.encoder_model.replace("/", "_").replace("-", "_")
    task_tag = task_names[0].replace("/", "_") if task_names else "unknown"
    storage_dir = f".basinrag/mteb_cache_{cache_tag}_{task_tag}"
    rag = BasinRAG.create(
        storage_dir=storage_dir,
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
    )
    wrapper = BasinRAGMTEBWrapper(
        rag,
        search_type=args.search_type,
        rerank_top_k=args.rerank_top_k,
        query_prompt=prompt,
        force_reindex=args.force_reindex,
        cache_tag=f"{task_tag}|{args.encoder_model}|flat",
    )
    if args.max_queries:
        wrapper.max_queries = args.max_queries

    # 2. Carrega tarefas MTEB
    tasks = mteb.get_tasks(tasks=task_names)
    print(f"Carregadas {len(tasks)} tarefas do MTEB.")

    # 3. Configura Cache de Saída do MTEB
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(cache_path=output_dir)

    # 4. Executa Avaliação Oficial
    results = mteb.evaluate(
        model=wrapper,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="always",
        prediction_folder=output_dir,
        show_progress_bar=True,
    )

    print("\n" + "="*70)
    print("      MTEB EVALUATION CONCLUÍDO COM SUCESSO!")
    print(f"      Resultados salvos em: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
