"""
Runner oficial do MTEB para BasinRAG 2.0.
Executa tarefas de Retrieval padronizadas (ex: SciFact) e gera
os arquivos oficiais de submissão para o Hugging Face MTEB Leaderboard.
"""
from __future__ import annotations

import os
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
    parser = argparse.ArgumentParser(description="Official MTEB Runner for BasinRAG 2.0")
    parser.add_argument("--tasks", type=str, default="SciFact", help="Tarefas do MTEB separadas por vírgula (ex: SciFact)")
    parser.add_argument("--search-type", type=str, default="hybrid", help="Estratégia de busca do BasinRAG (hybrid, local, global)")
    parser.add_argument("--output", type=str, default="results/mteb", help="Diretório de saída dos resultados")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"\n{'='*70}")
    print(f"      EXECUÇÃO OFICIAL MTEB (HUGGING FACE LEADERBOARD)")
    print(f"      Tarefas: {task_names} | Busca: {args.search_type}")
    print(f"{'='*70}\n")

    # 1. Instancia BasinRAG e Wrapper MTEB
    rag = BasinRAG.create(storage_dir=".basinrag/mteb_eval_cache")
    wrapper = BasinRAGMTEBWrapper(rag, search_type=args.search_type)

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
