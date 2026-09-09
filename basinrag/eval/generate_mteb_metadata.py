"""
Official MTEB Metadata Generator for BasinRAG.
Generates mteb_metadata.md with YAML frontmatter conforming to:
https://huggingface.co/blog/mteb

Usage:
    python -m basinrag.eval.generate_mteb_metadata [results_dir] [--model-name Basinfy/BasinRAG]
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


TASK_TYPE_MAPPING = {
    "SciFact": "Retrieval",
    "NFCorpus": "Retrieval",
    "FiQA2018": "Retrieval",
    "ArguAna": "Retrieval",
    "SCIDOCS": "Retrieval",
    "Touche2020": "Retrieval",
    "DBPedia": "Retrieval",
    "TRECCOVID": "Retrieval",
    "QuoraRetrieval": "Retrieval",
    "CQADupstackRetrieval": "Retrieval",
    "ClimateFEVER": "Retrieval",
    "FEVER": "Retrieval",
    "HotpotQA": "Retrieval",
    "MSMARCO": "Retrieval",
}


def parse_task_results(task_file: Path) -> Dict[str, Any]:
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_name = data.get("task_name", task_file.stem)
    dataset_revision = data.get("dataset_revision", "")
    task_type = TASK_TYPE_MAPPING.get(task_name, "Retrieval")
    dataset_type = f"mteb/{task_name.lower()}" if not task_name.lower().startswith("mteb/") else task_name.lower()

    # Extrai o split de scores (test por padrão)
    scores = data.get("scores", {})
    split = "test" if "test" in scores else list(scores.keys())[0]
    split_scores = scores[split]
    if isinstance(split_scores, list) and len(split_scores) > 0:
        raw_metrics = split_scores[0]
    elif isinstance(split_scores, dict):
        raw_metrics = split_scores
    else:
        raw_metrics = {}

    # Filtra métricas relevantes e converte para escala percentual (0 a 100) como padrão do Hub
    metrics_list: List[Dict[str, Any]] = []
    priority_metrics = [
        "map_at_1", "map_at_3", "map_at_5", "map_at_10", "map_at_20", "map_at_100", "map_at_1000",
        "mrr_at_1", "mrr_at_3", "mrr_at_5", "mrr_at_10", "mrr_at_20", "mrr_at_100", "mrr_at_1000",
        "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "ndcg_at_100", "ndcg_at_1000",
        "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10", "precision_at_20", "precision_at_100", "precision_at_1000",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_100", "recall_at_1000",
        "accuracy"
    ]

    for k in priority_metrics:
        if k in raw_metrics and raw_metrics[k] is not None:
            val = raw_metrics[k]
            val_pct = round(val * 100.0, 5) if isinstance(val, (int, float)) else val
            metrics_list.append({"type": k, "value": val_pct})

    for k, val in sorted(raw_metrics.items()):
        if k not in priority_metrics and not k.startswith("nauc_") and isinstance(val, (int, float)):
            val_pct = round(val * 100.0, 5)
            metrics_list.append({"type": k, "value": val_pct})

    return {
        "task": {"type": task_type},
        "dataset": {
            "type": dataset_type,
            "name": f"MTEB {task_name}",
            "config": "default",
            "split": split,
            "revision": dataset_revision,
        },
        "metrics": metrics_list,
    }


def generate_metadata_yaml(model_name: str, results_list: List[Dict[str, Any]]) -> str:
    lines = [
        "---",
        "pipeline_tag: sentence-similarity",
        "tags:",
        "- sentence-transformers",
        "- feature-extraction",
        "- sentence-similarity",
        "- mteb",
        "model-index:",
        f"- name: {model_name}",
        "  results:",
    ]

    for r in results_list:
        lines.append("  - task:")
        lines.append(f"      type: {r['task']['type']}")
        lines.append("    dataset:")
        lines.append(f"      type: {r['dataset']['type']}")
        lines.append(f"      name: {r['dataset']['name']}")
        lines.append(f"      config: {r['dataset']['config']}")
        lines.append(f"      split: {r['dataset']['split']}")
        if r['dataset']['revision']:
            lines.append(f"      revision: {r['dataset']['revision']}")
        lines.append("    metrics:")
        for m in r["metrics"]:
            lines.append(f"    - type: {m['type']}")
            lines.append(f"      value: {m['value']}")

    lines.append("---")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate mteb_metadata.md according to huggingface.co/blog/mteb")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/mteb/results/Basinfy__BasinRAG/1.0.3",
        help="Caminho para o diretório contendo os JSONs das tarefas (ex: SciFact.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Basinfy/BasinRAG",
        help="Nome do modelo no Hugging Face Hub (ex: Basinfy/BasinRAG)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mteb/mteb_metadata.md",
        help="Caminho para salvar o mteb_metadata.md",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {results_dir}")

    json_files = [f for f in results_dir.glob("*.json") if f.name != "model_meta.json"]
    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo JSON de tarefa encontrado em: {results_dir}")

    print(f"\nProcessando {len(json_files)} tarefas do MTEB em: {results_dir}")
    all_results = []
    for jf in json_files:
        print(f"  - Carregando tarefa: {jf.stem}")
        all_results.append(parse_task_results(jf))

    metadata_yaml = generate_metadata_yaml(args.model_name, all_results)

    output_file = Path(args.output).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(metadata_yaml + "\n")

    print(f"\n[OK] Arquivo oficial gerado com sucesso: {output_file}")
    print("\n" + "=" * 65)
    print("Copie o bloco YAML abaixo para o topo do README.md no Hugging Face Hub:")
    print("=" * 65)
    print(metadata_yaml[:1200] + "\n... [truncado para exibição] ...\n---")


if __name__ == "__main__":
    main()
