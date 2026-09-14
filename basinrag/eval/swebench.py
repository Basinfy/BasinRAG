"""
BasinRAG SWE-bench Lite Evaluation Module.
Evaluates Topological Retrieval / Fault Localization (Hit@k, Recall@k, MRR)
and exports official predictions for the SWE-bench Docker Harness.
Reference: https://www.swebench.com/lite.html (arXiv:2310.06770)
"""
from __future__ import annotations

import os
import json
import hashlib
import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import List, Dict, Set, Any, Optional

from datasets import load_dataset

from ..factory import BasinRAG, BasinRAGConfig
from ..contracts import SearchType
from ..logging_config import setup_logging

logger = setup_logging()

from .swebench_protocol import docker_available, build_eval_command

DATASET_ID = "SWE-bench/SWE-bench_Lite"
CHECKPOINT_METRIC_VERSION = "exact-path-v1"
CHECKPOINT_PROTOCOL_REVISION = "swebench-retrieval-v2"
CODE_CHUNKER_REVISION = "recursive-character-v1"
CODE_CHUNK_SIZE = 1200
CODE_CHUNK_OVERLAP = 150
CODE_CHUNK_SEPARATORS = ("\nclass ", "\ndef ", "\n\n", "\n", " ")
EMBEDDING_TOKENIZER_POLICY = "sentence-transformers-default-for-encoder-id-unpinned-v1"


def _normalise_repo_path(path: str) -> str:
    """Normalize separators and harmless leading './' without fuzzy matching."""
    value = (path or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return PurePosixPath(value).as_posix() if value else ""


def _matches_gold_path(path: str, gold_files: Set[str]) -> bool:
    candidate = _normalise_repo_path(path)
    return bool(candidate) and candidate in {_normalise_repo_path(gold) for gold in gold_files}


def _checkpoint_protocol_id(
    *,
    split: str,
    top_k: int,
    search_type: str,
    encoder_model: str,
    reranker_model: str,
    use_rerank: bool,
    query_prompt: str,
    ranking_mode: str,
    candidate_k: Optional[int] = None,
    chunk_size: int = CODE_CHUNK_SIZE,
    chunk_overlap: int = CODE_CHUNK_OVERLAP,
    chunker_revision: str = CODE_CHUNKER_REVISION,
    tokenizer_policy: str = EMBEDDING_TOKENIZER_POLICY,
) -> str:
    protocol = {
        "protocol_revision": CHECKPOINT_PROTOCOL_REVISION,
        "dataset": DATASET_ID,
        "split": split,
        "top_k": int(top_k),
        "candidate_k": int(candidate_k if candidate_k is not None else max(20, top_k)),
        "search_type": search_type,
        "ranking_mode": ranking_mode,
        "encoder_model": encoder_model,
        "reranker_model": reranker_model,
        "use_rerank": bool(use_rerank),
        "query_prompt": query_prompt,
        "chunking": {
            "implementation": "langchain_text_splitters.RecursiveCharacterTextSplitter",
            "revision": chunker_revision,
            "length_unit": "unicode-code-points",
            "size": int(chunk_size),
            "overlap": int(chunk_overlap),
            "separators": list(CODE_CHUNK_SEPARATORS),
            "embedding_tokenizer_policy": tokenizer_policy,
        },
        "metric_version": CHECKPOINT_METRIC_VERSION,
    }
    canonical = json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_subset_id(instances: List["SWEBenchInstance"]) -> str:
    subset = sorted(
        (
            instance.instance_id,
            instance.repo,
            instance.base_commit,
            instance.problem_statement,
            sorted(_normalise_repo_path(path) for path in instance.ground_truth_files),
        )
        for instance in instances
    )
    canonical = json.dumps(subset, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_official_eval(
    dataset: str,
    *,
    gold: bool = False,
    predictions: Optional[str] = None,
    run_id: str,
    **kwargs,
):
    if not docker_available():
        raise RuntimeError("Docker is required to run official SWE-bench evaluation")
    cmd = build_eval_command(dataset, gold=gold, predictions=predictions, run_id=run_id, **kwargs)
    subprocess.run(cmd, check=True)


@dataclass
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    version: str
    ground_truth_files: Set[str] = field(default_factory=set)

    @classmethod
    def from_hf_record(cls, record: Dict[str, Any]) -> SWEBenchInstance:
        patch_str = record.get("patch", "")
        gt_files = cls.extract_patch_files(patch_str)

        f2p = record.get("FAIL_TO_PASS", [])
        if isinstance(f2p, str):
            try:
                f2p = json.loads(f2p)
            except Exception:
                f2p = [f2p]

        p2p = record.get("PASS_TO_PASS", [])
        if isinstance(p2p, str):
            try:
                p2p = json.loads(p2p)
            except Exception:
                p2p = [p2p]

        return cls(
            instance_id=record["instance_id"],
            repo=record["repo"],
            base_commit=record["base_commit"],
            problem_statement=record["problem_statement"],
            patch=patch_str,
            test_patch=record.get("test_patch", ""),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            version=record.get("version", ""),
            ground_truth_files=gt_files,
        )

    @staticmethod
    def extract_patch_files(patch_str: str) -> Set[str]:
        from .swebench_protocol import gold_files_from_patch

        return gold_files_from_patch(patch_str)


class RepoWorkspaceManager:
    """Clona e gerencia checkouts de repositórios em commits específicos."""

    def __init__(self, cache_root: str = ".basinrag/swebench_repos"):
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def prepare_repo(self, repo: str, base_commit: str) -> Path:
        repo_clean_name = repo.replace("/", "__")
        repo_dir = self.cache_root / repo_clean_name

        if not (repo_dir / ".git").exists():
            logger.info(f"Clonando repositório https://github.com/{repo}.git...")
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
                check=True,
                capture_output=True,
            )

        # Reset para estado limpo e checkout no base_commit
        subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True, capture_output=True)

        return repo_dir


class CodeRepoIngestor:
    """Ingere código-fonte Python excluindo testes e docs para evitar data contamination."""

    @staticmethod
    def ingest_codebase(rag: BasinRAG, repo_path: Path) -> int:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from ..indexer.condensation import node_layers
        from ..core.ids import make_node_id

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CODE_CHUNK_SIZE,
            chunk_overlap=CODE_CHUNK_OVERLAP,
            separators=list(CODE_CHUNK_SEPARATORS),
        )

        # Blindagem metodológica contra vazamento de dados:
        # Ignora testes e docs para avaliar puramente a capacidade de encontrar o bug no código real.
        skip_dirs = {".git", ".basinrag", "__pycache__", "tests", "testing", "test", "docs", ".tox", "venv", ".env"}

        code_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for file in files:
                if file.endswith(".py") and not file.startswith("test_") and not file.endswith("_test.py"):
                    code_files.append(Path(root) / file)

        logger.info(f"Ingerindo {len(code_files)} arquivos Python de {repo_path.name}...")

        all_texts = []
        chunk_metadata: list[dict[str, str | int]] = []

        for fpath in code_files:
            rel_path = fpath.relative_to(repo_path).as_posix()
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            file_splits = splitter.split_text(content)
            for idx, text in enumerate(file_splits):
                all_texts.append(text)
                chunk_metadata.append({
                    "source": rel_path,
                    "rel_path": rel_path,
                    "chunk_index": idx,
                    "text": text,
                })

        if not all_texts:
            return 0

        embeddings = rag.ingestor.encoder.encode(
            all_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        nodes = []
        for meta, emb in zip(chunk_metadata, embeddings):
            source = str(meta["source"])
            chunk_index = int(meta["chunk_index"])
            text = str(meta["text"])
            layers = node_layers(text)
            node_id = make_node_id(source, chunk_index, text)
            nodes.append({
                "id": node_id,
                "text": text,
                "embedding": emb,
                "source": source,
                "chunk_index": chunk_index,
                "l1": layers["l1"],
                "l2": layers["l2"],
                "metadata": {"file_path": meta["rel_path"]},
            })

        rag.engine.encoder_model = rag.config.encoder_model
        rag.engine.build_graph(nodes)
        rag.engine.partition_into_basins()
        rag._attach_bm25()
        rag.retriever = None

        return len(nodes)


class SWEBenchEvaluator:
    """Orquestrador de Avaliação do SWE-bench Lite para BasinRAG."""

    def __init__(self, workspace_root: str = ".basinrag/swebench_workspaces"):
        self.workspace = RepoWorkspaceManager(workspace_root)

    def load_dataset(
        self,
        split: str = "test",
        repo_filter: Optional[str] = None,
        limit: Optional[int] = None,
        limit_per_repo: Optional[int] = None,
    ) -> List[SWEBenchInstance]:
        logger.info(f"Carregando dataset '{DATASET_ID}' ({split})...")
        ds = load_dataset(DATASET_ID, split=split)

        repo_targets = set(r.strip() for r in repo_filter.split(",")) if repo_filter else None
        repo_counts: Dict[str, int] = {}
        instances = []
        for row in ds:
            inst = SWEBenchInstance.from_hf_record(row)
            if repo_targets and inst.repo not in repo_targets:
                continue
            if limit_per_repo:
                count = repo_counts.get(inst.repo, 0)
                if count >= limit_per_repo:
                    continue
                repo_counts[inst.repo] = count + 1
            instances.append(inst)
            if limit and len(instances) >= limit:
                break
        return instances

    def evaluate_retrieval(
        self,
        instances: List[SWEBenchInstance],
        top_k: int = 10,
        search_type: SearchType = "hybrid",
        storage_dir: str = ".basinrag/eval_temp",
        checkpoint_path: Optional[str] = "logs/swebench_100_results.jsonl",
        split: str = "test",
        encoder_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        use_rerank: bool = False,
        query_prompt: str = "Represent this sentence for searching relevant passages: ",
        ranking_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executa avaliação de Fault Localization / Retrieval com checkpoint incremental.
        """
        hit_1 = 0
        hit_5 = 0
        hit_10 = 0
        total_recall = 0.0
        total_mrr = 0.0
        valid_count = 0
        per_instance_results = []
        selected_instance_ids = {inst.instance_id for inst in instances}
        subset_id = _checkpoint_subset_id(instances)
        config = (
            BasinRAGConfig.from_env()
            if ranking_mode is None
            else BasinRAGConfig.from_env(ranking_mode=ranking_mode)
        )
        effective_ranking_mode = config.ranking_mode
        candidate_k = max(20, top_k)
        protocol_id = _checkpoint_protocol_id(
            split=split,
            top_k=top_k,
            candidate_k=candidate_k,
            search_type=search_type,
            encoder_model=encoder_model,
            reranker_model=reranker_model,
            use_rerank=use_rerank,
            query_prompt=query_prompt,
            ranking_mode=effective_ranking_mode,
        )

        completed_records: Dict[str, Dict[str, Any]] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            instance_id = rec.get("instance_id")
                            if (
                                instance_id in selected_instance_ids
                                and rec.get("subset_id") == subset_id
                                and rec.get("protocol_id") == protocol_id
                            ):
                                completed_records[instance_id] = rec
                        except Exception:
                            pass
            logger.info(f"Retomando do checkpoint: {len(completed_records)} instâncias já avaliadas.")

        for rec in completed_records.values():
            valid_count += 1
            if rec.get("hit@1"):
                hit_1 += 1
            if rec.get("hit@5"):
                hit_5 += 1
            if rec.get("hit@10"):
                hit_10 += 1
            total_recall += rec.get("recall", 0.0)
            total_mrr += rec.get("mrr", 0.0)
            per_instance_results.append(rec)

        pending_instances = [inst for inst in instances if inst.instance_id not in completed_records]
        logger.info(f"Instâncias pendentes para execução: {len(pending_instances)} de {len(instances)}")

        if not pending_instances and valid_count > 0:
            return {
                "instances_evaluated": valid_count,
                "hit@1": hit_1 / valid_count,
                "hit@5": hit_5 / valid_count,
                "hit@10": hit_10 / valid_count,
                f"recall@{top_k}": total_recall / valid_count,
                "mrr": total_mrr / valid_count,
                "details": per_instance_results,
            }

        repo_groups: Dict[str, List[SWEBenchInstance]] = {}
        for inst in pending_instances:
            key = f"{inst.repo}@{inst.base_commit}"
            repo_groups.setdefault(key, []).append(inst)

        checkpoint_f = None
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            checkpoint_f = open(checkpoint_path, "a", encoding="utf-8")

        try:
            for key, group in repo_groups.items():
                first = group[0]
                repo_dir = self.workspace.prepare_repo(first.repo, first.base_commit)

                rag = BasinRAG.create(
                    storage_dir=storage_dir,
                    encoder_model=encoder_model,
                    reranker_model=reranker_model,
                    use_rerank=use_rerank,
                    query_prompt=query_prompt,
                    ranking_mode=effective_ranking_mode,
                )
                n_chunks = CodeRepoIngestor.ingest_codebase(rag, repo_dir)
                logger.info(f"Índice construído: {n_chunks} chunks para {first.repo}@{first.base_commit[:8]}")

                for inst in group:
                    gt_files = inst.ground_truth_files
                    if not gt_files:
                        continue

                    docs = rag.query(inst.problem_statement, search_type=search_type, top_k=candidate_k)

                    retrieved_files = []
                    for doc in docs:
                        fpath = doc.metadata.get("file_path") or doc.metadata.get("source") or ""
                        norm_path = fpath.replace("\\", "/").strip()
                        if norm_path and norm_path not in retrieved_files:
                            retrieved_files.append(norm_path)

                    valid_count += 1

                    normalized_gt_files = {
                        _normalise_repo_path(gt) for gt in gt_files if _normalise_repo_path(gt)
                    }

                    def matches_gt(f: str) -> bool:
                        return _normalise_repo_path(f) in normalized_gt_files

                    is_hit_1 = any(matches_gt(f) for f in retrieved_files[:1])
                    is_hit_5 = any(matches_gt(f) for f in retrieved_files[:5])
                    is_hit_10 = any(matches_gt(f) for f in retrieved_files[:10])

                    if is_hit_1:
                        hit_1 += 1
                    if is_hit_5:
                        hit_5 += 1
                    if is_hit_10:
                        hit_10 += 1

                    retrieved_top_k = {
                        _normalise_repo_path(f)
                        for f in retrieved_files[:top_k]
                        if _normalise_repo_path(f)
                    }
                    found_gt = len(normalized_gt_files & retrieved_top_k)
                    recall = found_gt / float(len(normalized_gt_files)) if normalized_gt_files else 0.0
                    total_recall += recall

                    mrr = 0.0
                    for rank, retrieved_file in enumerate(retrieved_files[:top_k], 1):
                        if matches_gt(retrieved_file):
                            mrr = 1.0 / rank
                            break
                    total_mrr += mrr

                    record = {
                        "instance_id": inst.instance_id,
                        "repo": inst.repo,
                        "ground_truth_files": list(gt_files),
                        "subset_id": subset_id,
                        "protocol_id": protocol_id,
                        "protocol_revision": CHECKPOINT_PROTOCOL_REVISION,
                        "ranking_mode": effective_ranking_mode,
                        "candidate_k": candidate_k,
                        "chunking_protocol": {
                            "revision": CODE_CHUNKER_REVISION,
                            "size": CODE_CHUNK_SIZE,
                            "overlap": CODE_CHUNK_OVERLAP,
                            "embedding_tokenizer_policy": EMBEDDING_TOKENIZER_POLICY,
                        },
                        "retrieved_files_top5": retrieved_files[:5],
                        "hit@1": is_hit_1,
                        "hit@5": is_hit_5,
                        "hit@10": is_hit_10,
                        "recall": recall,
                        "mrr": mrr,
                    }
                    per_instance_results.append(record)

                    if checkpoint_f:
                        checkpoint_f.write(json.dumps(record) + "\n")
                        checkpoint_f.flush()

                    print(f"[{valid_count}/{len(instances)}] {inst.instance_id} ({inst.repo}) "
                          f"-> Hit@1: {hit_1/valid_count:.1%} | Hit@5: {hit_5/valid_count:.1%} | Hit@10: {hit_10/valid_count:.1%} | MRR: {total_mrr/valid_count:.3f}")
        finally:
            if checkpoint_f:
                checkpoint_f.close()

        if valid_count == 0:
            return {"instances_evaluated": 0}

        return {
            "instances_evaluated": valid_count,
            "hit@1": hit_1 / valid_count,
            "hit@5": hit_5 / valid_count,
            "hit@10": hit_10 / valid_count,
            f"recall@{top_k}": total_recall / valid_count,
            "mrr": total_mrr / valid_count,
            "details": per_instance_results,
        }

    def generate_predictions_jsonl(
        self,
        instances: List[SWEBenchInstance],
        output_file: str,
        model_name: str = "basinrag-context",
    ) -> None:
        """Exporta arquivo predictions.jsonl pronto para o harness oficial do SWE-bench."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for inst in instances:
                pred = {
                    "instance_id": inst.instance_id,
                    "model_patch": "",  # Preenchido após geração de patch por LLM
                    "model_name_or_path": model_name,
                }
                f.write(json.dumps(pred) + "\n")
        logger.info(f"Predições exportadas para {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BasinRAG SWE-bench Lite Evaluator")
    parser.add_argument("--repo", type=str, default=None, help="Filtrar por repositório ou lista separada por vírgula (ex: pallets/flask,psf/requests)")
    parser.add_argument("--limit", type=int, default=None, help="Número total de instâncias a avaliar")
    parser.add_argument("--limit-per-repo", type=int, default=None, help="Número de instâncias por repositório")
    parser.add_argument("--top_k", type=int, default=10, help="Top K arquivos recuperados")
    parser.add_argument("--search-type", type=str, default="hybrid", help="Estratégia de busca (hybrid, auto, local, global)")
    parser.add_argument(
        "--ranking-mode",
        choices=("hybrid_rrf", "experimental_topology"),
        default=None,
        help="Modo de ranking; por padrão usa BASINRAG_RANKING_MODE ou hybrid_rrf",
    )
    parser.add_argument("--split", type=str, default="test", help="Split do dataset (test)")
    parser.add_argument("--checkpoint", type=str, default="logs/swebench_fair_benchmark.jsonl", help="Caminho do arquivo de checkpoint jsonl")
    parser.add_argument("--inspect-only", action="store_true", help="Apenas carregar e inspecionar os dados")

    args = parser.parse_args()

    evaluator = SWEBenchEvaluator()
    instances = evaluator.load_dataset(
        split=args.split,
        repo_filter=args.repo,
        limit=args.limit,
        limit_per_repo=args.limit_per_repo,
    )
    print(f"\nCarregadas {len(instances)} instâncias do SWE-bench Lite.")

    if args.inspect_only:
        for idx, inst in enumerate(instances, 1):
            print(f"\n[{idx}] Instância: {inst.instance_id}")
            print(f"    Repo: {inst.repo} (commit: {inst.base_commit[:10]})")
            print(f"    Arquivos Modificados (Ground Truth): {inst.ground_truth_files}")
            print(f"    Problema: {inst.problem_statement[:150]}...")
            print(f"    Fail to Pass: {inst.fail_to_pass}")
        return

    print(f"\nExecutando Benchmark de Recuperação / Fault Localization com BasinRAG (search_type={args.search_type})...")
    metrics = evaluator.evaluate_retrieval(
        instances,
        top_k=args.top_k,
        search_type=args.search_type,
        checkpoint_path=args.checkpoint,
        split=args.split,
        ranking_mode=args.ranking_mode,
    )

    print("\n" + "=" * 60)
    print("DETALHAMENTO POR INSTÂNCIA (GROUND TRUTH VS BASINRAG)")
    print("=" * 60)
    for d in metrics.get("details", []):
        status = "ACERTOU (TOP-5)" if d.get("hit@5") else "FORA DO TOP-5"
        print(f"\n[{d['instance_id']}] Status: {status}")
        print(f"  Repo: {d['repo']}")
        print(f"  Arquivos do Bug (Ground Truth): {d['ground_truth_files']}")
        print("  Top-5 Arquivos Recuperados pelo BasinRAG:")
        for r_rank, r_file in enumerate(d.get("retrieved_files_top5", []), 1):
            is_match = " [MATCH!]" if _matches_gold_path(r_file, set(d["ground_truth_files"])) else ""
            print(f"    {r_rank}. {r_file}{is_match}")
        print(f"  Recall: {d.get('recall', 0.0) * 100:.1f}% | MRR: {d.get('mrr', 0.0):.3f}")

    print("\n" + "=" * 60)
    print("RESULTADOS CONSOLIDADOS DO BASINRAG NO SWE-BENCH LITE")
    print("=" * 60)
    print(f"Instâncias avaliadas: {metrics.get('instances_evaluated')}")
    print(f"Hit@1:  {metrics.get('hit@1', 0.0) * 100:.1f}%")
    print(f"Hit@5:  {metrics.get('hit@5', 0.0) * 100:.1f}%")
    print(f"Hit@10: {metrics.get('hit@10', 0.0) * 100:.1f}%")
    print(f"Recall@{args.top_k}: {metrics.get(f'recall@{args.top_k}', 0.0) * 100:.1f}%")
    print(f"MRR:    {metrics.get('mrr', 0.0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
