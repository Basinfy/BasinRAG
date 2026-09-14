"""Fase 0 runner: SciFact (negative control) + long-doc ablation + DECISION=."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Patch PyTorch DTensor before heavy imports
try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

from ..factory import BasinRAG, BasinRAGConfig
from ..indexer.condensation import node_layers
from .gate import (
    SYSTEMS,
    GateSearcher,
    QUERY_PROMPT,
    basin_diagnostics,
    decide,
    evaluate_system,
    join_title_text,
    write_json,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scifact(revision: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Load SciFact corpus/queries/qrels (test split)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for the gate (`pip install datasets`)") from exc

    if not revision:
        raise RuntimeError("Informe um commit imutável em --scifact-revision para validar o gate")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus", revision=revision)
    query_ds = load_dataset("mteb/scifact", "queries", split="queries", revision=revision)
    qrel_ds = load_dataset("mteb/scifact", "default", split="test", revision=revision)

    corpus: Dict[str, str] = {}
    for row in corpus_ds:
        did = str(row.get("_id") or row.get("id"))
        corpus[did] = join_title_text(row.get("title") or "", row.get("text") or "")

    queries: Dict[str, str] = {}
    for row in query_ds:
        qid = str(row.get("_id") or row.get("id"))
        queries[qid] = str(row.get("text") or "")

    qrels: Dict[str, Set[str]] = {}
    for row in qrel_ds:
        if int(row.get("score") or 0) <= 0:
            continue
        qid = str(row.get("query-id") or row.get("query_id"))
        did = str(row.get("corpus-id") or row.get("corpus_id"))
        qrels.setdefault(qid, set()).add(did)
    return corpus, queries, qrels


QASPER_OFFICIAL_V03_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_OFFICIAL_V03_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
QASPER_OFFICIAL_V03_DEV = "qasper-dev-v0.3.json"


def _qasper_rows_from_official(papers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert official QASPER v0.3 JSON objects into the Hugging Face script row shape."""
    rows: List[Dict[str, Any]] = []
    for paper_id, paper in papers.items():
        if not isinstance(paper, dict):
            continue
        sections = paper.get("full_text") or []
        names: List[str] = []
        paragraphs: List[List[str]] = []
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                names.append(str(section.get("section_name") or ""))
                paras = section.get("paragraphs") or []
                paragraphs.append([str(p) for p in paras] if isinstance(paras, list) else [])
        qas = paper.get("qas") or []
        questions: List[str] = []
        question_ids: List[Optional[str]] = []
        if isinstance(qas, list):
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                questions.append(str(qa.get("question") or ""))
                question_ids.append(qa.get("question_id"))
        rows.append(
            {
                "id": str(paper_id),
                "title": paper.get("title") or "",
                "abstract": paper.get("abstract") or "",
                "full_text": {"section_name": names, "paragraphs": paragraphs},
                "qas": {"question": questions, "question_id": question_ids},
            }
        )
    return rows


def _load_official_qasper_v03(cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load QASPER validation from the canonical v0.3 tarball (HF scripts are gone)."""
    import hashlib
    import json
    import tarfile
    import urllib.request

    cache = cache_dir or (Path.home() / ".cache" / "basinrag")
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / "qasper-train-dev-v0.3.tgz"
    if not tarball.exists() or hashlib.sha256(tarball.read_bytes()).hexdigest() != QASPER_OFFICIAL_V03_SHA256:
        urllib.request.urlretrieve(QASPER_OFFICIAL_V03_URL, tarball)
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        if digest != QASPER_OFFICIAL_V03_SHA256:
            tarball.unlink(missing_ok=True)
            raise RuntimeError(f"checksum QASPER v0.3 inválido: {digest}")
    with tarfile.open(tarball, "r:gz") as archive:
        member = next((m for m in archive.getmembers() if m.name.endswith(QASPER_OFFICIAL_V03_DEV)), None)
        if member is None:
            raise RuntimeError("qasper-dev-v0.3.json ausente no tarball oficial")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("falha ao ler qasper-dev-v0.3.json do tarball oficial")
        papers = json.load(extracted)
    if not isinstance(papers, dict):
        raise RuntimeError("JSON oficial do QASPER v0.3 não é um objeto de papers")
    return _qasper_rows_from_official(papers)


def load_qasper(max_papers: Optional[int], max_queries: Optional[int], revision: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Long documents: one source per paper, questions retrieve the paper id."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for the gate (`pip install datasets`)") from exc

    if not revision:
        raise RuntimeError("Informe um commit imutável em --qasper-revision para validar o gate")
    load_qasper.source = "huggingface"
    try:
        ds = load_dataset("allenai/qasper", split="validation", revision=revision)
    except Exception as first_err:
        try:
            ds = _load_official_qasper_v03()
            load_qasper.source = "official_v0.3_json"
        except Exception as official_err:
            raise RuntimeError(
                "QASPER validation indisponível; o gate não pode usar fallback "
                f"arxiv ({first_err}); tarball oficial v0.3 também falhou: {official_err}"
            ) from official_err
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Set[str]] = {}

    for i, row in enumerate(ds):
        if max_papers is not None and i >= max_papers:
            break
        paper_id = str(row.get("id") or f"paper-{i}")
        parts = [join_title_text(row.get("title") or "", row.get("abstract") or "")]
        full = row.get("full_text") or {}
        names = list(full.get("section_name") or [])
        paragraphs = list(full.get("paragraphs") or [])
        for name, paras in zip(names, paragraphs):
            body = "\n".join(p for p in (paras or []) if p)
            if body.strip():
                parts.append(f"## {name}\n{body}" if name else body)
        text = "\n\n".join(p for p in parts if p and p.strip())
        if not text.strip():
            continue
        corpus[paper_id] = text

        qas = row.get("qas") or {}
        qtexts = list(qas.get("question") or [])
        qids = list(qas.get("question_id") or [])
        for qi, qtext in enumerate(qtexts):
            if not qtext:
                continue
            qid = str(qids[qi] if qi < len(qids) and qids[qi] else f"{paper_id}-q{qi}")
            if max_queries is not None and len(queries) >= max_queries:
                break
            queries[qid] = str(qtext)
            qrels.setdefault(qid, set()).add(paper_id)
        if max_queries is not None and len(queries) >= max_queries:
            continue
    if not corpus or not queries:
        raise RuntimeError("QASPER validation ficou vazio após o parse; o gate é inválido")
    return corpus, queries, qrels


def load_arxiv_longdoc(
    max_papers: Optional[int],
    max_queries: Optional[int],
    revision: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Full papers + abstract-as-query. Geometry is long-doc; task is retrieve-the-paper."""
    from datasets import load_dataset

    n = max_papers or 150
    if not revision:
        raise RuntimeError("Informe um commit imutável em --arxiv-revision para o experimento")
    ds = load_dataset("ccdv/arxiv-summarization", split="validation", streaming=True, revision=revision)
    article_key, abstract_key = "article", "abstract"

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Set[str]] = {}
    for i, row in enumerate(ds):
        if len(corpus) >= n:
            break
        paper_id = str(row.get("id") or row.get("article_id") or f"arxiv-{i}")
        article = str(row.get(article_key) or "").strip()
        abstract = str(row.get(abstract_key) or "").strip()
        if len(article) < 2000 or len(abstract) < 80:
            continue
        if len(article) > 40000:
            article = article[:40000]
        corpus[paper_id] = article
        qid = f"{paper_id}-abs"
        if max_queries is not None and len(queries) >= max_queries:
            continue
        queries[qid] = abstract
        qrels[qid] = {paper_id}
    if len(corpus) < 30 or len(queries) < 30:
        raise RuntimeError(
            f"arxiv long-doc fallback too small ({len(corpus)} docs, {len(queries)} queries)"
        )
    print(f"[gate] arxiv long-doc fallback: {len(corpus)} papers, {len(queries)} queries")
    return corpus, queries, qrels


def index_flat_docs(rag: BasinRAG, corpus: Dict[str, str], reranker_enabled: bool = False) -> None:
    """One node per document (SciFact control: topology must be inert)."""
    expected_build_id = rag.persistence.current_build_id()
    texts = list(corpus.values())
    ids = list(corpus.keys())
    embeddings = rag.ingestor.encoder.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    nodes = []
    for doc_id, text, emb in zip(ids, texts, embeddings):
        layers = node_layers(text)
        nodes.append({
            "id": str(doc_id),
            "text": text,
            "embedding": emb,
            "source": str(doc_id),
            "chunk_index": 0,
            "l1": layers["l1"],
            "l2": layers["l2"],
            "metadata": {"doc_id": str(doc_id)},
        })
    rag.engine.encoder_model = rag.config.encoder_model
    rag.engine.build_graph(nodes)
    rag.engine.partition_into_basins()
    rag.engine.build_id = expected_build_id
    _set_gate_manifest(rag, chunk_size=1, chunk_overlap=0, reranker_enabled=reranker_enabled)
    rag._attach_bm25()
    rag.retriever = None
    if not rag.persistence.save_topology(rag.engine):
        raise RuntimeError("Gate não conseguiu publicar o snapshot SciFact")


def index_long_docs(rag: BasinRAG, corpus: Dict[str, str], chunk_size: int, chunk_overlap: int, reranker_enabled: bool = False) -> None:
    """Sentence-window leaves under a shared source; SciFact stays in index_flat_docs."""
    from ..core.ids import make_node_id
    from ..indexer.ingestor import SENTENCE_WINDOW_RADIUS, split_sentence_leaves

    expected_build_id = rag.persistence.current_build_id()
    nodes: List[Dict[str, Any]] = []
    texts_acc: List[str] = []
    meta_acc: List[Tuple[str, int, str, int]] = []
    for doc_id, text in corpus.items():
        chunks = [c for c in split_sentence_leaves(text, chunk_size, chunk_overlap) if c.strip()]
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            texts_acc.append(chunk)
            meta_acc.append((doc_id, i, chunk, n_chunks))
    if texts_acc:
        embeddings = rag.ingestor.encoder.encode(
            texts_acc, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
    else:
        embeddings = []
    for (doc_id, i, chunk, n_chunks), emb in zip(meta_acc, embeddings):
        layers = node_layers(chunk)
        nodes.append({
            "id": make_node_id(doc_id, i, chunk),
            "text": chunk,
            "embedding": emb,
            "source": doc_id,
            "chunk_index": i,
            "l1": layers["l1"],
            "l2": layers["l2"],
            "metadata": {
                "doc_id": doc_id,
                "role": "child",
                "parent_span": [
                    max(0, i - SENTENCE_WINDOW_RADIUS),
                    min(n_chunks - 1, i + SENTENCE_WINDOW_RADIUS),
                ],
                "parent_doc": doc_id,
            },
        })
    rag.engine.encoder_model = rag.config.encoder_model
    rag.engine.build_graph(nodes)
    rag.engine.partition_into_basins()
    rag.engine.build_id = expected_build_id
    _set_gate_manifest(rag, chunk_size=chunk_size, chunk_overlap=chunk_overlap, reranker_enabled=reranker_enabled)
    rag._attach_bm25()
    rag.retriever = None
    if not rag.persistence.save_topology(rag.engine):
        raise RuntimeError("Gate não conseguiu publicar o snapshot long-doc")


def _set_gate_manifest(rag: BasinRAG, *, chunk_size: int, chunk_overlap: int, reranker_enabled: bool = False) -> None:
    from ..model_revisions import resolve_huggingface_revision
    tokenizer = getattr(rag.ingestor.splitter, "tokenizer", None)
    revision = resolve_huggingface_revision(
        rag.config.encoder_model,
        rag.config.encoder_revision or rag.ingestor.model_revision,
    )
    source_chunks = [
        {"source": data.get("source", ""), "id": node_id}
        for node_id, data in rag.engine.graph.nodes(data=True)
    ]
    rag.engine.index_metadata = {
        "format_version": 3,
        "encoder_model": rag.config.encoder_model,
        "encoder_revision": revision,
        "tokenizer": getattr(tokenizer, "name_or_path", None) or rag.config.encoder_model,
        "tokenizer_revision": revision,
        "reranker_model": rag.config.reranker_model if reranker_enabled else None,
        "reranker_revision": (
            resolve_huggingface_revision(rag.config.reranker_model, rag.config.reranker_revision)
            if reranker_enabled else "disabled"
        ),
        "chunking_mode": "characters",
        "chunk_policy_version": 1,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_size_tokens": None,
        "chunk_overlap_tokens": None,
        "sources": BasinRAG._source_manifest(source_chunks),
        "source_file_hashes": {},
        "ranking_mode": rag.config.ranking_mode,
    }


def maybe_reranker(enabled: bool, model_name: str, revision: Optional[str] = None):
    if not enabled:
        return None
    from ..retriever.reranker import CrossEncoderReranker
    rr = CrossEncoderReranker(model_name=model_name, revision=revision)
    rr._load_model()
    return rr


def run_corpus(
    name: str,
    storage_dir: str,
    encoder: str,
    reranker_model: str,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    systems: List[str],
    long_doc: bool,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    enable_rerank: bool,
    force_reindex: bool,
    encoder_revision: Optional[str] = None,
    reranker_revision: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    sampled: bool = False,
) -> Dict[str, Any]:
    print(f"\n=== GATE {name}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels ===")
    from ..model_revisions import resolve_huggingface_revision
    if enable_rerank:
        reranker_revision = resolve_huggingface_revision(reranker_model, reranker_revision)
    else:
        reranker_revision = "disabled"
    config = BasinRAGConfig(
        encoder_model=encoder,
        encoder_revision=encoder_revision,
        reranker_model=reranker_model,
        reranker_revision=reranker_revision if enable_rerank else None,
        use_rerank=False,
        chunk_size=1 if not long_doc else chunk_size,
        chunk_overlap=0 if not long_doc else chunk_overlap,
        storage_dir=storage_dir,
        search_type="hybrid",
    )
    rag = BasinRAG(config)
    loaded = False
    if not force_reindex:
        loaded = rag.load() and len(rag.engine.graph) > 0
        if loaded:
            rag._attach_bm25()
            print(f"[gate] loaded index ({len(rag.engine.graph)} nodes) from {storage_dir}")
    if not loaded:
        if long_doc:
            index_long_docs(rag, corpus, chunk_size, chunk_overlap, reranker_enabled=enable_rerank)
        else:
            index_flat_docs(rag, corpus, reranker_enabled=enable_rerank)
        print(f"[gate] built index ({len(rag.engine.graph)} nodes)")

    diag = basin_diagnostics(rag.engine)
    print(f"[gate] basins: {diag}")
    reranker = maybe_reranker(enable_rerank, reranker_model, reranker_revision)
    searcher = GateSearcher(rag, reranker=reranker, query_prompt=QUERY_PROMPT)

    metrics: Dict[str, Any] = {}
    for system in systems:
        needs_rerank = system in ("hybrid_min_rerank", "basinrag_full")
        if needs_rerank and reranker is None:
            metrics[system] = {"skipped": True, "reason": "reranker unavailable"}
            continue
        print(f"[gate] system={system}")
        row, _ = evaluate_system(searcher, queries, qrels, system, top_k=top_k)
        metrics[system] = row
        print(f"       {row}")
    index_meta = rag.engine.index_metadata or {}
    return {
        "basins": diag,
        "metrics": metrics,
        "provenance": {
            "dataset": "arxiv_longdoc" if name.lower() == "arxiv_longdoc" else name,
            "split": "validation" if name.upper() == "QASPER" else ("validation" if name.lower() == "arxiv_longdoc" else "test"),
            "dataset_revision": dataset_revision,
            "complete": not sampled,
            "sampled": sampled,
            "encoder": encoder,
            "encoder_revision": index_meta.get("encoder_revision"),
            "reranker": reranker_model if enable_rerank else None,
            "reranker_revision": reranker_revision,
            "query_prompt": QUERY_PROMPT,
            "chunk_policy": {"mode": "characters", "size": 1 if not long_doc else chunk_size, "overlap": 0 if not long_doc else chunk_overlap},
            "protocol_version": "gate-v3",
            "n_corpus": len(corpus),
            "n_queries": len(queries),
            "n_qrels": len(qrels),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="BasinRAG Fase 0 scientific gate")
    p.add_argument("--output", type=str, default="results/gate")
    p.add_argument("--encoder", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--encoder-revision", type=str, default=None)
    p.add_argument("--reranker", type=str, default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--reranker-revision", type=str, default=None)
    p.add_argument("--scifact-revision", type=str, default=None, help="Commit imutável do dataset mteb/scifact")
    p.add_argument("--qasper-revision", type=str, default=None, help="Commit imutável do dataset allenai/qasper")
    p.add_argument("--arxiv-revision", type=str, default=None, help="Commit imutável para experimento arxiv_longdoc")
    p.add_argument("--run-arxiv-experiment", action="store_true", help="Executa arXiv isolado; nunca alimenta a decisão rotulada QASPER")
    p.add_argument("--systems", type=str, default=",".join(SYSTEMS))
    p.add_argument("--skip-rerank", action="store_true")
    p.add_argument("--skip-scifact", action="store_true")
    p.add_argument("--skip-longdoc", action="store_true")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--max-scifact-queries", type=int, default=None)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--max-long-queries", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--chunk-overlap", type=int, default=32)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def _limit_queries(
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    limit: Optional[int],
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    if not limit:
        return queries, qrels
    keep = [qid for qid in queries if qid in qrels][:limit]
    return {qid: queries[qid] for qid in keep}, {qid: qrels[qid] for qid in keep}


def main():
    args = parse_args()
    root = _repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"O diretório de saída precisa ser novo/vazio para esta execução: {out}. "
            "Escolha outro --output; retomadas não são permitidas sem identidade integral validada."
        )
    index_root = out / "indices"
    index_root.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    enable_rerank = not args.skip_rerank

    payload: Dict[str, Any] = {
        "protocol": {
            "title_boost": 1,
            "score": "monotonic_raw_cross_encoder",
            "hop_missing": "penalty",
            "bm25_field": "text",
            "query_prompt": QUERY_PROMPT,
            "encoder": args.encoder,
            "encoder_revision": args.encoder_revision,
            "reranker": args.reranker if enable_rerank else None,
            "reranker_revision": args.reranker_revision if enable_rerank else None,
            "scifact_revision": args.scifact_revision,
            "qasper_revision": args.qasper_revision,
            "chunk_policy": {"mode": "characters", "longdoc_size": args.chunk_size, "longdoc_overlap": args.chunk_overlap},
            "split_policy": "SciFact test is a burned negative control; long-doc uses QASPER validation once",
        },
    }

    invalid_reasons = []
    sampled = any(value is not None for value in (args.max_scifact_queries, args.max_papers, args.max_long_queries))
    if sampled:
        invalid_reasons.append("limites de amostra ativos: gate parcial não pode decidir")
    scifact_metrics = {}
    scifact_provenance = None
    scifact_complete = False
    qasper_metrics = None
    qasper_basins = None
    qasper_provenance = None
    qasper_complete = False

    if args.skip_scifact:
        invalid_reasons.append("SciFact foi desativado por --skip-scifact")
    else:
        try:
            corpus, queries, qrels = load_scifact(args.scifact_revision)
            queries, qrels = _limit_queries(queries, qrels, args.max_scifact_queries)
            scifact = run_corpus(
                "SciFact", str(index_root / "scifact"), args.encoder, args.reranker,
                corpus, queries, qrels, systems, long_doc=False,
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                top_k=args.top_k, enable_rerank=enable_rerank,
                force_reindex=args.force_reindex,
                encoder_revision=args.encoder_revision,
                reranker_revision=args.reranker_revision,
                dataset_revision=args.scifact_revision,
                sampled=sampled,
            )
            payload["scifact"] = scifact
            scifact_metrics = scifact["metrics"]
            scifact_provenance = scifact["provenance"]
            control_rows = (scifact_metrics.get("hybrid_min"), scifact_metrics.get("hybrid_min_topo"))
            expected_queries = sum(1 for qid in queries if qrels.get(qid))
            scifact_complete = (
                not sampled
                and all(row and not row.get("skipped") and row.get("n_queries") == expected_queries for row in control_rows)
            )
            if not scifact_complete:
                invalid_reasons.append("SciFact completo ou seus sistemas de controle estão ausentes")
            write_json(out / "scifact.json", scifact)
        except Exception as exc:
            payload["scifact_error"] = str(exc)
            invalid_reasons.append(f"SciFact indisponível: {exc}")

    if args.skip_longdoc:
        invalid_reasons.append("QASPER foi desativado por --skip-longdoc")
    else:
        try:
            corpus, queries, qrels = load_qasper(args.max_papers, args.max_long_queries, args.qasper_revision)
            payload["protocol"]["qasper_source"] = getattr(load_qasper, "source", None)
            queries, qrels = _limit_queries(queries, qrels, args.max_long_queries)
            qasper = run_corpus(
                "QASPER", str(index_root / "qasper"), args.encoder, args.reranker,
                corpus, queries, qrels, systems, long_doc=True,
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                top_k=args.top_k, enable_rerank=enable_rerank,
                force_reindex=args.force_reindex,
                encoder_revision=args.encoder_revision,
                reranker_revision=args.reranker_revision,
                dataset_revision=args.qasper_revision,
                sampled=sampled,
            )
            payload["qasper"] = {"n_docs": len(corpus), **qasper}
            qasper_metrics = qasper["metrics"]
            qasper_basins = qasper["basins"]
            qasper_provenance = qasper["provenance"]
            control = qasper_metrics.get("hybrid_min")
            expected_queries = sum(1 for qid in queries if qrels.get(qid))
            qasper_complete = bool(
                not sampled and control and not control.get("skipped")
                and control.get("n_queries") == expected_queries
            )
            if not qasper_complete:
                invalid_reasons.append("QASPER completo ou seu resultado de controle está ausente")
            write_json(out / "qasper.json", payload["qasper"])
        except Exception as exc:
            payload["qasper_error"] = str(exc)
            invalid_reasons.append(f"QASPER indisponível: {exc}")
            print(f"[gate] QASPER failed: {exc}")

    if args.run_arxiv_experiment:
        try:
            arxiv_corpus, arxiv_queries, arxiv_qrels = load_arxiv_longdoc(
                args.max_papers, args.max_long_queries, args.arxiv_revision
            )
            arxiv = run_corpus(
                "arxiv_longdoc", str(index_root / "arxiv_longdoc"), args.encoder, args.reranker,
                arxiv_corpus, arxiv_queries, arxiv_qrels, systems, long_doc=True,
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                top_k=args.top_k, enable_rerank=enable_rerank,
                force_reindex=args.force_reindex,
                encoder_revision=args.encoder_revision,
                reranker_revision=args.reranker_revision,
                dataset_revision=args.arxiv_revision,
                sampled=sampled,
            )
            payload["arxiv_longdoc"] = {"n_docs": len(arxiv_corpus), **arxiv}
            write_json(out / "arxiv_longdoc.json", payload["arxiv_longdoc"])
        except Exception as exc:
            payload["arxiv_longdoc_error"] = str(exc)

    decision = decide(
        scifact_metrics, qasper_metrics, qasper_basins,
        scifact_complete=scifact_complete,
        qasper_complete=qasper_complete,
        scifact_provenance=scifact_provenance,
        qasper_provenance=qasper_provenance,
        invalid_reasons=invalid_reasons,
    )
    payload["decision"] = decision
    write_json(out / "decision.json", payload)
    print("\n==== DECISION ====")
    print(decision)
    print(f"wrote {out / 'decision.json'}")


if __name__ == "__main__":
    main()
