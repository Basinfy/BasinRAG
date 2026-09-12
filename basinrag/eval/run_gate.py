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
    official_cache_reference,
    write_json,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scifact() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Load SciFact corpus/queries/qrels (test split)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for the gate (`pip install datasets`)") from exc

    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    query_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrel_ds = load_dataset("mteb/scifact", "default", split="test")

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


def load_qasper(max_papers: Optional[int], max_queries: Optional[int]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Long documents: one source per paper, questions retrieve the paper id."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for the gate (`pip install datasets`)") from exc

    try:
        ds = load_dataset("allenai/qasper", split="validation")
    except Exception as first_err:
        print(f"[gate] allenai/qasper failed ({first_err}); trying arxiv long-doc fallback")
        return load_arxiv_longdoc(max_papers, max_queries)
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
        print("[gate] QASPER empty after parse; trying arxiv long-doc fallback")
        return load_arxiv_longdoc(max_papers, max_queries)
    return corpus, queries, qrels


def load_arxiv_longdoc(
    max_papers: Optional[int],
    max_queries: Optional[int],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Full papers + abstract-as-query. Geometry is long-doc; task is retrieve-the-paper."""
    from datasets import load_dataset

    n = max_papers or 150
    ds = load_dataset("ccdv/arxiv-summarization", split="validation", streaming=True)
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


def index_flat_docs(rag: BasinRAG, corpus: Dict[str, str]) -> None:
    """One node per document (SciFact control: topology must be inert)."""
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
    rag._attach_bm25()
    rag.retriever = None
    rag.persistence.save_topology(rag.engine)


def index_long_docs(rag: BasinRAG, corpus: Dict[str, str], chunk_size: int, chunk_overlap: int) -> None:
    """Chunk each document under a shared source so basins can have size > 1."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    nodes: List[Dict[str, Any]] = []
    texts_acc: List[str] = []
    meta_acc: List[Tuple[str, int, str]] = []
    for doc_id, text in corpus.items():
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        for i, chunk in enumerate(chunks):
            texts_acc.append(chunk)
            meta_acc.append((doc_id, i, chunk))
    embeddings = rag.ingestor.encoder.encode(
        texts_acc, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    from ..core.ids import make_node_id
    for (doc_id, i, chunk), emb in zip(meta_acc, embeddings):
        layers = node_layers(chunk)
        nodes.append({
            "id": make_node_id(doc_id, i, chunk),
            "text": chunk,
            "embedding": emb,
            "source": doc_id,
            "chunk_index": i,
            "l1": layers["l1"],
            "l2": layers["l2"],
            "metadata": {"doc_id": doc_id},
        })
    rag.engine.encoder_model = rag.config.encoder_model
    rag.engine.build_graph(nodes)
    rag.engine.partition_into_basins()
    rag._attach_bm25()
    rag.retriever = None
    rag.persistence.save_topology(rag.engine)


def maybe_reranker(enabled: bool, model_name: str):
    if not enabled:
        return None
    from ..retriever.reranker import CrossEncoderReranker
    rr = CrossEncoderReranker(model_name=model_name)
    rr._load_model()
    if rr._model == "disabled":
        print("[gate] reranker disabled; hybrid_min_rerank and basinrag_full will skip")
        return None
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
) -> Dict[str, Any]:
    print(f"\n=== GATE {name}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels ===")
    config = BasinRAGConfig(
        encoder_model=encoder,
        reranker_model=reranker_model,
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
            index_long_docs(rag, corpus, chunk_size, chunk_overlap)
        else:
            index_flat_docs(rag, corpus)
        print(f"[gate] built index ({len(rag.engine.graph)} nodes)")

    diag = basin_diagnostics(rag.engine)
    print(f"[gate] basins: {diag}")
    reranker = maybe_reranker(enable_rerank, reranker_model)
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
    return {"basins": diag, "metrics": metrics}


def parse_args():
    p = argparse.ArgumentParser(description="BasinRAG Fase 0 scientific gate")
    p.add_argument("--output", type=str, default="results/gate")
    p.add_argument("--encoder", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--reranker", type=str, default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--systems", type=str, default=",".join(SYSTEMS))
    p.add_argument("--skip-rerank", action="store_true")
    p.add_argument("--skip-scifact", action="store_true")
    p.add_argument("--skip-longdoc", action="store_true")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--max-scifact-queries", type=int, default=None)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--max-long-queries", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=100)
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
            "reranker": args.reranker if enable_rerank else None,
            "split_policy": "SciFact test is a burned negative control; long-doc uses QASPER validation once",
        },
        "official_cache_reference": official_cache_reference(root),
    }

    scifact_metrics = {}
    long_metrics = None
    long_basins = None
    prev_scifact = out / "scifact.json"
    if args.skip_scifact and prev_scifact.exists():
        import json as _json
        payload["scifact"] = _json.loads(prev_scifact.read_text(encoding="utf-8"))
        scifact_metrics = payload["scifact"].get("metrics", {})

    if not args.skip_scifact:
        corpus, queries, qrels = load_scifact()
        queries, qrels = _limit_queries(queries, qrels, args.max_scifact_queries)
        scifact = run_corpus(
            "SciFact",
            str(root / ".basinrag" / "gate_scifact"),
            args.encoder,
            args.reranker,
            corpus,
            queries,
            qrels,
            systems,
            long_doc=False,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            enable_rerank=enable_rerank,
            force_reindex=args.force_reindex,
        )
        payload["scifact"] = scifact
        scifact_metrics = scifact["metrics"]
        write_json(out / "scifact.json", scifact)

    if not args.skip_longdoc:
        try:
            corpus, queries, qrels = load_qasper(args.max_papers, args.max_long_queries)
            queries, qrels = _limit_queries(queries, qrels, args.max_long_queries)
            # Isolate storage by corpus family so QASPER cache is never reused for arxiv fallback.
            sample_id = next(iter(corpus), "")
            longdoc_tag = "arxiv" if str(sample_id).startswith("arxiv") else "qasper"
            longdoc_storage = str(root / ".basinrag" / f"gate_{longdoc_tag}")
            longdoc = run_corpus(
                longdoc_tag.upper(),
                longdoc_storage,
                args.encoder,
                args.reranker,
                corpus,
                queries,
                qrels,
                systems,
                long_doc=True,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=args.top_k,
                enable_rerank=enable_rerank,
                force_reindex=args.force_reindex,
            )
            payload[longdoc_tag] = {"n_docs": len(corpus), **longdoc}
            # Keep legacy key for decide()/consumers that expect "qasper".
            payload["qasper"] = payload[longdoc_tag]
            long_metrics = longdoc["metrics"]
            long_basins = longdoc["basins"]
            write_json(out / f"{longdoc_tag}.json", payload[longdoc_tag])
            if longdoc_tag != "qasper":
                write_json(out / "qasper.json", payload["qasper"])
        except Exception as exc:
            payload["qasper_error"] = str(exc)
            print(f"[gate] long-doc failed: {exc}")

    decision = decide(scifact_metrics, long_metrics, long_basins)
    payload["decision"] = decision
    write_json(out / "decision.json", payload)
    print("\n==== DECISION ====")
    print(decision)
    print(f"wrote {out / 'decision.json'}")


if __name__ == "__main__":
    main()
