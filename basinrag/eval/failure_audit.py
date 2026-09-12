"""Stage-drop failure audit: where relevant docs fall out of the retrieval pipeline.

Reports per-query presence in BM25 / FAISS / RRF / hop / CE pools, plus optional
HNSW vs FlatIP miss counts. Reuses gate loaders and does not change decide().

Usage:
  python -m basinrag.eval.failure_audit
  python -m basinrag.eval.failure_audit --max-queries 50 --compare-flat
  python -m basinrag.eval.failure_audit --skip-scifact --longdoc-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

import numpy as np

from ..factory import BasinRAG, BasinRAGConfig
from ..retriever.fusion import apply_hop_prior, ranked_ids, weighted_rrf
from .gate import (
    QUERY_PROMPT,
    GateSearcher,
    collapse_to_docs,
    recall_at_k,
    write_json,
)
from .run_gate import (
    index_flat_docs,
    index_long_docs,
    load_qasper,
    load_scifact,
    maybe_reranker,
)


STAGES = ("bm25", "dense", "rrf", "hop", "ce")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stage_docs(engine, nids: Sequence[str], top_k: int) -> List[str]:
    hits = [{"id": nid} for nid in nids if nid in engine.graph]
    return collapse_to_docs(engine, hits, top_k)


def _hits_relevant(docs: Sequence[str], relevant: Set[str], k: int) -> bool:
    return bool(set(docs[:k]) & relevant)


def audit_query_stages(
    searcher: GateSearcher,
    query: str,
    relevant: Set[str],
    top_k: int = 10,
    candidate_k: Optional[int] = None,
    use_rerank: bool = True,
) -> Dict[str, Any]:
    """Trace one query through BM25 → dense → RRF → hop → optional CE."""
    ck = candidate_k if candidate_k is not None else max(50, top_k * 5)
    emb = searcher.encode(query)
    engine = searcher.engine
    hybrid = searcher.hybrid
    local = searcher.local

    bm25_ids: List[str] = []
    if searcher.bm25 is not None:
        bm25_ids = [nid for nid, _ in searcher.bm25.score(query, top_k=ck)]

    dense = local.dense_hits(emb, top_k=ck)
    dense_ids = [h["id"] for h in dense]

    rrf_scores = weighted_rrf(bm25_ids, dense_ids)
    rrf_order = ranked_ids(rrf_scores, ck)

    # Same hop BFS as HybridSearch (candidates only)
    from collections import deque

    all_candidate_ids = set(bm25_ids) | set(dense_ids)
    seeds = (set(dense_ids[:3]) | set(bm25_ids[:3])) & all_candidate_ids
    if not seeds:
        seeds = set(list(all_candidate_ids)[:3])
    hops: Dict[str, int] = {s: 0 for s in seeds}
    q_bfs = deque((s, 0) for s in seeds)
    while q_bfs:
        curr, d = q_bfs.popleft()
        if curr in engine.graph:
            for nbr in engine.graph.neighbors(curr):
                if nbr in all_candidate_ids and nbr not in hops:
                    hops[nbr] = d + 1
                    q_bfs.append((nbr, d + 1))

    hop_scores = apply_hop_prior(rrf_scores, hops, enabled=True, missing="penalty")
    hop_order = ranked_ids(hop_scores, ck)

    ce_order = list(hop_order)
    if use_rerank and searcher.reranker and hop_order:
        hits = []
        for nid in hop_order[: max(25, top_k)]:
            if nid not in engine.graph:
                continue
            hits.append({
                "id": nid,
                "text": engine.graph.nodes[nid].get("text", ""),
            })
        if hits:
            reranked = searcher._rerank(query, hits, top_k=min(len(hits), max(25, top_k)))
            ce_order = [h["id"] for h in reranked]

    stage_nids = {
        "bm25": bm25_ids,
        "dense": dense_ids,
        "rrf": rrf_order,
        "hop": hop_order,
        "ce": ce_order,
    }
    stage_hit = {}
    stage_recall = {}
    for name, nids in stage_nids.items():
        docs = _stage_docs(engine, nids, top_k)
        stage_hit[name] = _hits_relevant(docs, relevant, top_k)
        stage_recall[name] = recall_at_k(docs, relevant, top_k)

    # First stage that lost all relevant (drop after previous hit)
    drop_at = None
    prev_hit = False
    for name in STAGES:
        hit = stage_hit[name]
        if prev_hit and not hit:
            drop_at = name
            break
        prev_hit = hit or prev_hit
    if not any(stage_hit.values()):
        drop_at = "never_retrieved"

    return {
        "stage_hit": stage_hit,
        "stage_recall": stage_recall,
        "drop_at": drop_at,
        "n_relevant": len(relevant),
    }


def compare_hnsw_vs_flat(
    searcher: GateSearcher,
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    top_k: int = 10,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Count queries where FlatIP recovers a relevant that HNSW misses @top_k."""
    import faiss
    from ..core.vector_index import build_ip_index

    local = searcher.local
    if local._index is None or local._node_ids is None:
        return {"error": "no FAISS index", "hnsw_misses": 0, "n_queries": 0}

    vecs = []
    for nid in local._node_ids:
        emb = searcher.engine.graph.nodes[nid].get("embedding")
        if emb is None:
            return {"error": "missing embeddings", "hnsw_misses": 0, "n_queries": 0}
        vecs.append(emb)
    matrix = np.asarray(vecs, dtype=np.float32)
    flat = build_ip_index(matrix)  # may still be HNSW if N large — force Flat
    flat = faiss.IndexFlatIP(matrix.shape[1])
    faiss.normalize_L2(matrix)
    flat.add(matrix)

    usable = [(qid, queries[qid], qrels[qid]) for qid in queries if qid in qrels and qrels[qid]]
    if max_queries is not None:
        usable = usable[:max_queries]

    hnsw_miss = 0
    both_miss = 0
    flat_only = 0
    hnsw_hit = 0
    for qid, qtext, rel in usable:
        emb = searcher.encode(qtext)
        ck = max(50, top_k * 5)
        hnsw_hits = local.dense_hits(emb, top_k=ck)
        hnsw_docs = collapse_to_docs(searcher.engine, hnsw_hits, top_k)
        h_ok = bool(set(hnsw_docs) & rel)

        q = np.array([emb], dtype=np.float32)
        faiss.normalize_L2(q)
        D, I = flat.search(q, min(ck, flat.ntotal))
        flat_hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            nid = local._node_ids[int(idx)]
            flat_hits.append({"id": nid, "score": float(score)})
        flat_docs = collapse_to_docs(searcher.engine, flat_hits, top_k)
        f_ok = bool(set(flat_docs) & rel)

        if h_ok:
            hnsw_hit += 1
        elif f_ok:
            flat_only += 1
            hnsw_miss += 1
        else:
            both_miss += 1

    n = len(usable)
    return {
        "n_queries": n,
        "hnsw_hit": hnsw_hit,
        "hnsw_miss_flat_hit": flat_only,
        "both_miss": both_miss,
        "index_type": type(local._index).__name__,
        "n_vectors": int(local._index.ntotal),
    }


def aggregate_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_queries": 0}
    hit_rates = {s: sum(1 for r in rows if r["stage_hit"].get(s)) / n for s in STAGES}
    mean_recall = {
        s: sum(float(r["stage_recall"].get(s, 0.0)) for r in rows) / n for s in STAGES
    }
    drops = Counter(r.get("drop_at") or "none" for r in rows)
    return {
        "n_queries": n,
        "stage_hit_rate@10": hit_rates,
        "stage_mean_recall@10": mean_recall,
        "drop_heatmap": dict(drops),
    }


def run_corpus_audit(
    name: str,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    storage_dir: str,
    encoder: str,
    reranker_model: str,
    flat_index: bool,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    max_queries: Optional[int],
    compare_flat: bool,
    enable_rerank: bool,
) -> Dict[str, Any]:
    config = BasinRAGConfig(
        storage_dir=storage_dir,
        encoder_model=encoder,
        reranker_model=reranker_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    rag = BasinRAG(config)
    if flat_index:
        index_flat_docs(rag, corpus)
    else:
        index_long_docs(rag, corpus, chunk_size, chunk_overlap)

    reranker = maybe_reranker(enable_rerank, reranker_model) if enable_rerank else None
    searcher = GateSearcher(rag, reranker=reranker, query_prompt=QUERY_PROMPT)

    usable = [(qid, queries[qid], qrels[qid]) for qid in queries if qid in qrels and qrels[qid]]
    if max_queries is not None:
        usable = usable[:max_queries]

    rows = []
    for i, (qid, qtext, rel) in enumerate(usable, 1):
        if i == 1 or i == len(usable) or i % 50 == 0:
            print(f"  [{name}] audit {i}/{len(usable)}", flush=True)
        row = audit_query_stages(
            searcher, qtext, rel, top_k=top_k, use_rerank=bool(reranker)
        )
        row["qid"] = qid
        rows.append(row)

    out: Dict[str, Any] = {
        "corpus": name,
        "flat_index": flat_index,
        "summary": aggregate_audit(rows),
        "samples": [r for r in rows if r.get("drop_at")][:30],
    }
    if compare_flat:
        print(f"  [{name}] HNSW vs FlatIP...", flush=True)
        out["hnsw_vs_flat"] = compare_hnsw_vs_flat(
            searcher, queries, qrels, top_k=top_k, max_queries=max_queries
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="BasinRAG retrieval stage-drop audit")
    parser.add_argument("--output", type=str, default="results/failure_audit")
    parser.add_argument("--encoder", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--compare-flat", action="store_true")
    parser.add_argument("--with-rerank", action="store_true", help="Include CE stage (slower)")
    parser.add_argument("--skip-scifact", action="store_true")
    parser.add_argument("--longdoc-only", action="store_true")
    parser.add_argument("--skip-longdoc", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = _repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"protocol": {"query_prompt": QUERY_PROMPT, "top_k": args.top_k}}

    if not args.skip_scifact and not args.longdoc_only:
        print("[audit] SciFact...", flush=True)
        corpus, queries, qrels = load_scifact()
        report["scifact"] = run_corpus_audit(
            name="scifact",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            storage_dir=str(_repo_root() / ".basinrag" / "audit_scifact"),
            encoder=args.encoder,
            reranker_model=args.reranker,
            flat_index=True,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            max_queries=args.max_queries,
            compare_flat=args.compare_flat,
            enable_rerank=args.with_rerank,
        )
        write_json(out_dir / "scifact.json", report["scifact"])

    if not args.skip_longdoc:
        print("[audit] QASPER long-doc...", flush=True)
        corpus, queries, qrels = load_qasper(args.max_papers, args.max_queries)
        report["qasper"] = run_corpus_audit(
            name="qasper",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            storage_dir=str(_repo_root() / ".basinrag" / "audit_qasper"),
            encoder=args.encoder,
            reranker_model=args.reranker,
            flat_index=False,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            max_queries=args.max_queries,
            compare_flat=args.compare_flat,
            enable_rerank=args.with_rerank,
        )
        write_json(out_dir / "qasper.json", report["qasper"])

    write_json(out_dir / "summary.json", {
        "scifact": report.get("scifact", {}).get("summary"),
        "qasper": report.get("qasper", {}).get("summary"),
        "hnsw_vs_flat_scifact": report.get("scifact", {}).get("hnsw_vs_flat"),
        "hnsw_vs_flat_qasper": report.get("qasper", {}).get("hnsw_vs_flat"),
    })
    print(f"[audit] wrote {out_dir}", flush=True)
    print(json.dumps(report.get("scifact", {}).get("summary") or {}, indent=2))


if __name__ == "__main__":
    main()
