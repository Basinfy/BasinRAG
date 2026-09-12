"""Fase 0 scientific gate: one harness, frozen protocol, pre-registered decision."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .metrics import ndcg_at_k, mrr_at_k, hit_rate_at_k
from ..retriever.prompts import QUERY_PROMPT as _SHARED_QUERY_PROMPT

SYSTEMS = (
    "encoder_pure",
    "bm25_pure",
    "hybrid_min",
    "hybrid_min_rerank",
    "hybrid_min_topo",
    "basinrag_full",
)

SCIFACT_DELTA_MAX = 0.005
LONGDOC_TOPO_GAIN_MIN = 0.01
LONGDOC_ENCODER_SLACK = 0.01
MIN_MEDIAN_BASIN = 3.0
MAX_SINGLETON_FRAC = 0.40

QUERY_PROMPT = _SHARED_QUERY_PROMPT


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def aggregate_metrics(
    qrels: Dict[str, Set[str]],
    results: Dict[str, List[str]],
    k: int = 10,
) -> Dict[str, float]:
    qids = [qid for qid, rel in qrels.items() if rel]
    if not qids:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@10": 0.0, "hit@10": 0.0, "n_queries": 0}
    ndcg = mrr = rec = hit = 0.0
    for qid in qids:
        ranked = results.get(qid, [])
        rel = qrels[qid]
        ndcg += ndcg_at_k(ranked, rel, k)
        mrr += mrr_at_k(ranked, rel, k)
        rec += recall_at_k(ranked, rel, k)
        hit += hit_rate_at_k(ranked, rel, k)
    n = len(qids)
    return {
        "ndcg@10": ndcg / n,
        "mrr@10": mrr / n,
        "recall@10": rec / n,
        "hit@10": hit / n,
        "n_queries": n,
    }


def node_doc_id(engine, nid: str) -> str:
    data = engine.graph.nodes[nid]
    meta = data.get("metadata") or {}
    return str(meta.get("doc_id") or data.get("source") or nid)


def collapse_to_docs(engine, hits: Sequence[Dict[str, Any]], top_k: int) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for hit in hits:
        doc_id = node_doc_id(engine, hit["id"])
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) >= top_k:
            break
    return out


def basin_diagnostics(engine) -> Dict[str, Any]:
    sizes = [b.rho_tree.number_of_nodes() for b in engine.basins.values()] if engine.basins else []
    hops: List[int] = []
    for nid, data in engine.graph.nodes(data=True):
        hops.append(int(data.get("hops", 0)))
    sources = {data.get("source") for _, data in engine.graph.nodes(data=True)}
    singleton = sum(1 for s in sizes if s <= 1)
    n_basins = len(sizes)
    return {
        "n_nodes": engine.graph.number_of_nodes(),
        "n_basins": n_basins,
        "n_sources": len(sources),
        "median_members": float(statistics.median(sizes)) if sizes else 0.0,
        "mean_members": float(statistics.mean(sizes)) if sizes else 0.0,
        "max_members": int(max(sizes) if sizes else 0),
        "singleton_frac": (singleton / n_basins) if n_basins else 1.0,
        "mean_hops": float(statistics.mean(hops)) if hops else 0.0,
        "size_histogram": dict(Counter(sizes)),
        "gate_geometry_ok": bool(
            sizes
            and statistics.median(sizes) >= MIN_MEDIAN_BASIN
            and (singleton / n_basins) < MAX_SINGLETON_FRAC
        ),
    }


def decide(
    scifact: Dict[str, Dict[str, float]],
    longdoc: Optional[Dict[str, Dict[str, float]]],
    longdoc_basins: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pre-registered kill switch. Do not edit thresholds after seeing numbers."""
    warnings: List[str] = []
    scifact_hybrid = _ndcg(scifact, "hybrid_min")
    scifact_topo = _ndcg(scifact, "hybrid_min_topo")
    scifact_delta = None
    if scifact_hybrid is not None and scifact_topo is not None:
        scifact_delta = abs(scifact_topo - scifact_hybrid)
        if scifact_delta >= SCIFACT_DELTA_MAX and scifact_topo > scifact_hybrid:
            warnings.append(
                "SciFact control failed: topology gained "
                f"{scifact_topo - scifact_hybrid:.4f} where h(v) should be 0 (leakage)."
            )

    if not longdoc or not longdoc_basins:
        return {
            "DECISION": "GATE_INVALID",
            "reason": "long-doc corpus missing; gate did not run",
            "scifact_delta_ndcg@10": scifact_delta,
            "warnings": warnings,
        }

    if not longdoc_basins.get("gate_geometry_ok"):
        return {
            "DECISION": "GATE_INVALID",
            "reason": (
                f"median members/basin={longdoc_basins.get('median_members')} "
                f"(need >={MIN_MEDIAN_BASIN}) singleton_frac="
                f"{longdoc_basins.get('singleton_frac')}"
            ),
            "scifact_delta_ndcg@10": scifact_delta,
            "warnings": warnings,
        }

    long_hybrid = _ndcg(longdoc, "hybrid_min")
    long_topo = _ndcg(longdoc, "hybrid_min_topo")
    long_encoder = _ndcg(longdoc, "encoder_pure")
    if long_hybrid is None or long_topo is None or long_encoder is None:
        return {
            "DECISION": "GATE_INVALID",
            "reason": "long-doc ablation incomplete (need encoder_pure, hybrid_min, hybrid_min_topo)",
            "scifact_delta_ndcg@10": scifact_delta,
            "warnings": warnings,
        }

    topo_gain = long_topo - long_hybrid
    candidates = [
        _ndcg(longdoc, name)
        for name in ("hybrid_min_topo", "basinrag_full")
        if _ndcg(longdoc, name) is not None
    ]
    best_basin = max(candidates) if candidates else long_topo
    continue_b = topo_gain >= LONGDOC_TOPO_GAIN_MIN and best_basin >= (long_encoder - LONGDOC_ENCODER_SLACK)
    return {
        "DECISION": "CONTINUE_B" if continue_b else "CONVERT_C",
        "reason": (
            f"longdoc topo-hybrid={topo_gain:+.4f} (need >={LONGDOC_TOPO_GAIN_MIN}), "
            f"best_basin={best_basin:.4f} vs encoder={long_encoder:.4f} "
            f"(slack {LONGDOC_ENCODER_SLACK})"
        ),
        "scifact_delta_ndcg@10": scifact_delta,
        "longdoc_topo_gain_ndcg@10": topo_gain,
        "longdoc_best_basin_ndcg@10": best_basin,
        "longdoc_encoder_ndcg@10": long_encoder,
        "warnings": warnings,
    }


def _ndcg(block: Dict[str, Dict[str, float]], system: str) -> Optional[float]:
    row = block.get(system)
    if not row or row.get("skipped"):
        return None
    val = row.get("ndcg@10")
    if val is None:
        return None
    return float(val)


def official_cache_reference(repo_root: Path) -> Dict[str, Any]:
    """Numbers already verified from the in-repo MTEB cache — not re-run here."""
    return {
        "source": "results/mteb/remote + results/mteb/results (verified 2026-09-11)",
        "SciFact": {
            "BAAI/bge-base-en-v1.5": {"ndcg@10": 0.74345, "recall@10": 0.87422},
            "intfloat/e5-base-v2": {"ndcg@10": 0.71944},
            "BAAI/bge-small-en-v1.5": {"ndcg@10": 0.71273},
            "mteb/baseline-bm25s": {"ndcg@10": 0.68736},
            "BasinRAG-1.0.4-published": {"ndcg@10": 0.64995, "recall@10": 0.73078, "mrr@10": 0.629763},
        },
    }


class GateSearcher:
    def __init__(self, rag, reranker=None, query_prompt: str = QUERY_PROMPT):
        rag._ensure_retriever(search_type="hybrid", top_k=50)
        self.rag = rag
        self.engine = rag.engine
        self.hybrid = rag.retriever._hybrid
        self.local = rag.retriever._local
        self.encoder = rag.ingestor.encoder
        self.bm25 = rag.engine.bm25
        self.reranker = reranker
        self.query_prompt = query_prompt
        self._q_cache: Dict[str, Any] = {}

    def encode(self, query: str):
        key = query
        if key not in self._q_cache:
            text = f"{self.query_prompt}{query}" if self.query_prompt else query
            emb = self.encoder.encode(text)
            import numpy as np
            emb = np.asarray(emb, dtype=np.float32)
            n = float((emb ** 2).sum()) ** 0.5
            if n > 0:
                emb = emb / n
            self._q_cache[key] = emb
        return self._q_cache[key]

    def search(self, query: str, system: str, top_k: int = 10) -> List[str]:
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system}")
        use_hop = system in ("hybrid_min_topo", "basinrag_full")
        use_rerank = system in ("hybrid_min_rerank", "basinrag_full")
        emb = self.encode(query)
        candidate_k = max(50, top_k * 5)

        if system == "encoder_pure":
            hits = self.local.dense_hits(emb, top_k=candidate_k)
        elif system == "bm25_pure":
            hits = []
            if self.bm25 is not None:
                for nid, score in self.bm25.score(query, top_k=candidate_k):
                    if nid in self.engine.graph:
                        hits.append({
                            "id": nid,
                            "text": self.engine.graph.nodes[nid].get("text", ""),
                            "score": float(score),
                        })
        else:
            # Gate protocol: hop missing=penalty; no graph expand (avoids kNN leakage on flat SciFact).
            hits = self.hybrid.search_nodes(
                query,
                emb,
                top_k=candidate_k,
                use_hop_prior=use_hop,
                use_confidence_gate=False,
                hop_missing="penalty",
                expand_graph=False,
            )

        if use_rerank and self.reranker and hits:
            hits = self._rerank(query, hits, top_k=min(len(hits), max(25, top_k)))
        return collapse_to_docs(self.engine, hits, top_k)

    def _rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        passages = [h.get("text") or self.engine.graph.nodes[h["id"]].get("text", "") for h in hits]
        scores = self.reranker.predict_scores(query, passages)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        return [h for h, _ in ranked[:top_k]]


def evaluate_system(
    searcher: GateSearcher,
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    system: str,
    top_k: int = 10,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    results: Dict[str, List[str]] = {}
    usable = {qid: qrels[qid] for qid in queries if qid in qrels and qrels[qid]}
    total = len(usable)
    for i, qid in enumerate(usable, 1):
        if i == 1 or i == total or i % 50 == 0:
            print(f"  [{system}] {i}/{total}", flush=True)
        results[qid] = searcher.search(queries[qid], system, top_k=top_k)
    return aggregate_metrics(usable, results, k=top_k), results


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def join_title_text(title: str, body: str) -> str:
    t = (title or "").strip()
    x = (body or "").strip()
    if t and x:
        return f"{t} {x}"
    return t or x
