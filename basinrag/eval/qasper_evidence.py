"""QASPER evidence / passage-level KPI (non-saturated long-doc recall).

Paper-id retrieval on QASPER validation saturates at Recall@10≈1.0. This harness
retrieves **chunks** and scores hit against gold evidence spans (token overlap),
so topology expand / hop can be measured.

Usage:
  python -m basinrag.eval.qasper_evidence --max-papers 40 --max-queries 80
  python -m basinrag.eval.qasper_evidence --ablate-expand
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch.distributed._tensor as _t
    sys.modules.setdefault("torch.distributed.tensor", _t)
except Exception:
    pass

from ..factory import BasinRAG, BasinRAGConfig
from .gate import QUERY_PROMPT, GateSearcher, write_json
from .run_gate import index_long_docs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def evidence_overlap(chunk: str, evidence_texts: List[str], min_jaccard: float = 0.15) -> bool:
    """True if chunk overlaps any gold evidence span enough to count as a hit."""
    ctoks = tokenize(chunk)
    if not ctoks:
        return False
    for ev in evidence_texts:
        etoks = tokenize(ev)
        if not etoks:
            continue
        inter = len(ctoks & etoks)
        if inter == 0:
            continue
        jacc = inter / len(ctoks | etoks)
        # Also accept high containment of evidence inside chunk
        contain = inter / max(1, len(etoks))
        if jacc >= min_jaccard or contain >= 0.5:
            return True
    return False


def _mid_excerpts(article: str, n: int = 3, span: int = 400) -> List[str]:
    """Synthetic evidence spans from the body (not the abstract) for arxiv fallback."""
    text = (article or "").strip()
    if len(text) < span * 2:
        return [text] if text else []
    # Skip the opening (often abstract-like); sample from the middle third.
    start = len(text) // 3
    end = min(len(text), start + max(span * n * 2, span))
    window = text[start:end]
    out: List[str] = []
    step = max(span, len(window) // max(1, n))
    for i in range(n):
        a = i * step
        chunk = window[a : a + span].strip()
        if len(chunk) > 80:
            out.append(chunk)
    return out


def load_arxiv_passage_evidence(
    max_papers: Optional[int],
    max_queries: Optional[int],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Fallback KPI when QASPER script datasets are unavailable."""
    from datasets import load_dataset

    n = max_papers or 40
    ds = load_dataset("ccdv/arxiv-summarization", split="validation", streaming=True)
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    evidence: Dict[str, List[str]] = {}
    for i, row in enumerate(ds):
        if len(corpus) >= n:
            break
        paper_id = str(row.get("id") or row.get("article_id") or f"arxiv-{i}")
        article = str(row.get("article") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        if len(article) < 2000 or len(abstract) < 80:
            continue
        if len(article) > 40000:
            article = article[:40000]
        spans = _mid_excerpts(article)
        if not spans:
            continue
        corpus[paper_id] = article
        if max_queries is not None and len(queries) >= max_queries:
            continue
        qid = f"{paper_id}-abs"
        queries[qid] = abstract
        evidence[qid] = spans
    if len(queries) < 10:
        raise RuntimeError(
            f"arxiv passage evidence too small ({len(corpus)} docs, {len(queries)} queries)"
        )
    print(
        f"[qasper_evidence] arxiv passage fallback: {len(corpus)} papers, {len(queries)} queries",
        flush=True,
    )
    return corpus, queries, evidence


def load_qasper_evidence(
    max_papers: Optional[int],
    max_queries: Optional[int],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Corpus = paper full text; qrels = list of evidence strings per query."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required (`pip install datasets`)") from exc

    try:
        try:
            ds = load_dataset("allenai/qasper", split="validation")
        except TypeError:
            ds = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
        except Exception as first_err:
            print(f"[qasper_evidence] allenai/qasper failed ({first_err}); arxiv fallback", flush=True)
            return load_arxiv_passage_evidence(max_papers, max_queries)
    except Exception as err:
        print(f"[qasper_evidence] allenai/qasper failed ({err}); arxiv fallback", flush=True)
        return load_arxiv_passage_evidence(max_papers, max_queries)

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    evidence: Dict[str, List[str]] = {}

    n_q = 0
    for i, row in enumerate(ds):
        if max_papers is not None and i >= max_papers:
            break
        paper_id = str(row.get("id") or f"paper-{i}")
        parts = []
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        if title or abstract:
            parts.append(f"{title}\n{abstract}".strip())
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
        answers = list(qas.get("answers") or [])
        for qi, qtext in enumerate(qtexts):
            if max_queries is not None and n_q >= max_queries:
                break
            qid = f"{paper_id}::q{qi}"
            ev_texts: List[str] = []
            ans_block = answers[qi] if qi < len(answers) else None
            if isinstance(ans_block, dict):
                for ann in ans_block.get("answer") or []:
                    if not isinstance(ann, dict):
                        continue
                    for ev in ann.get("evidence") or []:
                        if isinstance(ev, str) and ev.strip():
                            ev_texts.append(ev.strip())
                    for ev in ann.get("highlighted_evidence") or []:
                        if isinstance(ev, str) and ev.strip():
                            ev_texts.append(ev.strip())
            if not ev_texts:
                continue
            queries[qid] = str(qtext)
            evidence[qid] = ev_texts
            n_q += 1
        if max_queries is not None and n_q >= max_queries:
            break

    if not queries:
        print("[qasper_evidence] QASPER empty after parse; arxiv fallback", flush=True)
        return load_arxiv_passage_evidence(max_papers, max_queries)
    return corpus, queries, evidence


def evaluate_passage_recall(
    searcher: GateSearcher,
    queries: Dict[str, str],
    evidence: Dict[str, List[str]],
    top_k: int = 10,
    expand_graph: bool = True,
) -> Dict[str, float]:
    hits = 0
    recall_sum = 0.0
    n = 0
    for i, (qid, qtext) in enumerate(queries.items(), 1):
        gold = evidence.get(qid) or []
        if not gold:
            continue
        n += 1
        if i == 1 or i == len(queries) or i % 25 == 0:
            print(f"  [qasper_evidence] {i}/{len(queries)}", flush=True)
        emb = searcher.encode(qtext)
        ck = max(50, top_k * 5)
        ranked = searcher.hybrid.search_nodes(
            qtext,
            emb,
            top_k=ck,
            use_hop_prior=True,
            use_confidence_gate=False,
            hop_missing="neutral",
            expand_graph=expand_graph,
        )
        chunks = [h.get("text") or "" for h in ranked[:top_k]]
        matched = [c for c in chunks if evidence_overlap(c, gold)]
        # Treat each matched chunk as retrieving the query's evidence set (binary hit + soft recall)
        hit = 1.0 if matched else 0.0
        # Soft recall: fraction of gold evidence strings hit by any top chunk
        covered = sum(
            1 for ev in gold if any(evidence_overlap(c, [ev]) for c in chunks)
        )
        soft = covered / max(1, len(gold))
        hits += hit
        recall_sum += soft
    if n == 0:
        return {"hit@10": 0.0, "evidence_recall@10": 0.0, "n_queries": 0}
    return {
        "hit@10": hits / n,
        "evidence_recall@10": recall_sum / n,
        "n_queries": n,
    }


def main():
    parser = argparse.ArgumentParser(description="QASPER passage-level evidence recall KPI")
    parser.add_argument("--output", type=str, default="results/qasper_evidence")
    parser.add_argument("--encoder", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--max-queries", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ablate-expand", action="store_true", help="Also run expand_graph=False")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = _repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[qasper_evidence] loading...", flush=True)
    corpus, queries, evidence = load_qasper_evidence(args.max_papers, args.max_queries)
    print(f"[qasper_evidence] papers={len(corpus)} queries={len(queries)}", flush=True)

    config = BasinRAGConfig(
        storage_dir=str(_repo_root() / ".basinrag" / "qasper_evidence"),
        encoder_model=args.encoder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    rag = BasinRAG(config)
    index_long_docs(rag, corpus, args.chunk_size, args.chunk_overlap)
    searcher = GateSearcher(rag, reranker=None, query_prompt=QUERY_PROMPT)

    report: Dict[str, Any] = {
        "kpi": "QASPER evidence/passage recall (non-saturated)",
        "n_papers": len(corpus),
        "n_queries": len(queries),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    print("[qasper_evidence] expand_graph=ON", flush=True)
    report["expand_on"] = evaluate_passage_recall(
        searcher, queries, evidence, top_k=args.top_k, expand_graph=True
    )
    if args.ablate_expand:
        print("[qasper_evidence] expand_graph=OFF", flush=True)
        report["expand_off"] = evaluate_passage_recall(
            searcher, queries, evidence, top_k=args.top_k, expand_graph=False
        )
        on = report["expand_on"].get("evidence_recall@10", 0.0)
        off = report["expand_off"].get("evidence_recall@10", 0.0)
        report["expand_delta_evidence_recall@10"] = on - off

    write_json(out_dir / "report.json", report)
    print(report)
    print(f"[qasper_evidence] wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
