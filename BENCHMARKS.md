# BasinRAG — Benchmarks

**Gate:** [`results/gate/decision.json`](results/gate/decision.json) (`DECISION=CONVERT_C`)  
**Encoder:** `BAAI/bge-base-en-v1.5`  
**Ranking:** híbrido BM25 + FAISS  
**Bacias:** mapa de briefing (vizinhança de contexto)

Relatório de recall: [`results/recall_report.md`](results/recall_report.md).

---

## Papel de cada componente

| Componente | Função |
| :--- | :--- |
| Híbrido BM25 + FAISS | Ranking em corpora de documentos |
| Bacias / hop / expansão local | Vizinhança e briefing para o LLM |
| Indexação sem LLM | Ingestão local sem extração de entidades |

---

## Ranking primário — BEIR-EN-small (MTEB, sem CE)

Protocolo congelado: split `test`, encoder `bge-base-en-v1.5` + prompt BGE, RRF BM25+FAISS, **sem rerank**, hop off em corpus flat. Métrica: **nDCG@10**. Não é o score overall `MTEB(eng, v2)`.

| Tarefa | nDCG@10 | Recall@10 | Artefacto |
| :--- | :---: | :---: | :--- |
| SciFact | 0.733 | 0.866 | `results/mteb_beir_arena/` |
| NFCorpus | 0.370 | 0.178 | idem |
| FiQA2018 | 0.343 | 0.423 | idem |
| ArguAna | 0.589 | 0.846 | idem |
| SCIDOCS | 0.192 | 0.198 | idem |
| **Média (5/5)** | **0.445** | | |

Gate SciFact `hybrid_min`: nDCG@10 0.734 / Recall@10 0.869 (dense puro 0.740 — o híbrido não ganha do encoder neste corpus).

```powershell
python -m basinrag.eval.run_mteb --no-rerank --tasks SciFact,NFCorpus,FiQA2018,ArguAna,SCIDOCS --output results/mteb_beir_arena
```

---

## Long-doc (evidence / passage)

QASPER no HuggingFace falha (script dataset). Fallback: `ccdv/arxiv-summarization`, query = abstract, ouro = excertos do corpo.

Escala 120 papers / 120 queries (`results/qasper_evidence_scale/report.json`):

| Config | Hit@10 | evidence_recall@10 |
| :--- | :---: | :---: |
| expand ON | 0.742 | 0.428 |
| expand OFF | 0.733 | 0.422 |
| Δ | +0.008 | **+0.006** |

Smoke n=40: Δ evidence = +0.008. Expand ajuda pouco; KPI não satura (ao contrário do paper-id Recall@10 = 1.0).

```powershell
python -m basinrag.eval.qasper_evidence --max-papers 120 --max-queries 200 --ablate-expand --output results/qasper_evidence_scale --storage-dir .basinrag/qasper_evidence_scale
```

---

## SWE-bench

Não é o board global (`% Resolved` + Docker). Sonda de localização, só no Lite **300** com Avg/Any/All Recall. `--limit 13` e n=21 (astropy+django) não se publicam como score.

```powershell
python -m basinrag.eval.swebench
```

---

## Defaults de produção

| Knob | Valor |
| :--- | :--- |
| Encoder | `BAAI/bge-base-en-v1.5` + query prompt BGE |
| `candidate_k` | `max(50, top_k*5)` |
| Índice vetorial | FlatIP se N ≤ 20 000; senão HNSW `efSearch=256` |
| CE em índice flat (path MTEB) | off por padrão |
| Hop `missing` | `neutral` (produção); `penalty` no gate |
| Chunk | 512 / overlap 128 |
| Reranker (quando ligado) | `BAAI/bge-reranker-v2-m3`, `max_length=512` |

---

## Reprodução

```powershell
python -m basinrag.eval.run_gate --skip-rerank
python -m basinrag.eval.run_mteb --no-rerank --tasks SciFact,NFCorpus,FiQA2018,ArguAna,SCIDOCS --output results/mteb_beir_arena
python -m basinrag.eval.qasper_evidence --max-papers 120 --max-queries 200 --ablate-expand --output results/qasper_evidence_scale --storage-dir .basinrag/qasper_evidence_scale
```
