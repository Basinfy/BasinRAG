# Relatório: melhoria de recall BasinRAG

**Data:** 2026-09-12  
**Protocolo:** pós-mudanças (prompt BGE, `candidate_k`, FlatIP até 20k, CE off em flat, hop `neutral`, expand de grafo)  
**Artefatos:** `results/gate_recall_eval/`, `results/mteb_recall_eval/`, `results/failure_audit_smoke/`, `results/qasper_evidence_eval/`

## Veredito

O alvo do plano (**Recall@10 ≥ 0.85** no path publicado/MTEB sem CE) foi **atingido**.

| Score | Antes (1.0.4 publicado) | Agora | Δ |
|---|---:|---:|---:|
| SciFact Recall@10 | 0.731 | **0.866** (MTEB `--no-rerank`) | **+0.135** |
| SciFact nDCG@10 | 0.650 | **0.733** | **+0.083** |

Gate `hybrid_min` confirma o mesmo patamar (Recall@10 **0.869**). `DECISION` continua **CONVERT_C** — topologia não é claim de nDCG.

## 1. Gate (`--skip-rerank`)

### SciFact (300 queries, índice flat reutilizado)

| Sistema | nDCG@10 | Recall@10 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| encoder_pure | 0.740 | 0.874 | 0.883 | 0.703 |
| bm25_pure | 0.677 | 0.813 | 0.837 | 0.640 |
| hybrid_min | 0.734 | 0.869 | 0.883 | 0.699 |
| hybrid_min_topo | 0.741 | 0.882 | 0.893 | 0.702 |

- Δ topo−hybrid nDCG = **+0.007** → warning de leakage (limiar 0.005).
- hybrid Recall subiu vs gate anterior (0.864 → 0.869).

### Long-doc (arxiv fallback, 40 papers — QASPER script indisponível no `datasets` novo)

Geometria OK (`median_members=21`, `singleton_frac≈0.03`). Todos os sistemas: **Recall@10 = 1.0** (tarefa abstract→paper saturada). Topo gain = 0 → **CONVERT_C**.

## 2. MTEB SciFact (`--no-rerank`)

| Métrica | Valor |
|---|---:|
| nDCG@10 (main_score) | 0.7333 |
| Recall@10 | **0.8661** |
| MRR@10 | 0.6993 |
| Hit@10 | 0.8800 |
| Recall@100 | 0.9667 |

Rerank=False, flat=True, prompt BGE ligado. Arquivo: `results/mteb_recall_eval/results/Basinfy__BasinRAG/1.0.4/SciFact.json`.

## 3. Failure audit (SciFact smoke, n=40)

| Estágio | Hit@10 | Mean Recall@10 |
|---|---:|---:|
| BM25 | 0.90 | 0.86 |
| Dense | 0.925 | 0.908 |
| RRF | 0.925 | 0.901 |
| Hop | 0.925 | 0.908 |
| CE (off) | 0.925 | 0.908 |

Drops: 2 never_retrieved, 1 dense, 1 RRF. Índice **FlatIP** (N=5183 &lt; 20k) → 0 misses HNSW.

## 4. KPI evidence/passage (arxiv, 40 papers)

| Config | Hit@10 | evidence_recall@10 |
|---|---:|---:|
| expand ON | 0.750 | 0.417 |
| expand OFF | 0.725 | 0.408 |
| Δ | +0.025 | **+0.008** |

KPI não saturado (ao contrário do paper-id). Expand ajuda pouco nesta amostra.

## Conclusões

1. **Pipeline, não topologia**, fechou o gap do pacote publicado.
2. Manter **CE off em flat** no path MTEB/publicação.
3. Não tunar SciFact; leakage topo permanece sinal de arestas kNN.
4. Feito: evidence/passage n=120 (Δ expand +0.006) e BEIR-EN-small média nDCG@10 0.445. SWE-bench não é o claim global.

## Comandos reproduzidos

```powershell
python -m basinrag.eval.run_gate --skip-rerank --output results/gate_recall_eval
python -m basinrag.eval.run_mteb --tasks SciFact --no-rerank --output results/mteb_recall_eval
python -m basinrag.eval.failure_audit --max-queries 40 --skip-longdoc --compare-flat
python -m basinrag.eval.qasper_evidence --max-papers 40 --ablate-expand
```
