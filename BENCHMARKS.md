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

## SciFact (híbrido)

Protocolo do gate: prompt BGE, sem cross-encoder no path flat.

| Métrica | Gate `hybrid_min` | MTEB `--no-rerank` |
| :--- | :---: | :---: |
| nDCG@10 | **0.734** | **0.733** |
| Recall@10 | **0.869** | **0.866** |
| Hit@10 | 0.883 | 0.880 |

Artefatos: `results/gate/`, `results/mteb_recall_eval/`.

---

## Long-doc

Para expansão de grafo / chunking, use **evidence / passage recall**:

```powershell
python -m basinrag.eval.qasper_evidence --max-papers 40 --max-queries 80 --ablate-expand
```

---

## SWE-bench

Localização de arquivo em issues de código (Hit@k) — bateria a reexecutar e publicar aqui quando estiver estável:

```powershell
python -m basinrag.eval.swebench --limit 13
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
python -m basinrag.eval.run_mteb --no-rerank --tasks SciFact
python -m basinrag.eval.qasper_evidence --max-papers 40 --max-queries 80 --ablate-expand
```
