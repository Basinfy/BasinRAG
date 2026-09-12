# Fase 0 gate

Runner: `python -m basinrag.eval.run_gate`

- `scifact.json` — controle negativo (1 doc = 1 bacia)
- `decision.json` — protocolo, métricas, `DECISION=CONVERT_C`

QASPER via Hugging Face não carrega mais (dataset script). O corpus longo usou `ccdv/arxiv-summarization` (validation, streaming): 80 papers, mediana de 20 membros/bacia.
