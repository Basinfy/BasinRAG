# Avaliação

Como medir o BasinRAG sem misturar runs nem publicar média incompleta. Protocolo longo: [BENCHMARKS.md](../BENCHMARKS.md). Ranking de produção: [RANKING.md](RANKING.md).

**Não há pontuação oficial de release** até SciFact **e** QASPER passarem no mesmo commit, com provenance compatível. Artefatos em `results/` são históricos ou locais e estão gitignored (exceto este guia em `results/gate/README.md`).

## Gate (SciFact + QASPER)

```powershell
$run = "results/runs/gate-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m basinrag.eval.run_gate --output $run --skip-rerank
```

- SciFact é o controle **flat** (1 doc = 1 nó). QASPER é o long-doc oficial (tarball v0.3, não script Hugging Face).
- `--skip-rerank` é o caminho de produto. Cross-encoder no gate é ablação (`hybrid_min_rerank`), não o default.
- ArXiv long-doc é experimento separado e **não** substitui QASPER.
- Cada run usa diretório novo. Não mescle JSONs de commits diferentes.

## MTEB

```powershell
$run = "results/runs/mteb-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m basinrag.eval.run_mteb --no-rerank --tasks SciFact,NFCorpus,FiQA2018,ArguAna,SCIDOCS --output $run
```

Se alguma tarefa falhar, reporte o status parcial. **Não** publique a média como resultado de cinco tarefas. SWE-bench de localização de arquivo **não** é `% Resolved`.

## Medições pontuais (não são release)

Encoder em todos os números abaixo: `BAAI/bge-base-en-v1.5` (`dd9f42942e0729b6c53632f3c23b0e801f236569`). Sem hop no ranking de produção.

### SciFact (flat, 300 queries, MTEB test)

| Sistema | nDCG@10 | Nota |
|---|---|---|
| Híbrido `hybrid_rrf`, sem CE | 0.74389 | Commit `c1da17e`, artefato `results/runs/mteb-20260913-173459`. Só SciFact. |

### QASPER (long-doc, 1005 queries, validation oficial)

Sentence-window (256/32), RM3, BM25@10, rescue m=2.

| Sistema | nDCG@10 | Recall@10 | Nota |
|---|---|---|---|
| Híbrido sem CE | 0.430 | 0.542 | Melhor ranking medido neste corpus |
| Híbrido + CE (`bge-reranker-v2-m3`) | 0.421 | 0.545 | Ablação `results/runs/ablation-qasper-ce-20260914-103858`; CE piorou nDCG |

Por isso o produto deixa o cross-encoder **opt-in**.

## Claims permitidos em PRs

- Cite commit, comando, diretório de run, encoder/revisão, corpus/split e se o CE estava ligado.
- Não misture SciFact com QASPER numa média.
- Não trate paper LaTeX, `docs/archive/` ou `docs/BENCHMARK_COMPARISON_pt.md` como score vigente.
