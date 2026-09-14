# Gate SciFact + QASPER

Runner: `python -m basinrag.eval.run_gate`

Documentação: [docs/EVAL.md](../../docs/EVAL.md). Os JSON deste diretório, se existirem localmente, são artefatos de run e **não** entram no Git.

- SciFact: controle flat (1 documento = 1 nó).
- QASPER: long-doc oficial (tarball v0.3). Não use dataset script Hugging Face nem ArXiv como substituto.
- Decisão de métricas (`decision.json`) exige as duas tarefas no mesmo commit, com provenance compatível.
- Ranking de produto: `hybrid_rrf` sem cross-encoder (`--skip-rerank`).
