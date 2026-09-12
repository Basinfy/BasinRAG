# BasinRAG — Benchmarks

Fonte de verdade: o gate pré-registrado em [`results/gate/decision.json`](results/gate/decision.json).

**Decisão:** `DECISION=CONVERT_C`. A topologia **não** é reivindicada como ganho de retrieval. O produto é um híbrido BM25+FAISS; as bacias existem como mapa de briefing (vizinhança de contexto).

## SciFact (300 queries de teste, harness congelado)

| Sistema | nDCG@10 |
| :--- | :---: |
| Encoder isolado (`BAAI/bge-base-en-v1.5`) | **0.741** |
| Híbrido BasinRAG (BM25+FAISS) | **0.727** |
| Pacote publicado (BasinRAG 1.0.4) | **0.650** |

No SciFact o corpus é 1 documento = 1 nó, então $h(v)=0$ e a geometria de bacia não opera. O recorte de 50 queries e comparações GraphRAG antigas **não** são scores oficiais.

Reprodução:

```powershell
python -m basinrag.eval.run_gate
```

## SWE-bench Lite

Localização de arquivos em issues reais permanece um experimento separado (Hit@1 / Hit@10 em repositórios filtrados). Não misturar com o claim de nDCG do SciFact.

## Fora de escopo

Não submeter MTEB; não republicar 0,650 como evidência topológica; não tunar SciFact.
