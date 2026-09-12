# Auditoria BasinRAG 1.0.4

Data: 11 de setembro de 2026  
Escopo: núcleo topológico, indexer, retriever, eval/benchmarks, API/CLI (parcial), testes (parcial), docs vs. código, higiene.  
Veredito de tese: revisor de conferência + PhD de arquitetura (Fase 0 do plano de execução).

---

## Veredito

**O projeto não faz sentido como paper, biblioteca SOTA ou produto no estado atual.**

A matemática das bacias é válida e está implementada. A hipótese empírica — *partição em bacias + prior `exp(-λh)` melhora recuperação* — **não foi testada** no SciFact: cada documento vira `source` própria com um chunk, `h(v) = 0` para todos, e o prior topológico é identidade. Os números publicados medem BM25 + FAISS + reranker.

Como retriever, o sistema está **refutado pelos dados do próprio repositório**:

| Sistema | nDCG@10 SciFact (test, 300 queries) |
|---|---:|
| `BAAI/bge-base-en-v1.5` sozinho (encoder default) | **0,74345** |
| `intfloat/e5-base-v2` sozinho | 0,719 |
| `BAAI/bge-small-en-v1.5` sozinho | 0,713 |
| BasinRAG `mteb_fase3` (pós-tuning no test set) | 0,716 |
| BM25 puro (`mteb/baseline-bm25s`) | 0,687 |
| **BasinRAG 1.0.4 publicado** | **0,650** |

Fontes: `results/mteb/results/Basinfy__BasinRAG/1.0.4/SciFact.json` e o cache oficial MTEB em `results/mteb/remote/results/`.

**Decisão estratégica:** manter a tese só até ela ser testável (ramo B), com morte pré-registrada para C (briefing/produto, sem claim de nDCG). O gate está em `basinrag/eval/run_gate.py`. Sem `DECISION=` em `results/gate/decision.json`, não há paper, leaderboard nem rewrite de README.

Um revisor de EMNLP/SIGIR **rejeitaria** o manuscrito atual — não major revision. A ablação de λ no SciFact é incompatível com `h ≡ 0`. A baseline GraphRAG do repo é IDF lexical sem embeddings.

---

## O que está bom

- `detect_attractors` em `basinrag/core/functional_graph.py` é DFS iterativo de 3 estados, O(V), com tie-break determinístico em ciclos (`min(cycle)`).
- `reverse_hops` é BFS reverso correto. A convergência de φ é trivial (out-degree ≤ 1, V finito).
- Fusão híbrida usa RRF por *rank* (`basinrag/retriever/fusion.py`), não soma de scores incomparáveis (cosseno vs BM25).
- PPR local conserva massa e trata nós dangling.
- IDF do BM25 usa a variante Lucene (sempre positiva). IDs content-addressed com SHA-256.
- Persistência sem `pickle`; `np.load(..., allow_pickle=False)`. Invalidação de BM25 stale por `build_id`.
- Métricas em `basinrag/eval/metrics.py` estão corretas (nDCG binário canônico). Os números do `BENCHMARKS.md` reproduzem a partir dos JSON brutos — não há fabricação.
- Blindagem anti-contaminação no SWE-bench (exclui `tests/`, `test_*.py`).

---

## Problemas

Severidade: CRÍTICO / ALTO / MÉDIO / BAIXO.

### Ciência e benchmarks

| Sev. | Onde | Problema |
|---|---|---|
| CRÍTICO | `eval/mteb_wrapper.py` ~88–99 | Cada doc SciFact é `source=doc_id`, `chunk_index=0`. Topologia inativa. |
| CRÍTICO | `results/mteb_fase*` | Nove avaliações no test split em ~3,5 h, nDCG 0,650→0,716. Tuning no conjunto de teste. |
| CRÍTICO | `eval/graphrag_baseline.py` | Baseline sem embeddings. O “+142,5%” mede presença de encoder. |
| CRÍTICO | `eval/mteb_wrapper.py` 176–183 | Score identidade em [0,1] e sigmoide fora: **não monotônico**, inverte rankings. Pipeline de eval ≠ `brief()`. |
| ALTO | `run_mteb.py` 58–59 | Cache só por encoder, sem tarefa. `--tasks SciFact,NFCorpus` contamina corpora. |
| ALTO | `mteb_wrapper.py` 63–74 | Title-boost 3× foge do protocolo BEIR/MTEB. |
| ALTO | `mteb_wrapper.py` 223–242 | ModelMeta falso: `embed_dim=384` (bge-base=768), 118M params (ignora reranker ~568M), `license="mit"` vs Apache-2.0. |
| ALTO | `BENCHMARKS.md` | 0,771 (50 queries, outro pipeline) ao lado de 0,650 (300 queries) como oficiais. SWE-bench n=13. |
| ALTO | `pyproject.toml` | `mteb`, `beir`, `datasets`, `spacy` usados e não declarados. |

### Núcleo e persistência

| Sev. | Onde | Problema |
|---|---|---|
| CRÍTICO | `persistence.py` + `topology.py` 48–49 | `safe_replace_dir` move o diretório com SQLite aberto. No Windows cai em `copytree(..., dirs_exist_ok=True)` e **mescla** bacias antigas. |
| CRÍTICO | `persistence.py` 151 | `basin_id` vira nome de arquivo sem sanitizar (path traversal via IDs de eval). |
| ALTO | `ARCHITECTURE.md` §6 | Promete substituição atômica; o código não usa ponteiro + `os.replace`. |
| ALTO | `topology.py` 173 | `DiskKVStore` materializado inteiro em RAM — anula “out-of-core”. |
| ALTO | `kv_store.py` | `LIMIT/OFFSET` sem `ORDER BY`: O(n²) e ordem instável. |
| ALTO | `vector_index.py` | HNSW acima de 4096 (SciFact=5183). Build all-pairs é O(N²·d), não O(N). |
| MÉDIO | `protocols.py` | Zero imports; contratos divergem da API real. |
| MÉDIO | `functional_graph.py` 124–129 | `compute_trapping_bounds` / `is_peripheral` inalcançáveis (`max_hops=50` > tamanho de seção). |

### Indexer

| Sev. | Onde | Problema |
|---|---|---|
| CRÍTICO | `factory.ingest` → `reset()` | Reingerir um arquivo apaga o índice inteiro. Sem incremental. |
| CRÍTICO | `factory.py` 81–88 vs `hybrid_search.py` 24–30 | BM25 indexa `l1+text` num caminho e só `text` no outro. **Corrigido no freeze da Fase 0** (ambos usam `text`). |
| CRÍTICO | `indexer/bm25.py` | Stemmer do documento ≠ stemmer da query em corpus misto → recall zero. |
| ALTO | `ingestor.py` 116–125 | Fallback `latin-1` nunca falha (mojibake). `chunk_size`/`overlap` da config são ignorados. |
| ALTO | `summarizer.py` | Resumo vazio marcado `l3_source='llm'` (nunca re- tentado); `save_topology` síncrono no event loop; texto ingerido cru no prompt. |

### Retriever

| Sev. | Onde | Problema |
|---|---|---|
| CRÍTICO | `retriever/base.py` 92–106 | Realinhamento pós-rerank por **texto** como chave: troca `doc_id` em duplicatas. |
| CRÍTICO | `retriever/router.py` | Centroides são estado de **classe**; o primeiro encoder no processo vence. |
| ALTO | `fusion.py` (antes do freeze) | `hops.get(nid, 0)` dava bônus máximo a nós inalcançáveis. **Corrigido:** missing hop = penalidade. |
| ALTO | `factory.py` 27–32 | `hybrid_alpha`, `rrf_k`, `knn_threshold`, `hop_lambda`, `similarity_threshold` documentados e não lidos. |
| ALTO | `router.py` | Nunca retorna `"local"` → PPR de `local_search` é inalcançável no default `"auto"`. |
| ALTO | `hybrid_search.py` 64 | Gate `return []` zera recall em eval. |

### API / CI / testes (onda 2 incompleta; evidência parcial)

- Sem `BASINRAG_API_KEY` a API fica aberta; CORS default `*`; `serve` faz bind em `0.0.0.0`.
- `_ensure_retriever` muta `search_type`/`top_k` do retriever compartilhado sob concorrência.
- CI (`.github/workflows`) só em `workflow_dispatch` — nada roda em push/PR.
- Suíte: ~56 testes, cobertura baixa; `test_retrieval_quality` toca `.basinrag` real.
- `eval/beir.py`: TLS desabilitado + `extractall` sem sanitização (zip-slip).

### Docs vs. código

- Encoder documentado como MiniLM multilíngue; `factory.py` usa `bge-base-en-v1.5`.
- Versões 1.0.3 (CITATION/README) vs 1.0.4 (`pyproject`, CHANGELOG).
- Paper: `E_virt` = todos os pares ≥ 0,85; código = top-6 kNN com `> 0,85`.
- “Determinístico” e “build O(N)” não se sustentam (HNSW, LLM T=0,3, FAISS exato N²).

---

## O que remover

| Caminho | Motivo |
|---|---|
| `basinrag/core/protocols.py` | Morto; contratos errados. |
| `basinrag/retriever/projector.py` | Só chamado pelo próprio teste. |
| `engine.meta_basins` / `build_meta_basins` | O(n²) a cada ingest; ninguém lê; não persiste. |
| `HybridSearch.search` / `LocalSearch.search` / `GlobalSearch.search` | Sem callers de produção. |
| `compute_trapping_bounds` + `is_peripheral` | Inalcançáveis. |
| `BasinTopologyEngine.cosine_similarity` | Morto; duplica `functional_graph`. |
| `summarize_all`, schemas Pydantic não usados, `_get_stemmers` | Sem callers. |
| `results/mteb_fase*`, `mteb_*_test`, `mteb_optimized` | Tuning no test set; não publicáveis. |
| Claims GraphRAG +142%, badge SOTA, 0,771 como “oficial” | Recolher até o gate. |

Não commitar `results/mteb/remote/` (~3 GB de cache MTEB).

---

## Trajetória de tuning (test split SciFact)

| Diretório | nDCG@10 | Nota |
|---|---:|---|
| `mteb` | 0,650 | Número 1.0.4 |
| `mteb_optimized` | 0,668 | |
| `mteb_fase1` | 0,683 | |
| `mteb_fase2` | 0,706 | |
| `mteb_fase3` | 0,716 | Ainda < encoder nu |

Padrão: smoke test de 3–5 queries → medição nas 300 → ajuste. O test set está queimado.

---

## Freeze da Fase 0 (aplicado neste trabalho)

O harness de gate **congela** o protocolo; não “melhora” o ranking para passar no SciFact.

1. Title-boost 3× **desligado** (`mteb_wrapper.py`: concatenação `title + text`).
2. Score de rerank **monotônico** (score cru do cross-encoder).
3. Hop prior **comutável**; hop ausente = penalidade, não bônus (`fusion.py`).
4. BM25 unificado em `text` (`factory._attach_bm25` = `HybridSearch._rebuild_bm25`).
5. Cache persistente com chave `tarefa + encoder + modo de chunk`.
6. Runner único: `python -m basinrag.eval.run_gate`.

Sistemas no mesmo harness: encoder puro, BM25, híbrido-min, híbrido-min+rerank, híbrido-min+topo, BasinRAG-full.

Critério de morte (não negociável depois de ver o número):

- **SciFact (controle):** `|Δ nDCG@10|` topo vs híbrido-sem-topo `< 0,005`. Ganho aqui = vazamento.
- **Corpus longo (mediana de membros/bacia ≥ 3):** `híbrido+topo − híbrido-min ≥ +0,01` **e** melhor BasinRAG ≥ encoder puro − 0,01. Senão → `CONVERT_C`.
- Mediana < 3 → `GATE_INVALID` (o gate não rodou).

---

## O que não fazer

- Submeter MTEB / publicar 0,650 ou 0,715 como evidência de topologia.
- Comparar de novo com `graphrag_baseline.py`.
- Tunar no test split do SciFact.
- Reescrever paper/README com números novos antes de `results/gate/decision.json`.
- Priorizar ingest incremental, meta-bacias, PPR ou “build O(N)” antes do gate.

---

## Próximos ramos (depois do `DECISION=`)

**CONTINUE_B:** Fase 1 (sinal de ranking: rerank por `node_id`, knobs vivos, um pipeline `brief`=eval, router ou PPR fora da tese) → Fase 2 (persistência/ingest) → Fase 3 (higiene e claims).

**CONVERT_C:** só bugs de produto (id pós-rerank, BM25 unificado, gates, `brief`=eval) + persistência + recolher claims. Sem hop-as-science, sem paper de nDCG.

O resultado mais provável, dado que o híbrido já perde para o encoder nu **com a topologia desligada**, é `CONVERT_C`.

---

## Resultado do gate (11–12 set 2026)

Harness: `python -m basinrag.eval.run_gate` (title-boost off, BM25=`text`, hop missing=penalty, sem reranker). Artefato: `results/gate/decision.json`.

**SciFact (controle, 300 queries, 5183 bacias unitárias, `h=0`):**

| Sistema | nDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| encoder_pure (`bge-base-en-v1.5`) | **0,741** | 0,704 | 0,874 |
| bm25_pure | 0,675 | 0,641 | 0,802 |
| hybrid_min (BM25+FAISS+RRF) | 0,727 | 0,690 | 0,864 |
| hybrid_min_topo | 0,733 | 0,694 | 0,875 |

O encoder nu foi reproduzido (0,741 vs 0,743 do cache oficial). O híbrido **perde** para o encoder. Δ topo−híbrido = +0,006 (limiar 0,005) — efeito residual de arestas kNN virtuais, não de bacias.

**Corpus longo (80 papers arXiv, 2570 chunks, mediana 20 membros/bacia, geometria OK):**

| Sistema | nDCG@10 |
|---|---:|
| encoder_pure | 1,000 |
| bm25_pure | 0,988 |
| hybrid_min | 0,995 |
| hybrid_min_topo | 1,000 |

Tarefa abstract→paper no teto do encoder. Ganho topológico +0,005 < +0,01 exigido.

**`DECISION=CONVERT_C`.** A tese de ganho de nDCG não passou no critério pré-registrado. O ramo seguinte é produto/briefing: persistência, ingest incremental, rerank por `node_id`, recolher claims. Sem submissão MTEB e sem paper de nDCG até outro experimento (não abstract-retrieval) ser pré-registrado.
