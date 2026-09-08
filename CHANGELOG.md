# Changelog

All notable changes to this project will be documented in this file.

## [2.2.0] - 2026-09-06

### Fixed & Hardened (Auditoria Especializada de RAG)
- **CRITICAL-01 (Sincronização de IDs no Re-ranking Cross-Encoder)**:
  - Corrigido desalinhamento crítico em `basinrag.retriever.base.BasinRAGRetriever`: `packet.node_ids` agora é estritamente mapeado e sincronizado com a ordem reordenada pelo Cross-Encoder, prevenindo corrupção de metadados e citações errôneas.
- **CRITICAL-02 (Eliminação de OOM e Streaming SQLite no DiskKVStore)**:
  - Substituída a serialização redundante do banco de dados SQLite inteiro dentro do `meta.json` pelo utilitário transacional nativo de streaming `sqlite3.Connection.backup` em `basinrag.core.kv_store.DiskKVStore` e `persistence.py`.
- **CRITICAL-03 (Tratamento de Exceções no Summarizer)**:
  - Corrigido import ausente de `logger` no bloco `except Exception` de `basinrag.indexer.summarizer.AgenticSummarizer`.
- **HIGH-01 (Suporte Linguístico PT/EN Bilíngue no BM25)**:
  - Implementada detecção automática de idioma do documento (`detect_language`) em `basinrag.indexer.bm25` e `projector.py`. O `BM25Index` agora aplica o `RSLPStemmer` (PT) ou `SnowballStemmer` (EN) consistentemente entre indexação e consulta.
- **HIGH-02 (Normalização L2 de Centroides no Roteador)**:
  - Adicionada normalização vetorial $L2$ explícita em `basinrag.retriever.router.IntelligentQueryRouter.train_centroids` para garantir distâncias euclidianas e cosseno matematicamente válidas.
- **HIGH-03 (Normalização Relativa no Decaimento Espectral PPR)**:
  - Substituído o min-max scaling frágil por normalização relativa ao escore máximo ($score / max\_score$) em `basinrag.retriever.local_search.TopologicalLocalSearch`, evitando divisão por zero quando todos os nós têm escores uniformes.
- **HIGH-04 (Otimização Bulk em Particionamento Topológico)**:
  - Substituídas $N$ chamadas individuais repetitivas de gravação por transação em lote `DiskKVStore.set_many()` em `basinrag.core.topology.TopologicalBasinEngine`.
- **Segurança da API FastAPI**:
  - Middleware de CORS configurável via variável de ambiente `BASINRAG_CORS_ORIGINS`.
  - Comparação de chave de API protegida contra ataques de temporização (*timing attacks*) via `secrets.compare_digest`.
  - Desconexão de WebSocket tratada de forma limpa via `WebSocketDisconnect`.

### Benchmarks (Reavaliação Empírica Rigorosa)
- **BEIR Oficial (SciFact — 50 queries padrão)**:
  - NDCG@10 subiu para **0.771** (+142.5% vs GraphRAG 0.318).
  - MRR@10 subiu para **0.750** (+217.8% vs GraphRAG 0.236).
  - Tempo de indexação caiu para **90.80s** (3.05x mais rápido que GraphRAG 276.68s).
  - Latência de busca por consulta caiu para **2.498,7ms** (4.43x mais rápido que GraphRAG 11.059,4ms).
- **MTEB Hugging Face (SciFact — Bateria completa de 300 queries, k=1000)**:
  - NDCG@10 alcançou **0.650** (+3.6% sobre v0.2.1 baseline).
  - MRR@10 alcançou **0.629** (+9.1% sobre v0.2.1 baseline).
  - MAP@10 alcançou **0.619** (+9.5% sobre v0.2.1 baseline).
  - Hit@1 alcançou **57.0%** (documento correto no topo em 171 de 300 consultas).
  - Recall@1000 atingiu **98.7%**.
- **SWE-bench Lite (13 instâncias de repositórios reais)**:
  - Hit@1 dobrou de 15.4% para **30.8%**.
  - Hit@10 subiu de 76.9% para **84.6%**.
  - MRR subiu de 0.369 para **0.434**.

---

## [2.1.0] - 2026-09-06

### Added
- **Suíte Oficial MTEB (Hugging Face Leaderboard)**:
  - Runner de avaliação padronizada `basinrag.eval.run_mteb` compatível com MTEB v2.20+.
  - Implementação estrita do `SearchProtocol` e `ModelMeta` Pydantic v2 em `basinrag.eval.mteb_wrapper`.
  - Execução oficial completa de 300 queries de teste no dataset `SciFact` ($k \le 1000$).
  - Geração automática do pacote oficial de submissão do Leaderboard do Hugging Face (`results/mteb/results/alexmart1ns__BasinRAG-2.0/2.0.0/`).
- **Documentação Comparativa Global**:
  - Atualização completa do `README.md` com tabelas comparativas contra baselines mundiais (ColBERT, Contriever, BGE-large, SPLADE, BM25, GraphRAG, HyperFL).
  - Publicação do relatório técnico detalhado `relatorio_comparativo_tops_globais.md`.

---

## [2.0.0] - 2026-09-05

### Added
- **Suite Oficial de Benchmarking Comparativo (BasinRAG 2.0 vs GraphRAG)**:
  - Integração oficial com o framework BEIR (`beir.retrieval.evaluation.EvaluateRetrieval`) no dataset `scifact` (5.183 documentos).
  - Runner comparativo de código real no SWE-bench Lite (`basinrag.eval.compare_graphrag`) com 13 instâncias de produção (`flask`, `requests`, `seaborn`).
  - Documento mestre de benchmarks em `BENCHMARKS.md` e logs exportados em `logs/beir_comparison_scifact.json` e `logs/basinrag_vs_graphrag.jsonl`.
- **Topologia de Bacias em 2 Níveis (Meta-Basins)**:
  - Agrupamento hierárquico de bacias funcionalmente correlatas via `AgglomerativeClustering` no espaço de atratores.
- **Difusão Espectral Local por Personalized PageRank (PPR)**:
  - Substituição da expansão heurística por difusão espectral analítica com Power Iteration sobre subgrafo induzido da bacia.
- **Fusão Canônica Min-Max RRF**:
  - Normalização unificada de escores densos e esparsos (BM25) evitando distorção de escala em consultas híbridas.

### Fixed
- **Prevenção de Perda de Dados em Bacias**: Eliminada a remoção destrutiva de nós com hops elevados em `compute_trapping_bounds`.
- **Compatibilidade e Resiliência no Windows**: `safe_replace_dir` com tratamento seguro de handles e concorrência multithread com `RLock` no SQLite DiskKVStore.
- **Operações Não-Bloqueantes no FastAPI**: Métodos assíncronos (`aquery`, `abrief`, `achat`) delegados para threads de background via `asyncio.to_thread`.
- **GraphRAG Baseline**: Correção do parsing de noun chunks no SpaCy e adição de extração de identificadores de código (`CamelCase` e `snake_case`).

---

## [0.2.1] - 2026-09-01

### Fixed
- Local search now hydrates one sequential hop across the φ section cut and virtual-edge neighbors of FAISS hits, not only the three entry points.
- Queries with dense similarity below 0.15 return no passages; chat does not call the LLM.
- Global search returns nothing when every basin score is ~0 instead of dict-order leftovers.
- Projector matches whole tokens and drops BM25 stopwords, so filler PT/EN words are not extra hubs.
- Index save writes to a temp directory then replaces; missing `basins/` is rebuilt from φ; stale BM25 (`buildId` / node set) is discarded.
- JSON ingest assigns one `source` per document and splits like TXT/PDF; PDF `page` is kept in metadata.
- CLI `query`/`chat` error out without an index; chat `input()` runs in a thread so the L3 daemon is not blocked.
- UniversalLLM is constructed on first chat/summarize, not on ingest.

### Changed
- Satellites are `l3_summary` or `l3_label`, capped at 400 characters.

### Docs
- ARCHITECTURE.md documents that DSPM \(f_{k,b}\) is not instantiated; φ/basin/attractor are homonyms of sequential index partitions.

### Removed
- Legacy root scripts (chat/eval/compare/ingest-dataset), generated reports, `requirements.txt`, and dead experimental modules (`dr9_sieve`, `diophantine`, `seed_selector`). Use `basinrag` CLI only.

## [0.2.0] - 2026-09-01

### Added
- Functional successor φ (sequential drain to section start), real rho-tree edges, hop prior in fusion.
- CSR BM25 (`k1=1.2`, `b=0.75`) persisted beside the graph.
- Weighted RRF (`α=0.55`) fused on node ids; BriefingPacket with L0 hubs / L2 neighbors / L3 satellites.
- Stable content-addressed node ids; recursive ingest for `.md` / `.txt` / `.pdf`.
- Extractive L0–L3 at ingest; global search scores L3 text.
- HNSW when the corpus exceeds 256 vectors; `buildId` in `meta.json`.
- English keywords on the query router; multilingual cross-encoder.

### Fixed
- Re-ingest no longer appends duplicate UUID nodes.
- Empty ingest no longer wipes a good index; missing ingest dirs are not created.
- Extractive basin labels no longer block the L3 LLM daemon; global ranks attractor embeddings until a real L3 exists.
- Auto router no longer sends factual queries (`quais são…`) to global.
- Hybrid semantic arm is corpus-wide FAISS, not basin-local hop walk.
- Local ranking uses dense similarity × hop floor instead of hop-only sort.
- Chat/`configure(search_type=None)` no longer sticks on a previous strategy.
- CLI imports uvicorn only for `serve`.
- `encoder_model` is stored in `meta.json`; load warns on mismatch.
- `search_type` / `top_k` honored on every query (CLI + API).
- Router tests match hybrid/global behavior.
- Summarizer uses the same `model_name` as chat.
- NetworkX `node_link` load/save with `edges="links"`.
- Shallow `Graph.copy()` no longer strips live embeddings on save; load refuses a graph without `embeddings.npz`.

### Changed
- Attractors are section sinks, not PageRank 15%. DR9/QMC stay off the live ranking path.
- ARCHITECTURE.md rewritten to match the product (two layers; theorems do not rank documents).

## [1.1.0] - 2026-08-07

### Added
- **Agentic Summarizer**: Implementado um loop agentic avançado (Draft -> Critique -> Refine) para geração de resumos L3 altamente estruturados.
- **Standby Background Daemon**: A ingestão de documentos agora é instantânea. O LLM gera os resumos L3 silenciosamente em background durante o uso do Chat ou da API Server.
- **Multilingual Support**: O modelo de sentence transformers foi atualizado para `paraphrase-multilingual-MiniLM-L12-v2`, permitindo buscar em português documentos em inglês com alta precisão semântica.
- **Verbose Flag**: Adicionada flag `verbose` para o background daemon, permitindo ocultar logs no CLI interativo mas exibi-los no servidor FastAPI.

### Changed
- **Query Router**: Atualizado o fallback do `IntelligentQueryRouter` para usar "hybrid" em vez de "local" para lidar melhor com buscas exatas de entidades cross-language.
- **Factory**: A configuração padrão de `search_type` foi alterada de "local" para "auto" para se beneficiar ativamente do Roteador Inteligente.
- **Async Fixes**: Corrigido um bug onde o CLI disparava exceções de "Event loop is closed" durante interações contínuas. O Chat CLI agora utiliza um único Event Loop para todo o ciclo de vida.
- **Encoding**: Forçado UTF-8 no stdout do Windows no `cli.py` para exibir emojis corretamente.

### Removed
- Removido o bloqueio síncrono durante a geração de resumos na ingestão.
