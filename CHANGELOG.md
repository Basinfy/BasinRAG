# Changelog

All notable changes to this project will be documented in this file.

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
