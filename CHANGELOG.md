# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Docs
- README / README_pt, `BENCHMARKS.md`, arquitetura e índice descrevem o produto: híbrido BM25+FAISS; bacias como mapa de briefing; números SciFact / BEIR atuais.

### CONVERT_C
- Gate (`results/gate/decision.json`): bacias como briefing; ranking híbrido no SciFact.
- Persistência por ponteiro `current.json` + builds; `ingest` mescla; API default em localhost.

## [1.0.4] - 2026-09-09

### Mathematical Alignment & Formal Rigor
- **Formalização do Operador de Contração/Predecessor $\phi$**: Ajustada a nomenclatura e a definição formal no paper LaTeX (`basinrag_paper.tex` e `basinrag_paper_pt.tex`) para caracterizar $\phi$ explicitamente como operador estrutural retrógrado de antecessor em direção ao atrator raiz da seção textual.
- **Harmonização do Prior Topológico Exponencial**: Alinhada a formulação teórica em `docs/THEORY.md` com a cota convexa amortecida $\omega_{\min} = 0.70$ e taxa de decaimento $\lambda = 0.35$ estritamente congruente com o código-fonte (`retriever/fusion.py`).
- **Extirpação de Jargão Espúrio ("Projeção de Cauchy")**: Removidas referências incorretas a Projeções e Atratores de Cauchy em `docs/BENCHMARK_COMPARISON_pt.md`, substituindo-as pelas definições matemáticas reais de representações métricas $L_2$ e particionamento determinístico por grafos funcionais.
- **Formalização da Calibração Afim RRF & Mistura Afim PPR**: Documentadas formalmente nos papers a transformação afim de escala ($60 \times$) para o intervalo $[0, 1]$ e a combinação convexa pós-PPR ($0.6 \cdot \text{sim} + 0.4 \cdot \mathbf{p}$) aplicada às sementes antes do amortecimento por hops.
- **Padronização MTEB / BEIR**: Atualização dos metadados de submissão oficial para `Basinfy/BasinRAG` e consolidação das diretrizes de publicação pública.

---

## [1.0.3] - 2026-09-08

### Multilingual & Global Expansion (Top 10 Languages)
- **Tokenização Abrangente Unicode & CJK**: Regex nativa em `basinrag.core.ids.tokenize` para processar caracteres latinos, cirílicos e ideogramas/sílabas CJK (Hanzi, Hiragana, Katakana, Hangul) sem dependências pesadas em C++ (`jieba`/`mecab`).
- **Indexador Lexical BM25 Multilíngue**: Detecção automática de idioma e suporte a stemmers do NLTK Snowball para os 10 principais idiomas de desenvolvimento global (EN, PT, ES, ZH, JA, DE, FR, RU, KO, IT) com pass-through seguro para idiomas CJK em `basinrag.indexer.bm25`.
- **Roteador Inteligente de Consultas Multilíngue**: Expansão de frases e termos de intenção de visão geral e comparação em `basinrag.retriever.router.IntelligentQueryRouter` para os 10 idiomas, com centroides semânticos multilíngues calibrados.
- **Preservação de Idioma no Summarizer L3**: Diretivas de language-awareness nos prompts de Draft, Critique e Refine em `basinrag.indexer.summarizer.BasinSummarizer`, garantindo resumos no idioma de origem.
- **Documentação de Arquitetura Multilíngue**: Atualização completa de `docs/MULTILINGUAL.md` com a matriz técnica dos 10 idiomas.
- **Suíte de Testes Multilíngues**: Criação de `tests/test_multilingual.py` com 6 novos cenários cobrindo todos os 10 idiomas (56 testes aprovados).

---

## [1.0.2] - 2026-09-08

### Security & Hardening
- **Neutralização de RCE via Pickle**: Removida a flag insegura `allow_pickle=True` no carregamento de embeddings em `basinrag.core.persistence.BasinPersistence`, forçando desserialização nativa segura.
- **Sanitização de CORS na API FastAPI**: Ajustado o `CORSMiddleware` em `basinrag.api.server` para dinamicamente desligar credenciais (`allow_credentials=False`) quando `origins == ["*"]`, eliminando o `AssertionError` do Starlette no startup.
- **Gestão Segura de File Descriptors**: Fechamento automático de descritores de arquivo `.npz` via gerenciadores de contexto `with np.load(...) as data:`.

### Resilience & Data Integrity
- **Salvamento Atômico Transacional de Topologia**: O método `save_topology` em `persistence.py` realiza a substituição atômica completa do diretório de armazenamento via `safe_replace_dir(tmp, self.storage_dir)`, impedindo corrupção de índices em paradas abruptas.
- **Preservação de Subdiretórios em Fallback**: Corrigido o fallback de `safe_replace_dir` com `shutil.copytree(dirs_exist_ok=True)`, garantindo integridade de pastas aninhadas.
- **Tratamento de Exceções de Background**: Registro explícito em log de exceções assíncronas não capturadas no encerramento da task de background em `server.py`.

### Performance & Scaling
- **Streaming Paginado no DiskKVStore**: Implementados os iteradores paginados `iter_keys()`, `iter_items()` e `iter_values()` em lotes no SQLite com liberação de lock entre lotes, além de refatorar `__iter__` para streaming, prevenindo picos de memória (OOM).
- **Otimização de RAM no BM25Index**: O método `load` consome estruturas massivas via `pop()` e `del`, liberando imediatamente memória transitória no boot.
- **Estabilidade no LLM**: Substituído retorno mascarado de string por exceptions formais (`RuntimeError`, `ValueError`) em `basinrag.core.llm.UniversalLLM`, habilitando a política de retries exponenciais.

### CLI & UX
- **Flag de Versão**: Adicionado argumento `basinrag --version` no CLI.
- **Encoding UTF-8**: Reconfiguração explícita de `sys.stdin` e `sys.stdout` para UTF-8.
- **Licenciamento Open-Source (Apache 2.0)**: Migração oficial da licença para **Apache License, Version 2.0**, assegurando concessão e proteção de patentes para a comunidade e viabilizando o modelo de expansão Open-Core.
- **Testes de Regressão**: Inclusão de `tests/test_audit_fixes.py` cobrindo 100% dos novos comportamentos (50 testes aprovados).

---

## [1.0.1] - 2026-09-06

### Fixed & Hardened
- **Sincronização de IDs no Re-ranking Cross-Encoder**: Corrigido desalinhamento em `basinrag.retriever.base.BasinRAGRetriever`: `packet.node_ids` agora é estritamente mapeado com a ordem reordenada pelo Cross-Encoder.
- **Streaming SQLite no DiskKVStore**: Substituída a serialização redundante do SQLite inteiro no `meta.json` pelo utilitário nativo `sqlite3.Connection.backup`.
- **Tratamento de Exceções no Summarizer**: Corrigido import de logger no bloco `except Exception` de `basinrag.indexer.summarizer.AgenticSummarizer`.
- **Suporte Bilíngue PT/EN no BM25**: Implementada detecção de idioma e stemmers correspondentes (`RSLPStemmer` / `SnowballStemmer`) em `basinrag.indexer.bm25` e `projector.py`.
- **Normalização L2 de Centroides**: Adicionada normalização vetorial explícita no roteador de consultas para distâncias válidas.
- **Normalização Relativa no PageRank Personalizado**: Substituído min-max scaling por normalização relativa ao escore máximo ($score / max\_score$), evitando divisão por zero.
- **Operações Bulk no Motor Topológico**: Gravações em lote via `DiskKVStore.set_many()`.

---

## [1.0.0] - 2026-09-06

### Added
- **Primeiro Release Estável do BasinRAG**: Lançamento oficial do motor de RAG topológico de alta performance.
- **Topologia de Bacias em 2 Níveis (Meta-Basins)**: Agrupamento hierárquico de bacias funcionalmente correlatas via `AgglomerativeClustering` no espaço de atratores.
- **Difusão Espectral Local por Personalized PageRank (PPR)**: Difusão espectral analítica com Power Iteration sobre subgrafo induzido da bacia.
- **Fusão Canônica Min-Max RRF**: Normalização unificada de escores densos (FAISS) e esparsos (BM25 CSR).
- **Suíte Oficial de Avaliação e Benchmarks**:
  - Integração com BEIR/MTEB (SciFact). O número de 50 queries foi recolhido; ver gate CONVERT_C.
  - Avaliação no SWE-bench Lite (Hit@1 30.8%).
  - MTEB Hugging Face (SciFact 300 queries — NDCG@10 0.650, Recall@1000 98.7%).
- **Infraestrutura Aberta**: Documentação completa (`ARCHITECTURE.md`, `BENCHMARKS.md`, `CONTRIBUTING.md`, `LICENSE` MIT).

---

## [1.0.0-rc.1] - 2026-09-01

### Added
- **Pré-release Candidate**: Fundação inicial da arquitetura BasinRAG.
- **Grafo Funcional Sequencial $\phi$**: Bacias de atração como partição natural de índices documentais.
- **Índice Híbrido**: BM25 CSR bilíngue + embeddings vetoriais densos via FAISS (Flat/HNSW).
- **API FastAPI e CLI**: Endpoints REST `/query`, WebSocket `/chat` com streaming e interface de linha de comando `basinrag`.
- **Ingestão Multiformato**: Suporte nativo para documentos Markdown, PDF e texto simples.
