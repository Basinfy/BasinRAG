# 📚 Referência de API do BasinRAG

Esta documentação define formalmente as principais classes, métodos e endpoints HTTP disponibilizados pela biblioteca **BasinRAG**.

## SDK Python Core

### `basinrag.factory.BasinRAG`
Classe principal e ponto de entrada da biblioteca. Controla a orquestração entre indexadores, retrievers e geração.

- **`from_directory(directory: str, config: BasinRAGConfig) -> BasinRAG`**
  Constrói a engine a partir de um diretório. Realiza parsing, chunking e indexação total (esparsa, densa e topológica).
- **`ingest(documents: List[Document]) -> None`**
  Ingere documentos de forma programática.
- **`query(prompt: str, top_k: int = 5, rerank: bool = True, use_global: bool = False) -> QueryResponse`**
  Executa busca híbrida (BM25 + FAISS) com RRF, aplicando priors topológicos. Opcionalmente passa pelo Re-ranker e envia contextos para o LLM.
- **`chat_stream(prompt: str, session_id: str = None) -> Iterator[str]`**
  Semelhante ao `query`, mas retorna um gerador com a resposta parcial do LLM em streaming. Mantém histórico por `session_id`.
- **`get_stats() -> Dict[str, Any]`**
  Retorna métricas do index: contagem de vetores, atratores descobertos, e tamanhos dos clusters.

### `basinrag.factory.BasinRAGConfig`
Pydantic BaseSettings / Dataclass para configuração profunda da engine.

| Atributo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `storage_dir` | `str` | `".basinrag"` | Diretório de persistência atômica do índice. |
| `provider` | `str` | `"ollama"` | Provedor do LLM (`ollama` ou `openai`). |
| `model_name` | `str` | `"qwen2.5"` | Nome do modelo LLM para síntese e chat. |
| `encoder_model` | `str` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Modelo SentenceTransformers para embeddings densos. |
| `reranker_model` | `str` | `"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"` | Modelo Cross-Encoder para re-ranking neural. |
| `search_type` | `str` | `"auto"` | Estratégia de busca padrão (`auto`, `hybrid`, `local`, `global`). |
| `chunk_size` | `int` | `1000` | Tamanho dos chunks textuais em caracteres. |
| `chunk_overlap` | `int` | `100` | Sobreposição de caracteres entre chunks adjacentes. |
| `hybrid_alpha` | `float` | `0.55` | Peso da busca vetorial na fusão RRF (0.0 a 1.0). |
| `rrf_k` | `int` | `60` | Constante de suavização do Reciprocal Rank Fusion. |
| `similarity_threshold` | `float` | `0.55` | Limiar de similaridade de cosseno para arestas sequenciais do $\phi$. |
| `knn_threshold` | `float` | `0.85` | Limiar mínimo para conexão de arestas virtuais $k$-NN. |
| `min_confidence` | `float` | `0.15` | Confiança mínima do packet para acionar o LLM. |
| `hop_lambda` | `float` | `0.35` | Taxa de decaimento exponencial do prior topológico. |

### Componentes Internos Principais

- **`basinrag.core.topology.BasinTopologyEngine`**
  Engine matemática do grafo funcional e bacias de atração.
  - `build_graph(nodes)`: Constrói o grafo direcionado sequencial $\phi$ e sinapses virtuais $k$-NN.
  - `partition_into_basins()`: Decompõe o grafo funcional em componentes fortemente conexos e árvores $\rho$.
  - `build_meta_basins()`: Agrupa bacias afins via clustering aglomerativo dos centroides de atratores.

- **`basinrag.core.kv_store.DiskKVStore`**
  Armazenamento chave-valor chaveado em SQLite com WAL e cache LRU em memória.
  - `set(key, value)` / `get(key)`: Operações atômicas com thread-safety via `RLock`.
  - `set_many(mapping)`: Transação em lote de alta performance para particionamento topológico.
  - `backup_to(dest_conn)` / `restore_from(src_conn)`: Streaming online com `sqlite3.Connection.backup` (zero OOM).

- **`basinrag.indexer.bm25.BM25Index`**
  Índice esparso bilíngue (PT/EN) com stemming dinâmico e detecção de idioma.
  - `detect_language(text)`: Classificação automática via diacríticos e stopwords lexicais.
  - Suporte consistente a português sem acentos via `RSLPStemmer` e inglês via `SnowballStemmer`.
  - Serialização atômica em disco via `save(path)` / `load(path)`.

- **`basinrag.retriever.hybrid_search.HybridSearch`**
  Fusão de busca BM25 + FAISS com prior de saltos topológicos ($\exp(-\text{hops} \cdot \lambda)$).

- **`basinrag.retriever.reranker.CrossEncoderReranker`**
  Reclassificador neural Transformer (`mmarco-mMiniLMv2-L12-H384-v1`).
  - Sincronização estrita de `node_ids` pós-rerank, prevenindo desalinhamento de citações.

## Endpoints HTTP & WebSocket (FastAPI)

- **`POST /query`**
  Executa a recuperação completa e opcionalmente gera resposta com LLM.
  - Header: `X-API-KEY` (opcional, validado via `secrets.compare_digest`).
  - Rate limit: 60 req/min por IP.
- **`POST /ingest`**
  Recebe documentos (`.pdf`, `.md`, `.txt`) para indexação imediata.
- **`GET /stats`**
  Retorna número de nós, atratores identificados, bacias e status do cache.
- **`GET /health`**
  Healthcheck para orquestradores (Kubernetes/Docker). Retorna `{"status": "healthy"}`.
- **`WS /chat`**
  WebSocket bidirecional para chat conversacional em streaming contínuo.
  - Tratamento limpo de encerramento via `WebSocketDisconnect`.
