# Referência de API do BasinRAG

O produto CONVERT_C é um híbrido **BM25 + FAISS**. As bacias são um mapa de briefing (vizinhança de contexto), não um knob de nDCG.

## SDK Python

### `basinrag.factory.BasinRAG`

- **`BasinRAG.create(**kwargs)`** — instancia e tenta `load()` do índice em `storage_dir`.
- **`ingest(path)`** — mescla documentos novos por `id`. Não apaga o grafo existente.
- **`reindex(path)`** — reconstrói o índice do zero a partir de `path`.
- **`query(question, search_type=None, top_k=None)`** — devolve `Document`s. Overrides não mutam o retriever compartilhado.
- **`brief(question, search_type=None, top_k=None)`** — `BriefingPacket` (hubs, satélites, `node_ids`). O gate de confiança **não** zera hubs; no `chat`, só abstém a geração.
- **`chat(question)`** — streaming LLM; abstém se `confidence < min_confidence`.

### `basinrag.factory.BasinRAGConfig`

| Atributo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `storage_dir` | `str` | `".basinrag"` | Raiz do índice. Builds em `builds/<id>/`; ponteiro atômico `current.json`. |
| `provider` | `str` | `"ollama"` | Provedor do LLM (`ollama` ou `openai`). |
| `model_name` | `str` | `"qwen2.5"` | Modelo LLM para chat e L3. |
| `encoder_model` | `str` | `"BAAI/bge-base-en-v1.5"` | SentenceTransformer (768d). |
| `reranker_model` | `str` | `"BAAI/bge-reranker-v2-m3"` | Cross-Encoder. |
| `search_type` | `str` | `"auto"` | `auto`, `hybrid`, `local`, `global`. |
| `chunk_size` | `int` | `512` | Tamanho do splitter (retrieval child). |
| `chunk_overlap` | `int` | `128` | Overlap entre chunks (~25%). |
| `query_prompt` | `str` | BGE instruction | Prefixo de query para o encoder. |
| `use_rerank` | `bool` | `True` | Cross-Encoder no path de produção. |
| `chunk_overlap` | `int` | `100` | Overlap real do splitter. |
| `min_confidence` | `float` | `0.15` | Só abstém o `chat`; não esvazia o briefing. |

Knobs de tese (`hybrid_alpha`, `rrf_k`, `knn_threshold`, `hop_lambda`, `similarity_threshold`) **não** fazem parte da config pública.

### Persistência

`BasinPersistence.save_topology` grava em `storage_dir/builds/<build_id>/` e faz `os.replace` de `current.json`. Fecha os SQLite (`successor` / `attractor_of`) antes de trocar o ponteiro e reabre no build ativo. `basin_id` vira hash no nome do arquivo; o id real fica em `_meta`.

## HTTP e WebSocket

Bind default: `127.0.0.1`. Bind público (`0.0.0.0`) exige `BASINRAG_API_KEY`. CORS default é localhost.

- **`POST /query`** — header `X-API-Key` quando a chave está definida. Rate limit 30/min.
- **`GET /`** — `{"status": "online"}`.
- **`WS /chat`** — autenticação por header `X-API-Key` (não query string). Rate limit próprio.
