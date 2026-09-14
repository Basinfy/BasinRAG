# Referência da API do BasinRAG

BasinRAG combina BM25 e FAISS por RRF no modo padrão `hybrid_rrf`. Em índices flat (1 nó por documento) α=0.15 e pool `max(50, top_k×5)`; o BM25 só vota no RRF dentro do pool denso. Em índices fatiados α=0.40, pool `max(200, top_k×10)`, folhas sentence-window (~1–2 sentenças), first-stage por paper, expansão léxica RM3 sem LLM, união BM25@10 no membership e até 2 papers só-BM25 inseridos na 3ª posição. As bacias particionam o índice e o briefing padrão hidrata irmãos da árvore ρ depois das sementes RRF; hop/DRF não reordenam as sementes. `experimental_topology` ativa a ordenação topológica experimental.

## SDK Python

### `basinrag.factory.BasinRAG`

- **`BasinRAG(config=None)`** — cria a fachada. Sem uma configuração explícita, resolve opções do processo, `.env` e padrões.
- **`BasinRAG.create(**kwargs)`** — aplica as mesmas regras de configuração, cria a instância e tenta carregar o índice existente.
- **`ingest(path)`** — incorpora arquivos novos ou atualizados por ID. Não remove fontes ausentes; use `sync` para sincronizar uma raiz.
- **`sync(path)`** — sincroniza um arquivo ou diretório: substitui os chunks das fontes alteradas e remove, dentro do escopo sincronizado, fontes suportadas que desapareceram. Retorna a quantidade de chunks ingeridos na operação.
- **`reindex(path)`** — constrói um índice novo apenas com o conteúdo de `path`; retorna a quantidade de chunks ingeridos.
- **`query(question, search_type=None, top_k=None)`** — retorna `langchain_core.documents.Document`. Os overrides valem somente para aquela chamada.
- **`brief(question, search_type=None, top_k=None)`** — retorna um `BriefingPacket` com sementes recuperadas, contexto e `node_ids`.
- **`chat(question)`** — gera uma resposta em streaming com o LLM configurado. O limiar `min_confidence` pode levar à abstenção; o valor de confiança atual é uma heurística de similaridade e não uma probabilidade calibrada.

`search_type` aceita `auto`, `hybrid`, `local` ou `global`. Prefira `hybrid` para a referência BM25 + FAISS. A expansão local pode adicionar contexto ao briefing sem alterar a ordem das sementes no modo padrão.

### `basinrag.BasinRAGConfig`

| Atributo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `storage_dir` | `str` | `".basinrag"` | Diretório de armazenamento dos snapshots. |
| `provider` | `str` | `"ollama"` | Provedor de chat/L3 (`ollama` ou `openai`). |
| `model_name` | `str` | `"qwen2.5"` | Modelo de chat e L3. |
| `encoder_model` | `str` | `"BAAI/bge-base-en-v1.5"` | Encoder denso. Trocar o encoder exige `reindex`. |
| `reranker_model` | `str` | `"BAAI/bge-reranker-v2-m3"` | Cross-encoder usado quando o reranking está habilitado. |
| `search_type` | `str` | `"auto"` | Busca padrão: `auto`, `hybrid`, `local` ou `global`. |
| `ranking_mode` | `str` | `"hybrid_rrf"` | `hybrid_rrf` é produção; `experimental_topology` é opt-in e requer avaliação por ablação. |
| `chunk_size` | `int` | `512` | Tamanho legado do splitter, em caracteres. |
| `chunk_overlap` | `int` | `128` | Overlap legado do splitter, em caracteres. |
| `chunk_size_tokens` | `int \| None` | `None` | Limite aditivo de tokens, limitado à capacidade do encoder. |
| `chunk_overlap_tokens` | `int \| None` | `None` | Overlap em tokens; requer `chunk_size_tokens`. |
| `embedding_batch_size` | `int` | `64` | Textos enviados ao encoder por lote. |
| `query_prompt` | `str` | Instrução BGE | Prefixo da consulta para o encoder. |
| `use_rerank` | `bool` | `True` | Habilita o cross-encoder após a busca. |
| `enable_background_l3` | `bool` | `False` | Habilita sumarização em background. |
| `allow_remote_l3_egress` | `bool` | `False` | Opt-in adicional para enviar trechos a um LLM remoto para sumarização. |
| `min_confidence` | `float` | `0.15` | Limiar heurístico de abstenção; não é confiança calibrada. |

As opções legadas `chunk_size` e `chunk_overlap` mantêm a unidade de caracteres da API 1.0.x. Depois dessa divisão, trechos que excedem a capacidade do encoder são subdivididos pelo tokenizer com margem para tokens especiais. Os novos campos de tokens são opcionais e aditivos; não alteram chunks já persistidos. O manifesto registra metadados do encoder e da política de chunking. Se o modelo/tokenizer persistido não for compatível com a configuração atual, reindexe explicitamente.

### Variáveis de ambiente

Ordem de precedência: **argumentos explícitos > variáveis do processo > `.env` > padrões**. O `.env` é carregado sem sobrescrever valores definidos pelo processo. A API exige `BASINRAG_API_KEY`; acesso sem chave só é permitido em `local_dev`, com opt-in específico e bind loopback.

| Variável | Uso |
|---|---|
| `BASINRAG_STORAGE_DIR` | Diretório de armazenamento. |
| `BASINRAG_ENCODER_MODEL`, `BASINRAG_RERANKER_MODEL` | Modelos de embedding e reranking. |
| `BASINRAG_LLM_PROVIDER`, `BASINRAG_LLM_MODEL` | Provedor e modelo para chat/L3. |
| `BASINRAG_SEARCH_TYPE`, `BASINRAG_RANKING_MODE` | Modo de busca e ranking. |
| `BASINRAG_USE_RERANK`, `BASINRAG_MIN_CONFIDENCE` | Reranking e limiar de abstenção. |
| `BASINRAG_CHUNK_SIZE`, `BASINRAG_CHUNK_OVERLAP` | Opções legadas de chunk em caracteres. |
| `BASINRAG_CHUNK_SIZE_TOKENS`, `BASINRAG_CHUNK_OVERLAP_TOKENS` | Opções aditivas em tokens. |
| `BASINRAG_EMBEDDING_BATCH_SIZE`, `BASINRAG_QUERY_PROMPT` | Lote de embedding e prefixo da consulta. |
| `BASINRAG_API_KEY` | Segredo da API. Necessário para clientes remotos e bind público. |
| `BASINRAG_CORS_ORIGINS` | Lista separada por vírgula de origens permitidas. |
| `BASINRAG_ENV`, `BASINRAG_ALLOW_KEYLESS_LOCAL` | Modo de execução e opt-in local sem chave. |
| `BASINRAG_TRUSTED_PROXIES` | Redes confiáveis apenas para IP/rate limit encaminhado. |
| `BASINRAG_ENABLE_BACKGROUND_L3`, `BASINRAG_ALLOW_REMOTE_L3_EGRESS` | Opt-ins independentes para sumarização e envio remoto. |
| `BASINRAG_ENCODER_REVISION`, `BASINRAG_RERANKER_REVISION` | Revisões imutáveis dos modelos. |
| `BASINRAG_SOURCE_ENCODING`, `BASINRAG_BM25_STEMMING` | Override explícito de encoding e stemming opcional. |

### Persistência

O formato do snapshot é versionado internamente para detectar índices incompatíveis sem escrita. Um índice antigo gera `IndexRebuildRequired` sem alterar arquivos. Reindexe as fontes originais para um destino novo/vazio, por padrão `.basinrag`; se esse diretório contiver um índice legado, escolha outro destino vazio e mantenha o root antigo para rollback. Cada geração registra manifesto, revisões, proveniência e checksums, e só troca `current.json` depois das validações. Faça backup/restaure o root como conjunto coerente.

## HTTP e WebSocket

A aplicação FastAPI é `basinrag.api.server:app`. Inicie pela CLI com `basinrag serve --host 127.0.0.1 --port 8000` ou pelo ASGI com `uvicorn basinrag.api.server:app --host 127.0.0.1 --port 8000`.

- **Autenticação:** somente `Authorization: Bearer <key>`; nunca use credencial em URL. WebSocket browser usa cookie same-origin autenticado, com Origin allowlisted. A chave é obrigatória no startup, salvo modo local explícito e loopback.
- **`GET /livez`** — processo ativo, sem depender do snapshot.
- **`GET /readyz`** — snapshot e dependências de consulta válidos; caso contrário 503.
- **`POST /query`** — corpo JSON com `query`, `search_type` (`auto|local|global|hybrid`) e `top_k` (1–50). Resultados incluem ref, texto, fonte, página, posição e papel, sem caminho absoluto. Sem snapshot válido, responde 503.
- **`WS /chat`** — mensagens JSON limitadas; eventos incluem request id, tokens, referências, erro e conclusão. Citações são validadas contra referências recuperadas; geração é cancelada ao desconectar.
- **Limites:** corpo WS 8 KiB, até 64 conexões, 8 gerações simultâneas e timeout de 120 s. Configure `BASINRAG_TRUSTED_PROXIES` somente para IP encaminhado; isso nunca autentica.
- **CORS/Origin:** allowlist explícita; wildcard não é aceito. Origem WebSocket também é validada.

Os limites de taxa são locais ao processo. A topologia suportada é um processo em um host; não aumente workers do ASGI sem uma implementação compartilhada de estado e uma estratégia de publicação compatível.
