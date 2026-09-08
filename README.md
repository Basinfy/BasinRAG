# BasinRAG

[![CI](https://github.com/alexmart1ns/BasinRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/alexmart1ns/BasinRAG/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG topológico de alta performance para documentos**, utilizando bacias de atração como **partição de índice**.

O BasinRAG indexa e recupera trechos de documentos (PDF, Markdown, TXT) combinando **BM25 CSR bilíngue (PT/EN)**, **embeddings densos FAISS (Flat/HNSW)** e a **estrutura topológica sequencial do documento** (grafo funcional $\phi$, atratores e árvores $\rho$).

---

## 🚀 Funcionalidades Principais

- **Topologia de Grafo Funcional ($\phi$)**: Partição em bacias baseada em fluxo sequencial e quebras adaptativas por similaridade de cosseno (sem clusters arbitrários ou instáveis).
- **Arestas Virtuais Semânticas**: Enriquecimento do grafo via $k$-NN semântico com limiar $> 0.85$ sem poluir a topologia das bacias.
- **Fusão Híbrida RRF + Hop Prior**: Fusão balanceada ($\alpha=0.55$) de ranking lexical e vetorial penalizada por decaimento exponencial de distância topológica $\exp(-\text{hops} \cdot \lambda)$.
- **Re-ranking Multilíngue**: Cross-Encoder mMARCO (`mmarco-mMiniLMv2-L12-H384-v1`) para precisão final no topo do ranking.
- **Roteador Inteligente de Consultas**: Classificação PT/EN (`global` vs `hybrid`) com fast-path regex e fallback de centroides de embeddings.
- **Resumos L3 Agentic**: Pipeline com loop *Draft → Critique → Refine* executado em background daemon sem bloquear o chat.
- **Persistência Atômica**: Troca segura via diretório temporário (`os.replace`), prevenindo corrupção em falhas repentinas.
- **Cache Inteligente de Consultas**: Cache LRU integrado na camada de retrieval evitando re-encoding redundante de queries.
- **API REST & WebSocket com Rate-Limiting**: Interface FastAPI com proteção de concorrência, endpoints documentados e autenticação por API Key.

---

## 📦 Instalação

```bash
# Instalação básica com dependências recomendadas
pip install -e ".[dev,api,ollama]"
```

---

## ⚡ Guia Rápido (CLI)

```bash
# 1. Ingerir documentos de um diretório
basinrag ingest ./docs

# 2. Fazer uma consulta direta
basinrag query "Qual é o tema principal do documento?" --type auto --top-k 5

# 3. Iniciar chat interativo em streaming
basinrag chat

# 4. Iniciar servidor FastAPI com WebSocket
basinrag serve --port 8000
```

---

## 🌐 API e WebSocket

Ao rodar `basinrag serve --port 8000`:
- **Documentação Swagger**: `http://localhost:8000/docs`
- **Endpoint de Consulta**: `POST /query` com payload `{"query": "sua pergunta", "top_k": 5}`
- **WebSocket Streaming Chat**: `ws://localhost:8000/chat?token=SEU_TOKEN`

---

## ⚙️ Configuração

Copie o arquivo de exemplo para configurar variáveis de ambiente:
```bash
cp .env.example .env
```

| Variável | Padrão | Descrição |
|---|---|---|
| `BASINRAG_API_KEY` | *None* | Chave opcional de autenticação via header `X-API-KEY` |
| `BASINRAG_LLM_PROVIDER` | `ollama` | Provedor de LLM (`ollama` ou `openai`) |
| `BASINRAG_LLM_MODEL` | `qwen2.5` | Nome do modelo para o LLM |
| `BASINRAG_ENCODER_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo SentenceTransformer para embeddings |
| `BASINRAG_STORAGE_DIR` | `.basinrag` | Diretório de persistência do índice |

---

## 📁 Estrutura do Repositório

```text
basinrag/
├── api/            # Servidor FastAPI, endpoints REST e WebSocket
├── core/           # Grafo funcional φ, bacias, FAISS, SQLite KVStore, persistência atômica, protocolos
├── eval/           # Métricas e wrappers de avaliação (BEIR, MTEB)
├── indexer/        # Ingestão de arquivos (PDF/MD/TXT), BM25 CSR, condensação L0-L3, summarizer agentic
├── retriever/      # Local search, global search, hybrid search, RRF fusion, CrossEncoder e query router
├── cli.py          # Interface de linha de comando
├── factory.py      # Facade BasinRAG e dataclass BasinRAGConfig
└── logging_config.py
docs/               # Documentação complementar de bacias
tests/              # Suíte completa de testes unitários e de integração
.github/workflows/  # Pipeline de CI (Matrix Python 3.10/3.11/3.12 + Ruff + Mypy + Pytest)
```

---

## 🧪 Execução de Testes

```bash
# Executar todos os testes com cobertura
pytest tests/ -v --cov=basinrag
```

---

## 📖 Documentação Detalhada

- Para detalhes da formulação matemática e arquitetura do pipeline, consulte [ARCHITECTURE.md](ARCHITECTURE.md).
- Para histórico de versões e mudanças, consulte [CHANGELOG.md](CHANGELOG.md).
