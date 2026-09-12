# Guia de Início Rápido (Quickstart)

Instalar, ingerir e consultar o **BasinRAG** — RAG híbrido BM25 + FAISS com bacias como **mapa de briefing**. Detalhes: [README](../README_pt.md), [BENCHMARKS](../BENCHMARKS.md).

## 1. Requisitos

- **Python** 3.10+
- FAISS, PyTorch, NumPy (puxados pelo pacote)
- Opcional: Ollama (ou API OpenAI-compatível) para chat / L3

## 2. Instalação

```bash
pip install -e ".[dev,api,ollama]"
```

Ou clone + mesmo comando na raiz do repo.

## 3. Uso rápido

### Python

```python
from basinrag import BasinRAG, BasinRAGConfig

config = BasinRAGConfig(
    storage_dir=".basinrag",
    encoder_model="BAAI/bge-base-en-v1.5",  # padrão
)
rag = BasinRAG(config)
rag.ingest("./meus_documentos")  # PDF / MD / TXT

docs = rag.query("Qual é o princípio central?", top_k=5)
for d in docs:
    print(d)
```

Chunks padrão: **512 / overlap 128**.

### CLI

```bash
basinrag ingest ./docs
basinrag query "Quais os principais achados?" --type auto --top-k 5
basinrag chat
basinrag serve --port 8000
```

## 4. O que esperar do retrieval

| Modo | Uso |
| :--- | :--- |
| `global` / `hybrid` | BM25 + FAISS + RRF (ranking principal) |
| `local` | Expansão na vizinhança da bacia (briefing) |

## 5. Próximos passos

- [API Reference](API_REFERENCE.md)
- [Arquitetura](../ARCHITECTURE.md)
- [Deploy](DEPLOYMENT.md)
