# 🚀 Guia de Início Rápido (Quickstart)

Este guia prático fornece o passo a passo para você instalar, configurar e extrair valor do **BasinRAG** em poucos minutos.

## 1. Requisitos do Sistema

Antes de começar, certifique-se de que seu ambiente atende aos seguintes requisitos:
- **Python:** 3.10 ou superior
- **Bibliotecas Base:** FAISS, PyTorch, NumPy, SciPy, FastAPI
- **Modelos Locais (Opcional):** Ollama ou serviço compatível com API do OpenAI para geração (LLM)

## 2. Instalação

### Instalação via Pip
A forma mais fácil de instalar o BasinRAG é via `pip`:
```bash
pip install basinrag
```

### Instalação para Desenvolvimento
Se desejar contribuir ou modificar o código-fonte:
```bash
git clone https://github.com/seu-usuario/BasinRAG.git
cd BasinRAG
pip install -e .[dev,api]
```

## 3. Passo a Passo Prático

### Passo 1: Ingestão de Documentos
Você pode ingerir diretórios contendo arquivos PDF, TXT ou MD usando o SDK Python.

```python
from basinrag.factory import BasinRAG, BasinRAGConfig

# Configuração básica
config = BasinRAGConfig(
    persist_dir="./.basinrag",
    embedding_model="all-MiniLM-L6-v2"
)

# Inicialização e Ingestão
engine = BasinRAG.from_directory(
    directory="./meus_documentos",
    config=config
)

print(f"Ingestão concluída. Documentos indexados: {engine.get_stats()['total_documents']}")
```

### Passo 2: Executando Queries de Busca Híbrida e Global
O método `query` aproveita o RRF (Reciprocal Rank Fusion) para mesclar buscas semânticas (FAISS) e lexicais (BM25), otimizadas pelo prior topológico das Bacias.

```python
resposta = engine.query(
    "Quais são os principais fatores de risco mencionados nos relatórios?",
    top_k=5,
    rerank=True
)

print(f"Resposta Gerada: {resposta.answer}")
for ctx in resposta.contexts:
    print(f"Fonte: {ctx.filename} - Score: {ctx.score:.4f}")
```

### Passo 3: Chat Interativo em Streaming
Para aplicações conversacionais, o BasinRAG suporta streaming nativo no terminal ou via API.

```python
for chunk in engine.chat_stream("Explique o impacto da nova regulamentação no setor."):
    print(chunk, end="", flush=True)
```

### Passo 4: Subindo o Servidor FastAPI
Inicie a API REST e WebSockets embutida.
```bash
python -m basinrag.server --port 8000 --host 0.0.0.0
```

### Passo 5: Containerização com Docker
Para rodar rapidamente via container, utilizando o Docker Compose:
```bash
docker-compose up -d
```
O serviço estará disponível em `http://localhost:8000`. Acesse a documentação Swagger em `http://localhost:8000/docs`.
