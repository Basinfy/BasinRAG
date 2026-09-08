# ⚙️ Guia de Implantação e Produção

Este manual fornece diretrizes de engenharia para operar o **BasinRAG** em ambientes de produção de alta demanda.

## 1. Dimensionamento de Hardware

### CPU vs GPU
- **CPU (Ambiente Otimizado):** O BasinRAG pode operar de forma extremamente eficiente em CPU utilizando modelos exportados para **ONNX Runtime**. Para cargas até 10 QPS (Queries per Second), uma máquina virtual com 4-8 vCPUs (ex: AWS c6i.2xlarge) é suficiente.
- **GPU (PyTorch/SentenceTransformers):** Para throughput máximo na geração de embeddings e re-ranking (Cross-Encoder), recomendamos o uso de GPUs (NVIDIA T4, A10g ou A100). Ative o flag `use_gpu=True` na classe de configuração `BasinRAGConfig`.

### Memória RAM (FAISS HNSW)
O índice FAISS HNSW é armazenado e consultado diretamente em memória (RAM). Uma regra prática conservadora:
- 1 milhão de chunks (512 tokens, 384 dimensões) consome aproximadamente **2.5 GB** a **3 GB** de RAM, incluindo os grafos HNSW. Preveja memória adicional para o carregamento do modelo LLM se este for hospedado localmente na mesma máquina.

## 2. Persistência e Backup

O BasinRAG armazena seu estado atômico no diretório especificado por `persist_dir` (padrão `.basinrag/`).
- **Estratégia de Backup:** Execute um snapshot ou `rsync` do diretório para o S3 (ou armazenamento blob equivalente) periodicamente.
- **Integridade Atômica:** O BasinRAG implementa lock de arquivos de sistema (file-locks) durante a reescrita do índice para evitar corrupção caso o processo sofra interrupção (OOM kill ou SIGKILL) abrupta.

## 3. Cache Inteligente

Para queries idênticas ou semanticamente muito próximas, o BasinRAG Server integra um cache LRU (Least Recently Used) agressivo.
- **Estratégia de Invalidação:** Sempre que o endpoint `/ingest` for chamado com novos documentos, o cache de buscas semânticas é marcado como obsoleto e esvaziado, garantindo fresh-data em real-time.

## 4. Segurança

Ao expor a API REST do BasinRAG à internet pública ou VPC, recomenda-se:
1. **Autenticação:** Habilite o middleware de verificação de tokens passando o header de autorização `X-API-KEY`.
2. **Rate-Limiting:** O servidor possui integração com a biblioteca `SlowAPI`. Configure limites estritos, especialmente nos endpoints pesados como o `/query` (ex: 20 req/min/IP), para evitar exaustão de GPU ou chamadas excessivas.
3. **Mitigação de Prompt Injection:** Utilize o validador L3 (Level-3) embutido no `BasinRAGConfig(enable_safety=True)` que analisa, sanitiza e recusa comandos diretivos maliciosos presentes na query antes que sejam passados para a camada de recuperação de contexto.

## 5. Configuração como Serviço Linux (systemd)

Para bare-metal ou VMs padrão do EC2/Compute Engine, crie o arquivo de serviço systemd no caminho `/etc/systemd/system/basinrag.service`:

```ini
[Unit]
Description=BasinRAG API Service
After=network.target

[Service]
User=basinrag
Group=basinrag
WorkingDirectory=/opt/basinrag
Environment="PATH=/opt/basinrag/venv/bin"
Environment="BASINRAG_API_KEY=sua_chave_secreta_aqui"
ExecStart=/opt/basinrag/venv/bin/python -m basinrag.server --port 8000 --host 127.0.0.1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Ative e inicie o serviço com: `systemctl enable --now basinrag.service`

## 6. Containerização (Docker)

A distribuição via Docker é a forma mais recomendada para ambientes modernos, facilitando o deploy via Kubernetes ou ECS.

### Exemplo de `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependências de sistema para compilação (ex: FAISS, HNSW)
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Instalar a aplicação em modo editável / pacote regular
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "basinrag.server", "--port", "8000", "--host", "0.0.0.0"]
```

### Exemplo de `docker-compose.yml`
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - basinrag_data:/app/.basinrag
    environment:
      - BASINRAG_API_KEY=sua_chave_super_secreta
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  basinrag_data:
```
