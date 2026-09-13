# Implantação do BasinRAG

Este guia descreve a topologia suportada pelo BasinRAG: **um processo em um host**, com snapshots de índice e uma API FastAPI opcional. O tamanho adequado de CPU, memória e disco depende do encoder, reranker, corpus e modelo de geração. Meça com o corpus e hardware de produção; este projeto não publica uma garantia genérica de QPS.

## Instalação

Use Python 3.10, 3.11 ou 3.12 e instale o pacote com os extras necessários. Para servir a API com Ollama:

```bash
python -m pip install ".[api,ollama]"
```

Para OpenAI, instale `.[api,openai]`. A avaliação e as ferramentas de desenvolvimento são extras separados: `.[dev,api,eval]`. O primeiro uso pode baixar os pesos do SentenceTransformers e do reranker, portanto prepare o cache ou permita acesso ao repositório de modelos antes de iniciar em rede restrita.

## Configuração

O índice v3 fica em `BASINRAG_STORAGE_DIR` (padrão `.basinrag-v3`). Configure variáveis no ambiente do serviço ou use `.env` para desenvolvimento local. Precedência: **argumentos explícitos > ambiente do processo > `.env` > padrões**. Não armazene segredos de produção no repositório.

Antes de expor a API, defina pelo menos:

```dotenv
BASINRAG_ENV=production
BASINRAG_API_KEY=gere-um-segredo-aleatorio-longo
BASINRAG_CORS_ORIGINS=https://app.example.com
BASINRAG_STORAGE_DIR=/var/lib/basinrag
BASINRAG_ENABLE_BACKGROUND_L3=false
BASINRAG_ALLOW_REMOTE_L3_EGRESS=false
```

`BASINRAG_CORS_ORIGINS` recebe origens separadas por vírgula e wildcard não é aceito. CORS não substitui autenticação. Mantenha o diretório do snapshot persistente entre reinícios. Sem chave, a API não inicia, exceto com `BASINRAG_ENV=local_dev`, `BASINRAG_ALLOW_KEYLESS_LOCAL=true` e bind loopback. Proxies confiáveis afetam apenas rate limiting.

As opções `BASINRAG_CHUNK_SIZE` e `BASINRAG_CHUNK_OVERLAP` continuam em caracteres (padrões 512/128). Os campos opcionais `BASINRAG_CHUNK_SIZE_TOKENS` e `BASINRAG_CHUNK_OVERLAP_TOKENS` permitem segmentação em tokens, limitada ao contexto do encoder. O manifesto do índice registra modelo e política de chunking. Alterar o encoder ou tokenizer exige `basinrag reindex <diretorio>`; o sistema não converte silenciosamente um índice existente.

A ingestão de diretórios e datasets JSON percorre arquivos ou documentos em sequência e usa um spool temporário para não manter uma segunda lista completa de chunks. O grafo NetworkX e o índice FAISS ainda residem em memória; durante uma publicação, o snapshot ativo e o candidato coexistem para manter leitores sem interrupção. Dimensione RAM conforme o corpus e não trate esse fluxo como armazenamento out-of-core.

## Construir e sincronizar o índice

```bash
basinrag ingest /srv/basinrag/documents
```

`ingest` mescla documentos; fontes que desapareceram não são removidas. Para espelhar as fontes presentes numa raiz — substituindo os chunks de arquivos alterados e removendo os que sumiram — execute:

```bash
basinrag sync /srv/basinrag/documents
```

Use `basinrag reindex /srv/basinrag/documents --storage-dir /srv/basinrag/.basinrag-v3` para reconstruir o índice inteiro em destino novo/vazio. O leitor v2 não modifica índices antigos; mantenha o root anterior para rollback. Aponte `sync` para a raiz que define o escopo de sincronização e evite usá-lo com uma pasta temporária ou incompleta.

Cada gravação constrói uma geração em staging, valida os artefatos e publica `current.json` atomicamente. Um escritor por vez é permitido. Consultas existentes permanecem presas ao snapshot capturado enquanto a nova geração é publicada; mantenha o volume de armazenamento no mesmo host e faça backup/restaure o diretório inteiro como conjunto. A geração atual e a anterior são mantidas para recuperação.

## API

### Iniciar pelo CLI

O caminho recomendado é o entrypoint instalado:

```bash
basinrag serve --host 127.0.0.1 --port 8000
```

O bind público exige `BASINRAG_API_KEY`. Prefira bind loopback atrás de um proxy reverso que forneça TLS e controles de rede. A aplicação ASGI é `basinrag.api.server:app`; para Uvicorn direto:

```bash
uvicorn basinrag.api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Clientes precisam apresentar `Authorization: Bearer <key>` mesmo quando a aplicação é iniciada diretamente como ASGI. Não passe credenciais em query strings. Não aumente `--workers`: a publicação e os limites de taxa são locais ao processo.

Rotas disponíveis:

- `GET /v2/livez` — processo ativo.
- `GET /v2/readyz` — snapshot e dependências válidos; retorna 503 enquanto indisponível.
- `POST /v2/query` — `search_type` em `auto|local|global|hybrid`, `top_k` de 1 a 50; requer snapshot carregado.
- `WS /v2/chat` — eventos JSON com referências validadas, erro e conclusão; origem validada e desconexão cancela geração.

Mensagens WebSocket têm limite de 8 KiB; o processo limita conexões e gerações simultâneas e encerra geração após 120 segundos. Contadores são em memória e valem apenas para este processo. L3 está desligado por padrão; sumarização remota só recebe trechos com os dois opt-ins.

### Serviço systemd

Crie um ambiente virtual e instale BasinRAG no diretório da aplicação. Mantenha as variáveis secretas em `/etc/basinrag.env`, com permissões restritas:

```ini
[Unit]
Description=BasinRAG API
After=network.target

[Service]
User=basinrag
Group=basinrag
WorkingDirectory=/opt/basinrag
Environment="PATH=/opt/basinrag/venv/bin"
EnvironmentFile=/etc/basinrag.env
ExecStart=/opt/basinrag/venv/bin/basinrag serve --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Ative o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now basinrag.service
```

### Docker

Exemplo mínimo, sem depender de um `requirements.txt` que não faz parte do pacote:

```dockerfile
FROM python:3.11-slim
ARG BASINRAG_PROVIDER_EXTRA=ollama
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 basinrag \
    && mkdir -p /data && chown basinrag:basinrag /data
COPY pyproject.toml README.md LICENSE ./
COPY basinrag ./basinrag
RUN python -m pip install --no-cache-dir ".[api,${BASINRAG_PROVIDER_EXTRA}]"
ENV BASINRAG_STORAGE_DIR=/data/.basinrag-v3
EXPOSE 8000
USER 10001:10001
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v2/livez', timeout=2)"
CMD ["basinrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

Monte `/data` em volume persistente e passe `BASINRAG_API_KEY` e `BASINRAG_CORS_ORIGINS` por um gerenciador de segredos. O `.dockerignore` exclui ambiente, índices, resultados, logs e caches. Não publique a porta diretamente na Internet sem TLS e controles de rede.

## Segurança e conteúdo recuperado

O índice pode conter texto malicioso ou instruções direcionadas ao modelo. Trate documentos recuperados como dados não confiáveis: delimite-os no prompt da aplicação e instrua o LLM a usá-los como evidência, não como comandos. Isso reduz ambiguidade, mas não é uma garantia contra prompt injection. Não há um `enable_safety` ou sanitizador de documentos no `BasinRAGConfig`.

Use origens CORS explícitas, API key de alta entropia, proxy TLS, permissões mínimas de arquivo e limites de rede. Os limites de 30/minuto não são compartilhados entre processos; mantenha uma única instância até haver um mecanismo distribuído suportado.

## Operação e validação

Antes de promover uma mudança, rode a suíte de testes e Ruff, reindexe uma cópia de corpus representativa e compare qualidade de recuperação, latência p50/p95, tempo de ingestão, memória e espaço em disco. Mantenha a geração anterior até validar o snapshot novo. Consulte [BENCHMARKS.md](../BENCHMARKS.md) para não misturar resultados de execuções diferentes.
