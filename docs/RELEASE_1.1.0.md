# Release 1.1.0 — migração e validação

Este documento acompanha a preparação do BasinRAG 1.1.0. A versão ainda não deve ser tratada como publicada até a CI da branch passar e o piloto de reindexação ser aprovado. Resultados antigos permanecem históricos; esta release não aprova claims quantitativos de qualidade nem de superioridade topológica.

> Atenção: `1.1.0` é o identificador solicitado para esta entrega. O formato interno de snapshot mudou e rotas da API não têm prefixo de versão; consumidores 1.x ainda devem avaliar a migração como potencialmente incompatível.

## Mudanças de contrato

- O leitor aceita o formato atual de snapshot. Índices legados não são atualizados automaticamente nem escritos durante a tentativa de abertura.
- O diretório padrão é `.basinrag`. Para rollback, mantenha o pacote e o root anteriores intactos.
- As rotas são `/query`, `/chat`, `/livez` e `/readyz`, sem prefixo de versão. A API exige `Authorization: Bearer`; não envie credenciais na URL.
- `hybrid_rrf` é o ranking padrão. A expansão topológica é experimental. Sumarização L3 em background vem desligada; egress de trechos para um provedor remoto requer os dois opt-ins.
- A versão de pacote é `1.1.0`; o número não altera a exigência de migração incompatível descrita acima.

## Migração segura

1. Faça backup das fontes originais e mantenha o índice anterior sem alterações.
2. Construa um snapshot em uma pasta nova/vazia, explicitando o destino:

   ```bash
   basinrag reindex /srv/basinrag/documents --storage-dir /srv/basinrag/.basinrag
   ```

3. Valide queries representativas, citações, `/readyz`, sincronização e restart usando apenas o novo root.
4. Promova o serviço somente depois do piloto. Para rollback, volte ao pacote anterior e ao root anterior; não copie artefatos novos sobre o índice antigo.

`ingest` é aditivo: uma fonte alterada deve ser processada por `sync`. `sync` trata a raiz como escopo completo e remove fontes ausentes; não o execute contra uma pasta temporária ou incompleta. Consulte [Deploy](DEPLOYMENT.md) para configuração de serviço e segurança de rede.

## Segurança e operação

- Sem chave, apenas `BASINRAG_ENV=local_dev`, `BASINRAG_ALLOW_KEYLESS_LOCAL=true` e bind loopback permitem desenvolvimento local.
- Configure uma allowlist de origens e mantenha um processo por diretório de snapshot. Proxy confiável pode influenciar rate limiting, nunca autenticação.
- Mantenha L3 desligado salvo necessidade operacional explícita. `BASINRAG_ENABLE_BACKGROUND_L3=true` ativa geração; `BASINRAG_ALLOW_REMOTE_L3_EGRESS=true` autoriza envio de trechos ao provedor remoto.
- O processo não tem como confirmar segurança ou qualidade do conteúdo-fonte. Documentos recuperados são evidência não confiável, não instruções.

## Gate científico

Gate de release de código e gate de publicação de métricas são independentes. SciFact e QASPER precisam de execuções completas e provenance/revisões compatíveis. Amostras limitadas, fallback para ArXiv ou cache incompatível invalidam a decisão. ArXiv é um experimento separado e não substitui QASPER. Até um protocolo completo e reproduzível passar, publique a release sem números comparativos nem claims de causalidade.

## Validação local desta preparação

Executado em Windows, Python 3.11:

| Verificação | Resultado |
| --- | --- |
| Suíte completa | 359 testes passaram |
| Cobertura global | 80,87% (limite 70%) |
| API / persistência / gate | 95% / 87% / 100% (limites 85%) |
| Ruff e Mypy | passaram; Mypy sem erros em 48 módulos |
| Lockfile e build | `uv lock --check` e build wheel/sdist passaram |
| Wheel e sdist | smoke de instalação em diretórios isolados passou; dependências já presentes no ambiente |
| Auditoria de dependências | `pip-audit` sem vulnerabilidades conhecidas nas dependências pinadas |
| Docker | não verificado localmente: daemon indisponível; job de build permanece na CI |
| Matriz de SO/Python | Windows 3.11 foi testado localmente; Python 3.10–3.12 e Docker dependem dos jobs da CI |

Comandos centrais de validação:

```bash
uv run ruff check .
uv run mypy basinrag
uv run python -m pytest --cov=basinrag --cov-report=term-missing --cov-fail-under=70 tests/
uv build --no-progress
```

Os três avisos da suíte completa são de depreciação em dependências externas (`starlette`, `mteb`/SentenceTransformers e PyTorch); não causaram falhas.
