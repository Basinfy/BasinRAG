# Índice da documentação do BasinRAG

BasinRAG é um RAG híbrido **BM25 + FAISS**. Bacias particionam o índice e o briefing hidrata irmãos da árvore ρ depois das sementes RRF. Hop/DRF no ranking são `experimental_topology`. Métricas históricas não são claims de release.

## Navegação

| Guia | Descrição | Público |
|------|-----------|---------|
| [README](../README_pt.md) / [English](../README.md) | Visão do produto | Todos |
| [Quickstart](QUICKSTART.md) | Instalar, ingest, query, chat | Desenvolvedores |
| [Ranking](RANKING.md) | `hybrid_rrf` vs experimental; flat vs long-doc; CE opt-in | Engenheiros |
| [API](API_REFERENCE.md) | SDK, config, HTTP e WebSocket | Backend |
| [Arquitetura](../ARCHITECTURE.md) | Ingestão, persistência, briefing | Engenheiros |
| [Deploy](DEPLOYMENT.md) | Produção, auth, rede | DevOps |
| [Avaliação](EVAL.md) | Gate, MTEB, medições pontuais | ML |
| [Protocolo de benchmarks](../BENCHMARKS.md) | Reprodução e limites de claims | ML / pesquisa |
| [Teoria](THEORY.md) | φ, atratores, ρ; hop formula = experimental | Pesquisadores |
| [Multilíngue](MULTILINGUAL.md) | Stemming, encoder, o que *não* é garantido | IA |
| [Release 1.1.0](RELEASE_1.1.0.md) | Migração e validação (ainda unreleased) | Mantenedores |
| [Contribuir](../CONTRIBUTING.md) | Dev, testes, claims | Contribuidores |
| [Segurança](../SECURITY.md) | Versões suportadas e reporte | Todos |

## Mapa de leitura

### Começar
1. [Quickstart](QUICKSTART.md)
2. [Ranking](RANKING.md)
3. [API](API_REFERENCE.md)

### Operar
1. [Deploy](DEPLOYMENT.md)
2. [Release 1.1.0](RELEASE_1.1.0.md)

### Medir
1. [Avaliação](EVAL.md)
2. [Protocolo de benchmarks](../BENCHMARKS.md)

### Teoria
1. [Teoria](THEORY.md)
2. [Arquitetura](../ARCHITECTURE.md)

## Histórico (não é o produto atual)

- [Auditoria v1.0.4](archive/AUDIT_REPORT_v1.0.4.md)
- [Auditoria v1.0.0](archive/AUDIT_REPORT_v1.0.0.md)
- [Multilíngue v1.0.3 (arquivado)](archive/MULTILINGUAL_v1.0.3.md)
- [Papers LaTeX](../paper/README.md)
