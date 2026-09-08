# Índice Geral da Documentação do BasinRAG

Bem-vindo à documentação oficial do **BasinRAG**, o sistema avançado de Geração Aumentada por Recuperação (RAG) impulsionado por Análise Topológica de Dados.

Este índice serve como o ponto central para explorar todos os manuais, referências e guias do projeto. Escolha o guia mais adequado ao seu perfil e necessidade atual.

## Navegação Rápida

| Guia | Descrição | Público-Alvo |
|------|-----------|--------------|
| [🚀 Quickstart](QUICKSTART.md) | Guia prático e direto para instalar, configurar e rodar sua primeira query e chat interativo. | Desenvolvedores, Novos Usuários |
| [🧠 Fundamentação Teórica](THEORY.md) | Detalhes sobre Atração Topológica, Particionamento de Bacias e Grafo de Vizinhança. | Pesquisadores, Cientistas de Dados |
| [🏗️ Especificação Arquitetural](../ARCHITECTURE.md) | Visão aprofundada da engenharia do sistema, pipelines de ingestão e recuperação. | Arquitetos, Engenheiros de Software |
| [📊 Benchmarks Oficiais](../BENCHMARKS.md) | Métricas de precisão, recall e latência em comparação com abordagens tradicionais. | Engenheiros de Machine Learning |
| [🌐 Relatório Comparativo Global](BENCHMARK_COMPARISON_pt.md) | Estudo comparativo com GraphRAG, HippoRAG, OpenAI e baselines globais MTEB. | Arquitetos, Pesquisadores |
| [🔍 Relatório de Auditoria Histórica (v1.0.0)](archive/AUDIT_REPORT_v1.0.0.md) | Diagnóstico completo de segurança, performance, consistência matemática e endurecimento. | Mantenedores, Auditores |
| [📚 Referência de API](API_REFERENCE.md) | Documentação completa de todas as classes, métodos, configurações e endpoints REST/WebSocket. | Desenvolvedores Back-end |
| [⚙️ Guia de Implantação](DEPLOYMENT.md) | Manual de engenharia para colocar o BasinRAG em ambiente produtivo, escalabilidade e segurança. | DevOps, SREs, SysAdmins |
| [🌐 Arquitetura Multilíngue](MULTILINGUAL.md) | Guia de recuperação cross-lingual, stemmers dos 10 principais idiomas e re-ranking mMARCO. | Engenheiros de IA, Arquitetos |
| [💧 Exemplo Prático de Bacias](exemplo_bacias.md) | Walkthrough de como a topologia atua em um corpus de teste. | Analistas, Curiosos |

## Mapa de Leitura Sugerido

### Para Desenvolvedores e Engenheiros
1. Inicie pelo **[Quickstart](QUICKSTART.md)** para colocar o sistema no ar rapidamente.
2. Explore a **[Referência de API](API_REFERENCE.md)** para integrar o BasinRAG às suas aplicações via SDK Python ou REST.
3. Leia o **[Guia de Implantação](DEPLOYMENT.md)** antes de mover sua aplicação para produção.

### Para Pesquisadores e Cientistas
1. Comece pela **[Fundamentação Teórica](THEORY.md)** para entender os conceitos matemáticos que baseiam nosso algoritmo.
2. Analise os **[Benchmarks Oficiais](../BENCHMARKS.md)** para avaliar ganhos reais em topologias não-lineares.
3. Teste o **[Exemplo Prático](exemplo_bacias.md)** para visualizar o comportamento real dos atratores.