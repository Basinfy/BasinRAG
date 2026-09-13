# Protocolo de benchmarks do BasinRAG

## Estado das evidências

Os resultados em `results/gate/`, `results/mteb*/` e `results/qasper*/` são artefatos históricos. Parte deles antecede o código e as alterações locais em avaliação; a decisão gravada em `results/gate/decision.json` não valida esta revisão. Também não publique a média dos arquivos MTEB existentes como se todos tivessem sido produzidos pela mesma execução.

Até uma nova execução limpa ser concluída, **não há pontuação atual de release declarada neste documento**. Os números históricos podem ser consultados nos artefatos originais, mas não são uma linha de base reproduzida da revisão atual.

## Protocolo reproduzível

Execute cada avaliação com um diretório de saída novo e exclusivo. Não reutilize nem mescle diretórios de runs anteriores. Grave a revisão do código, versão de Python e dependências, encoder/tokenizer, corpus e split, parâmetros, tarefas solicitadas e concluídas, e o hardware usado. Um score agregado só representa uma bateria completa se todas as tarefas esperadas concluírem com sucesso nesta mesma execução.

No PowerShell:

```powershell
$run = "results/runs/gate-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m basinrag.eval.run_gate --output $run --skip-rerank

$run = "results/runs/mteb-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m basinrag.eval.run_mteb `
  --no-rerank `
  --tasks SciFact,NFCorpus,FiQA2018,ArguAna,SCIDOCS `
  --output $run
```

O relatório MTEB deve apontar somente os artefatos das tarefas concluídas na execução registrada e usar caminhos relativos ao diretório do run. Se qualquer tarefa falhar ou for pulada, mostre o status parcial e não apresente a média como resultado completo de cinco tarefas. Para evitar mistura, use outro diretório se precisar repetir uma execução.

## Métricas de retrieval

Compare encoder denso, BM25, RRF e modo topológico separadamente, usando o mesmo corpus, split, consultas, candidatos e parâmetros de avaliação. Registre `nDCG@10`, `Recall@10`, `MRR@10`, ordem dos IDs recuperados, latência p50/p95, tempo de ingestão, pico de memória e espaço em disco. Fixe a semente e os embeddings dos testes sintéticos; não use `hash()` do Python para gerar dados determinísticos.

O modo `hybrid_rrf` é a referência de produção. O `experimental_topology` só deve ser considerado após ablação; o controle sem topologia precisa ser invariável às bacias, hops e arestas virtuais. Em validações fixas, os limites de comparação definidos pelo projeto são: não aceitar regressão superior a `0.005` em nDCG@10 nem `0.01` em Recall@10. Sem um SLA explícito, só chamar uma alteração de otimização de desempenho se ela melhorar pelo menos 10% uma métrica medida sem ultrapassar esses limites de qualidade.

`confidence` não é uma probabilidade calibrada. Para avaliar abstenção, use um conjunto de validação independente e reporte calibração, falsos aceites e falsas abstenções; não interprete o score de ordenação como confiança calibrada.

## SWE-bench

O benchmark padrão do BasinRAG mede localização/recuperação de arquivos — por exemplo, Recall@k, Any Recall e All Recall. Checkpoints devem corresponder ao mesmo split, protocolo e subconjunto de instâncias; resultados de subconjuntos diferentes não podem ser combinados. Caminhos oficiais são comparados após normalizar separadores e `./`, sem correspondência fuzzy.

Essa recuperação **não é `% Resolved`**. O indicador oficial só pode ser publicado quando patches reais forem avaliados pelo harness Docker oficial, no conjunto e protocolo correspondentes. JSONL de predições de localização sem patch não é evidência de resolução.

## Defaults e configuração registrados

- Encoder padrão: `BAAI/bge-base-en-v1.5` com o prefixo BGE de consulta.
- Ranking padrão: BM25 + FAISS com RRF; `experimental_topology` é opt-in.
- Chunk legado: `chunk_size=512` e `chunk_overlap=128`, em caracteres. Opções de tokens são aditivas e devem ser registradas quando usadas.
- Reranking: documente se habilitado. A arena MTEB reproduzível acima desliga reranking com `--no-rerank`.
- Índice, tokenizer, tamanho/overlap de chunks e batch de embeddings: inclua os valores efetivos no protocolo do run.

Não altere limiares de Flat/HNSW ou pesos de ranking com base em um único run. Para avaliar ANN, compare Recall@k contra busca Flat em uma amostra fixa e informe latência, memória e configuração do índice.

## Fontes dos dados

- Gate SciFact/long-document: `basinrag.eval.run_gate`.
- Tarefas padronizadas MTEB: `basinrag.eval.run_mteb`.
- Avaliação de localização SWE-bench: `basinrag.eval.swebench`.

Os resultados dependem das versões dos datasets, dependências, modelos e parâmetros. Preserve o manifesto e os arquivos brutos de cada execução junto do relatório.
