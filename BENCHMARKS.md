# BasinRAG — Benchmarks & Relatórios Empíricos Oficiais

Este documento consolida os resultados empíricos, metodologia e diretrizes de reprodução dos benchmarks do **BasinRAG**, incluindo a avaliação comparativa frente a uma implementação baseline do paradigma **GraphRAG** (Knowledge Graph Walk com extração de entidades e difusão de vizinhança multi-hop).

---

## 📊 1. Resumo Executivo da Avaliação Comparativa

O BasinRAG foi avaliado em dois cenários padrão da literatura de Information Retrieval (IR) e Code Intelligence:
1. **BEIR Benchmark Oficial (SciFact)**: Avaliação acadêmica rigorosa sobre 5.183 artigos científicos biomédicos usando o harness oficial `beir.retrieval.evaluation.EvaluateRetrieval`.
2. **SWE-bench Lite (Código de Produção)**: Avaliação de localização de bugs (*fault localization*) sobre repositórios reais do GitHub (`mwaskom/seaborn`, `pallets/flask`, `psf/requests`).

### Quadro Comparativo Geral

| Dimensão de Teste | Métrica Chave | GraphRAG Baseline | BasinRAG | Variação Relativa |
| :--- | :--- | :---: | :---: | :---: |
| **BEIR (SciFact)** | **NDCG@10** *(Ranking)* | 0.318 | **0.771** | **+142.5% (+0.453)** |
| **BEIR (SciFact)** | **Recall@10** *(Cobertura)* | 59.5% | **85.8%** | **+26.3 pp (+44.2%)** |
| **BEIR (SciFact)** | **MRR@10** | 0.236 | **0.750** | **+217.8% (+0.514)** |
| **BEIR (SciFact)** | **Tempo de Indexação** | 276.7s | **90.8s** | **3.05x mais rápido** |
| **BEIR (SciFact)** | **Latência de Query** | 11,059.4ms | **2,498.7ms** | **4.43x mais rápido** |
| **SWE-bench Lite** | **Hit@1** *(No Topo)* | 23.1% | **30.8%** | **+7.7 pp (BasinRAG)** |
| **SWE-bench Lite** | **Hit@10** *(Top 10)* | 69.2% | **84.6%** | **+15.4 pp (BasinRAG)** |
| **SWE-bench Lite** | **MRR** *(Mean Reciprocal Rank)* | 0.374 | **0.434** | **+0.060 (BasinRAG)** |
| **SWE-bench Lite** | **Tempo de Grafo** | 36.7s | **18.3s** | **2.00x mais rápido** |

---

## 🔬 2. Benchmark Oficial BEIR (Dataset: SciFact)

O **BEIR** ([beir-cellar/beir](https://github.com/beir-cellar/beir)) é o benchmark padrão-ouro para avaliação heterogênea de Information Retrieval.

### Configuração Experimental
- **Dataset**: `scifact` (scientific claim verification abstracts)
- **Documentos no Corpus**: 5.183 abstracts
- **Queries Avaliadas**: 50 queries do split `test`
- **Anotações Ground-Truth**: `qrels/test.tsv` com relevância binária de especialistas
- **Harness**: `beir.retrieval.evaluation.EvaluateRetrieval`

### Tabela Detalhada de Resultados

| Métrica Oficial BEIR | GraphRAG Baseline | BasinRAG | Delta Absoluto | Delta Relativo |
| :--- | :---: | :---: | :---: | :---: |
| **NDCG@10** *(Score Oficial Leaderboard)* | 0.318 | **0.771** | **+0.453** | **+142.5%** |
| **NDCG@5** | 0.246 | **0.748** | **+0.501** | **+204.1%** |
| **Recall@10** | 59.5% | **85.8%** | **+26.3 pp** | **+44.2%** |
| **Recall@5** | 37.0% | **79.5%** | **+42.5 pp** | **+114.9%** |
| **MRR@10 (Mean Reciprocal Rank)** | 0.236 | **0.750** | **+0.514** | **+217.8%** |
| **MAP@10 (Mean Average Precision)** | 0.234 | **0.738** | **+0.504** | **+215.4%** |
| **Precision@10** | 6.4% | **10.0%** | **+3.6 pp** | **+56.3%** |
| **Tempo de Construção do Grafo** | 276.68s | **90.80s** | **-185.89s** | **3.05x mais rápido** |
| **Latência Média por Consulta** | 11,059.4ms | **2,498.7ms** | **-8,560.6ms** | **4.43x mais rápido** |

*Dados brutos salvos em: `logs/beir_comparison_scifact_latest.json`.*

---

## 🏆 2.1 Pacote de Submissão Oficial do MTEB (Hugging Face Leaderboard)

O **MTEB (Massive Text Embedding Benchmark)** é a suíte oficial do Hugging Face que alimenta o leaderboard global de Retrieval ([huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)).

Executamos a avaliação oficial com a biblioteca `mteb` v2.20.5 cobrindo **todas as 300 queries de teste** do SciFact (5.183 documentos, `top_k=1000`):

### Métricas Consolidadas MTEB (300 Queries de Teste — Pós-Auditoria & Correções)
* **NDCG@10**: **0.650** (0.6497) *(+3.6% vs v1.0.0-rc.1 baseline)*
* **MRR@10**: **0.629** (0.6294) *(+9.1% vs v1.0.0-rc.1 baseline)*
* **MAP@10**: **0.619** (0.6188) *(+9.5% vs v1.0.0-rc.1 baseline)*
* **Hit@1 (Hit Rate no 1º lugar)**: **57.0%**
* **Hit@5**: **71.3%**
* **Hit@10**: **74.7%**
* **Recall@100**: **87.0%**
* **Recall@1000**: **98.7%**
* **Tempo Total de Execução**: 3h 50m (300 queries x 4.000 nós avaliados na CPU)

### Arquivos Oficiais Gerados para Submissão
O runner oficial do MTEB gerou a estrutura exata exigida pelo Hugging Face e pelo benchmark:
- 📁 `results/mteb/results/Basinfy__BasinRAG/1.0.3/SciFact.json` (Métricas oficiais da tarefa)
- 📁 `results/mteb/results/Basinfy__BasinRAG/1.0.3/model_meta.json` (Metadados estruturados Pydantic)
- 📁 `results/mteb/mteb_metadata.md` (Metadados YAML frontmatter gerados conforme [huggingface.co/blog/mteb](https://huggingface.co/blog/mteb))
- 📁 `results/mteb/SciFact_predictions.json` (Dump bruto de predições e scores)

### Ritos Oficiais de Submissão ao Leaderboard (Hugging Face & MTEB):

Conforme documentado no artigo oficial do Hugging Face ([MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb)), existem dois ritos complementares para ranqueamento público:

#### Rito 1: Publicação via Metadados no Hugging Face Hub (Recomendado & Automático)
O leaderboard do Hugging Face rastreia modelos no Hub que possuem a tag `mteb` e um `model-index` estruturado no frontmatter do seu `README.md`:
1. Gerar o arquivo de metadados atualizado a qualquer momento:
   ```powershell
   python -m basinrag.eval.generate_mteb_metadata
   ```
2. Abra o repositório do modelo no Hugging Face Hub (`https://huggingface.co/Basinfy/BasinRAG`).
3. Copie o bloco YAML contido em `results/mteb/mteb_metadata.md` e cole-o no topo do arquivo `README.md` (antes de qualquer conteúdo textual).
4. Faça commit no Hub. O [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) fará o crawling do seu model card e exibirá as métricas automaticamente!

#### Rito 2: Pull Request no Repositório do MTEB (Inclusão no Cache Estático Oficial)
1. Faça o fork do repositório oficial de resultados do MTEB: `https://github.com/embeddings-benchmark/results`
2. Clone seu fork e copie a pasta de resultados (atenção: nunca envie predições brutas pesadas):
   ```bash
   cp -r results/mteb/results/Basinfy__BasinRAG/ results/
   ```
3. Abra um Pull Request com o título: `[Model] Add BasinRAG results on SciFact`.
4. Uma vez mergeado pelos mantenedores do MTEB, o **BasinRAG** passa a constar no histórico estático do leaderboard!

---


## 💻 3. Benchmark SWE-bench Lite (Fault Localization em Código)

O **SWE-bench Lite** ([arXiv:2310.06770](https://arxiv.org/abs/2310.06770)) avalia a capacidade do RAG de localizar arquivos modificados por pull requests reais que resolveram issues em projetos open source.

### Configuração Experimental
- **Repositórios**: `mwaskom/seaborn`, `pallets/flask`, `psf/requests`
- **Instâncias**: 13 instâncias reais do SWE-bench Lite
- **Condições**:
  - Chunks de 1.200 caracteres (overlap de 150 caracteres).
  - Blindagem de contaminação: testes (`test_*.py`, diretórios `tests/`, `testing/`) e documentações (`docs/`) estritamente removidos.
  - Query de entrada: texto exato do `problem_statement` da issue.
  - Ground Truth: arquivos Python contidos no patch gold oficial da issue.

### Tabela Detalhada de Resultados

| Métrica de Fault Localization | GraphRAG Baseline | BasinRAG | Variação Observada |
| :--- | :---: | :---: | :---: |
| **Hit@1** *(Arquivo correto no topo)* | 23.1% | **30.8%** | **+7.7 pp (BasinRAG)** |
| **Hit@5** *(Arquivo correto no Top 5)* | 53.8% | 53.8% | Paridade |
| **Hit@10** *(Arquivo correto no Top 10)* | 69.2% | **84.6%** | **+15.4 pp (BasinRAG)** |
| **Recall@10** | 69.2% | **84.6%** | **+15.4 pp (BasinRAG)** |
| **MRR (Mean Reciprocal Rank)** | 0.374 | **0.434** | **+0.060 (BasinRAG)** |
| **Tempo Médio de Indexação do Grafo** | 36.71s | **18.32s** | **BasinRAG 2.00x mais rápido** |
| **Latência Média de Query** | 266.1ms | 7,477.4ms | GraphRAG mais rápido (sem reranker) |

*Dados brutos salvos em: `logs/basinrag_vs_graphrag_latest.jsonl`.*

---

## 🧠 4. Análise Teórica e Comparativa de Dinâmica de Grafos

### Análise de Densidade de Conexões em Grafos de Co-ocorrência (*Hub Congestion*)
Modelos baseados em grafos bipartidos discretos de Entidade $\leftrightarrow$ Documento utilizam expansão por co-ocorrência e ponderação IDF:
1. **Concentração de Densidade em Hubs**: Em bases de código (como `requests`), entidades e identificadores utilitários comuns (`encode`, `detect`, `buffer`, `string`) ocorrem com frequência elevada em submódulos de apoio. Em grafos de co-ocorrência não-confinados, termos de alta frequência tendem a acumular probabilidade durante caminhadas aleatórias, o que pode dispersar o ranking em relação aos módulos que contêm a lógica central (`sessions.py`, `models.py`, `adapters.py`). No benchmark do `requests`, esse comportamento de dispersão reduziu a cobertura nas instâncias avaliadas.
2. **Crescimento de Arestas em Vizinhanças 2-Hop**: A expansão de vizinhanças de ordem superior sobre matrizes de co-ocorrência densas aumenta substancialmente o número de transições ativas, elevando a latência de recuperação conforme o corpus cresce.

### Abordagem Baseada em Topologia de Bacias e Confinamento de Fluxo
1. **Variedade Métrica e Atratores Topológicos**: Em vez de conectar nós por co-ocorrência estocástica irrestrita, o BasinRAG estrutura os fragmentos ao longo do fluxo documental e projeta **árvores $\rho$ de gradiente métrico** em torno de **atratores de bacia**. O fluxo de ativação é restrito à componente funcional relevante, atenuando a dispersão causada por termos de alta frequência.
2. **Difusão Espectral Controlada (PPR)**: A difusão dentro do subgrafo induzido via **Personalized PageRank (PPR)** dissipa a energia de busca estritamente dentro da componente de contexto ativo.
3. **Fusão Canônica Min-Max RRF**: Integra vetores densos normalizados, BM25 esparso calibrado e Cross-Encoder mMARCO, equilibrando termos léxicos exatos com alinhamento semântico global.

---

## 🔁 5. Como Reproduzir os Benchmarks

### Reproduzindo o Benchmark Oficial BEIR (SciFact)
```powershell
# Executa o comparativo BEIR oficial com 50 queries
python -m basinrag.eval.compare_beir --dataset scifact --num-queries 50 --top-k 10

# Para avaliar todas as 300 queries do split de teste
python -m basinrag.eval.compare_beir --dataset scifact --num-queries 300 --top-k 10
```

### Reproduzindo o Benchmark SWE-bench Lite (Código)
```powershell
# Executa o comparativo no SWE-bench Lite nos repositórios flask, requests e seaborn
python -m basinrag.eval.compare_graphrag --repo-filter "pallets/flask,psf/requests,mwaskom/seaborn" --limit 13 --top-k 10
```

---

Para uma análise comparativa aprofundada incluindo GraphRAG, HippoRAG, LightRAG e baselines do MTEB, consulte o [Relatório Comparativo Global](docs/BENCHMARK_COMPARISON_pt.md).
