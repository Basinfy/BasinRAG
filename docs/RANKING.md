# Ranking e briefing

Contrato de produção do BasinRAG 1.1.0. Números e como medir: [EVAL.md](EVAL.md) e [BENCHMARKS.md](../BENCHMARKS.md).

O ranking padrão é **`hybrid_rrf`**: BM25 + FAISS fundidos por RRF. Bacias **particionam o índice** e o `brief()` hidrata irmãos da árvore ρ **depois** das sementes. Hop e DRF **não** reordenam as sementes nesse modo.

## Modos

| Modo | Quando usar |
|---|---|
| `hybrid_rrf` (padrão) | Produção e gate. Topologia não muda a ordem das sementes. |
| `experimental_topology` | Opt-in. Hop/DRF/expand no pool RRF. Só após ablação no mesmo pool de candidatos. Nunca substitui as sementes por topologia local/global. |

`BASINRAG_RANKING_MODE` / `ranking_mode` selecionam o modo. `experimental_topology` não é o default.

## Dois caminhos de índice

O retriever escolhe constantes a partir da geometria do índice (`index_is_flat`): um nó por documento versus folhas fatiadas.

| | Flat (SciFact) | Long-doc (QASPER / papers) |
|---|---|---|
| Unidade | 1 documento = 1 nó | Folhas sentence-window (~1–2 sentenças); overflow por caracteres |
| α RRF | `0.15` | `0.40` |
| Pool | `max(50, top_k×5)` | `max(200, top_k×10)` |
| BM25 no RRF | Só vota dentro do pool denso | União denso ∪ expansão RM3 ∪ BM25@10 no membership |
| Rescue só-BM25 | Não | Até 2 papers, inserção na 3ª posição |
| Cross-encoder | Sempre ignorado | Só se `use_rerank=True` |
| Bacias no briefing | 1:1; não acrescentam contexto | Irmãos ρ hidratados após as sementes RRF |

Ingestão de produto é **adaptativa** (`chunk_policy_version=2`): documentos curtos (`len <= 4000` caracteres) usam o splitter legado **512 / 128**; documentos longos usam as mesmas folhas sentence-window **256 / 32** que o gate QASPER (`split_document_chunks` / `split_sentence_leaves`). Reindex é obrigatório para fontes longas já indexadas na política v1.

O gate usa o mesmo kernel de pool: `top_k` é o corte de saída e `candidate_k` vem de `HybridSearch.resolve_candidate_k`. Cross-encoder no gate também é ignorado em índices flat.

## Cross-encoder

Desligado por padrão (`use_rerank=False`, `BASINRAG_USE_RERANK=false`).

Ligue só com flag explícita. Mesmo ligado, índices flat não rerankam. Modelo: `BAAI/bge-reranker-v2-m3`, só após o pool híbrido em índices fatiados.

## Bacias (mapa, não score)

1. Grafo funcional φ: cada chunk tem no máximo um sucessor estrutural.
2. Atratores e bacias \(B(A_i)\) particionam o corpus.
3. Árvore ρ dá hops até o atrator. Em corpus flat, \(h(v)=0\).

Isso define **vizinhança para o LLM** (hubs, irmãos, L3 opcional). Não é a métrica de ranking em `hybrid_rrf`. A fórmula de prior \(S_{\text{RRF}}\cdot(\omega_{\min}+\cdots)\) em [THEORY.md](THEORY.md) aplica-se ao modo **experimental**, não ao ranking de produção.

## Router

`search_type`: `auto`, `hybrid`, `local`, `global`. Prefira `hybrid` para a referência BM25 + FAISS. `local`/`global` expandem contexto de briefing; no modo padrão não substituem a ordem RRF das sementes.
