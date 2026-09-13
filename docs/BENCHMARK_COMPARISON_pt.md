# Registro histórico: BasinRAG vs. outros paradigmas

> Arquivado: tabelas, métricas e alegações abaixo pertencem a versões e protocolos anteriores. Não descrevem BasinRAG 1.1.0 e não podem ser usados como claims de release.

**Atualizado:** setembro 2026  
**Números:** [`BENCHMARKS.md`](../BENCHMARKS.md)

---

## 1. Posicionamento

BasinRAG é um **RAG híbrido local (BM25 + FAISS)** com grafo funcional e **bacias como mapa de briefing** para o LLM.

| Eixo | Papel no produto |
| :--- | :--- |
| Ranking | Híbrido BM25 + FAISS |
| Topologia (bacias) | Vizinhança estruturada para briefing e busca `local` |
| Indexação | Local, sem LLM de entidades / comunidades |
| Entrega ao gerador | Lista ranqueada + hubs / vizinhos / L3 opcional |

---

## 2. Matriz por paradigma

| Paradigma | Exemplo | Index LLM? | Força |
| :--- | :--- | :---: | :--- |
| Dense puro | `bge-base`, OpenAI emb. | Não* | Ranking denso |
| Sparse | BM25 | Não | Termos raros / IDs |
| KG + LLM | GraphRAG, HippoRAG | **Sim** | Síntese / entidades |
| **BasinRAG** | este repo | **Não** | Híbrido + briefing local |

\*Embeddings locais ou API — distinto de extração de entidades.

### Métricas

Valores numéricos deste registro foram removidos por não atenderem ao protocolo atual. Não há métrica quantitativa vigente aprovada para esta release. Consulte [`BENCHMARKS.md`](../BENCHMARKS.md) para os critérios de uma execução futura.

---

## 3. Papel da topologia

Em documentos longos, $\phi$ / atratores / $\rho$ definem **vizinhança** para briefing e modo `local`.  
Em corpora flat (1 documento = 1 nó), o path de ranking é o híbrido.

---

## 4. SWE-bench

Não entra no claim global. O board mede **% Resolved** (patch + Docker). BasinRAG é retriever; localização de ficheiros (Lite 300) é sonda de domínio, não leaderboard.

---

## 5. Custo e operação

| Dimensão | Dense API | GraphRAG-class | BasinRAG |
| :--- | :--- | :--- | :--- |
| Index LLM | Não / API emb. | Alto | **$0** |
| Privacidade | Depende | Texto → API LLM | Local possível |
| Determinismo do grafo | N/A | Baixo | Alto (φ local) |

---

## 6. Resumo

BasinRAG combina **retrieval híbrido local** com **estrutura de documento para briefing**. Detalhes em [`BENCHMARKS.md`](../BENCHMARKS.md).
