# BasinRAG: o que o sistema realmente faz

O BasinRAG é um **RAG para documentos** (PDF, Markdown, TXT) que agrupa chunks em **bacias** — partições de índice, não órbitas de um teorema numérico.

Ele **reutiliza vocabulário do Basinfy** (bacia, atrator, árvore rho, hops, L0–L3) e a **cara local/global do GraphRAG**. Não aplica o mapa \(f_{k,b}\) ao corpus. Não é Leiden. Não é um port fiel da matemática do Basinfy.

O Basinfy (BasinMind) deixa isso explícito no próprio produto: teoremas de soma de dígitos valem para **inteiros** no kernel WASM; a qualidade de query vem de **BM25 + embeddings + condensação L0–L3**. O BasinRAG segue a mesma honestidade no caminho de retrieval.

---

## Duas camadas (não misturar)

1. **Índice (produto)** — o que `query` / `brief` / `chat` usam:
   ingestão → embeddings → grafo de vizinhança + sucessor sequencial φ → atratores (início de seção) → hops → BM25 CSR + FAISS/HNSW + RRF ponderado → briefing com budget.
2. **Não misturar com o kernel Basinfy** — não há DR9, QMC nem PageRank no pacote. Ligar scramble/checksum modular como LSH piora ANN.

---

## Pipeline de indexação

```text
arquivo (txt/md/pdf, walk recursivo)
  → RecursiveCharacterTextSplitter (1000 / 100)
  → id estável SHA-256(source, index, text)[:16]
  → MiniLM multilingual, L2-normalizado
  → L1 keywords + L2 primeira sentença (extractivo, sem LLM)
  → grafo não-dirigido:
       sequential (ordem no documento)
       virtual-edge (kNN cosseno > 0.85)   ← fora de φ
  → sucessor φ: chunk aponta para o anterior na mesma seção (~20 chunks)
  → atrator = início da seção (sink)
  → rho-tree = inverso de φ (pai → filho)
  → BM25 CSR persistido (k1=1.2, b=0.75)
  → .basinrag/ {graph.json, embeddings.npz, bm25.json, meta.json, basins/}
```

Re-ingerir **substitui** o grafo em memória (`build_graph` faz `reset()`). Os mesmos arquivos produzem os mesmos IDs.

Resumos L3 **LLM** (Draft → Critique → Refine) continuam opcionais, em background no `chat`/`serve`. No índice já existe um L3 extractivo (rótulo de domínio + keywords) usado no ranking global.

---

## Pipeline de query

```text
pergunta
  → router (hybrid | global); default hybrid; nunca “local” silencioso
  → projector lexical (tokens, sem stopwords curtas)
  → BM25 CSR + FAISS/HNSW corpus-wide (hybrid) ou 3 entry points (local)
  → local: bacia do seed + 1 hop sequencial + virtual-edge dos hits FAISS
  → recusa se melhor similaridade < 0.15
  → RRF ponderado α=0.55 por node_id + prior exp(-hops·0.35) com piso 0.7
  → BriefingPacket: hubs L0, satélites L3 (label ou summary, cap 400)
  → CrossEncoder multilingual (não mistura headers L3)
  → LLM (chat) só se confidence ≥ 0.15
```

Busca **global** ranqueia bacias pelo embedding do **L3 LLM**; se ainda não existir, usa o vetor do atrator. Scores todos ~0 → lista vazia.

---

## Módulos

```text
basinrag/
├── core/
│   ├── topology.py          # grafo + φ + bacias + rho-tree
│   ├── functional_graph.py  # sucessor, Floyd/sinks, hops, arestas rho
│   ├── vector_index.py      # FlatIP pequeno N; HNSW se N > 256
│   ├── persistence.py       # JSON + NPZ + BM25 + buildId
│   ├── ids.py               # IDs estáveis, tokenize
│   └── llm.py
├── indexer/
│   ├── ingestor.py
│   ├── condensation.py      # L0–L3 extractivo
│   ├── bm25.py
│   └── summarizer.py        # L3 LLM opcional (mesmo model_name do chat)
├── retriever/
│   ├── router.py            # PT + EN
│   ├── projector.py
│   ├── fusion.py            # RRF α=0.55 + hops
│   ├── local_search.py      # intra-bacia
│   ├── hybrid_search.py
│   ├── global_search.py     # score L3
│   ├── briefing.py
│   └── reranker.py          # mMARCO multilingual
├── api/
│   └── server.py
├── cli.py
└── factory.py
```

---

## Relação com digit-sum-power-maps

O paper [Iterated Digit-Sum Power Maps](https://doi.org/10.5281/zenodo.22181953) define \(f_{k,b}(n)=S_b(n^k)\) em \(\mathbb{Z}^+\), organizado por \(\varphi_{k,b-1}(x)=x^k \bmod (b-1)\). Teoremas (limite inferior de atratores, densidade **agregada** por assinatura de resíduo, formas fechadas de \(\mathrm{Cyc}\)) valem para inteiros. Os autores avisam: teoremas de inteiros **não governam retrieval**.

O BasinRAG **não instancia** \(f_{k,b}\). Os nomes abaixo são homônimos:

| Termo | DSPM | BasinRAG vivo |
|---|---|---|
| φ | \(x^k \bmod (b-1)\) em resíduos | predecessor sequencial no documento (seção de 20 chunks) |
| Atrator | órbita periódica de \(f_{k,b}\) em \([1,M]\) | primeiro chunk da seção |
| Bacia | pontos cuja órbita entra em \(A\) | os ≤20 chunks que drenam para esse início |
| Hops | iterações até o ciclo | distância reversa no sucessor sequencial |
| Ranking | não há retrieval | BM25 + FAISS + RRF \(\alpha=0.55\) + prior de hops |

kNN cosseno > 0.85 é `virtual-edge` **fora** de φ. Não use soma de dígitos, resíduo ou raiz digital como LSH em embeddings: quebra a vizinhança ANN. O único empréstimo legítimo é o *padrão* de grafo funcional (1 sucessor, sink, árvore inversa), preenchido com **ordem do texto**, não com \(S_b\).

---

## O que não copiamos do Basinfy

Kernel WASM \(f_{k,b}\), MCP Drive, personas/skills, `gate_edit`, AST/símbolos de repositório, Collatz/Kaprekar, `attractorCoverage` como métrica de qualidade RAG, residue modular como similaridade.

O que **mantemos** e o Basinfy não tem: CrossEncoder e relatórios L3 com LLM.
