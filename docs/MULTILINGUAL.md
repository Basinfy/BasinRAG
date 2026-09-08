# 🌐 Multilingual & Cross-Lingual Architecture in BasinRAG

BasinRAG was architected from inception to support enterprise multilingual environments where documents and queries originate in different languages.

---

## 1. Architectural Overview

```
                               ┌────────────────────────┐
                               │       User Query       │
                               │  (e.g., in Portuguese) │
                               └───────────┬────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                        ▼                                     ▼
             ┌─────────────────────┐               ┌─────────────────────┐
             │    Multilingual     │               │   Bilingual BM25    │
             │   Dense Embedding   │               │    Lexical Match    │
             │   (Shared Metric)   │               │ (Language-specific) │
             └──────────┬──────────┘               └──────────┬──────────┘
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │  Canonical RRF Fusion │
                               │  + Topological Priors │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │  mMARCO Cross-Encoder │
                               │ Multilingual Re-rank  │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Top Cross-Lingual     │
                               │ Document Passages     │
                               └───────────────────────┘
```

---

## 2. Core Multilingual Components

### 2.1 Cross-Lingual Dense Retrieval (Shared Semantic Manifold)
BasinRAG defaults to `paraphrase-multilingual-MiniLM-L12-v2` as its primary dense encoder:
- **50+ Languages Supported**: Pre-trained on parallel multi-lingual text corpora to ensure semantic vectors for equivalent concepts align in the same metric neighborhood.
- **Cross-Lingual Invariance**: A query in Portuguese (e.g., *"Como funciona o decaimento de bacias de atração?"*) maps close to English text (*"The dynamical decay of attraction basins operates by..."*).

### 2.2 Dynamic Bilingual Lexical Indexing (BM25 CSR)
Unlike naive RAG systems that apply a generic English tokenizer across all documents, BasinRAG features automatic per-document language detection:
- **Automatic Language Detection**: At ingest time, `basinrag.indexer.bm25` analyzes character frequencies, accents, and stopwords to classify documents into Portuguese (`pt`) or English (`en`).
- **Dedicated Morphological Stemmers**:
  - Portuguese: `nltk.stem.RSLPStemmer` (standard Brazilian Portuguese morphological reducer).
  - English: `nltk.stem.SnowballStemmer('english')`.
- **Query Alignment**: When an incoming query is received, the lexical projector dynamically identifies query language markers to apply the matching stemmer before BM25 term scoring.

### 2.3 Multilingual Cross-Encoder Re-ranking (mMARCO)
For the final context re-ranking stage, BasinRAG employs `unicamp-dl/mmarco-mMiniLMv2-L12-H384-v1`:
- Trained specifically on the multilingual mMARCO dataset.
- Evaluates full query-passage cross-attention pairs across language boundaries.
- Prevents false-positive token overlap by verifying deep semantic relevance.

### 2.4 Bilingual Intent Router
The `IntelligentQueryRouter` inspects question semantics:
- Fast-path regex detects factual question prefixes in Portuguese (*"o que é"*, *"qual é"*, *"como funciona"*) and English (*"what is"*, *"how does"*, *"why is"*).
- Directs factual questions to `hybrid` topological search and exploratory high-level questions to `global` basin-summary search.

---

## 3. Practical Usage Examples

### 3.1 Cross-Lingual Querying (Python SDK)

```python
from basinrag import BasinRAG, BasinRAGConfig

config = BasinRAGConfig(
    storage_dir=".basinrag_multilingual",
    encoder_model="paraphrase-multilingual-MiniLM-L12-v2"
)
rag = BasinRAG(config)

# Ingest an English scientific paper
rag.ingest("./papers_en/")

# Query in Portuguese
results = rag.query("Quais foram as métricas de acurácia obtidas no teste?", top_k=3)

for r in results:
    print(f"Passage (EN): {r.text}")
    print(f"Score: {r.score:.3f}\n")
```

### 3.2 CLI Usage Across Languages

```bash
# Ingest mixed PT/EN folder
basinrag ingest ./mixed_corpus/

# Query in Portuguese
basinrag query "Quais os métodos de otimização topológica?" --top-k 5

# Query in English
basinrag query "What are the main topological optimization methods?" --top-k 5
```

---

## 4. Extending to Additional Languages (e.g., Spanish, French, German)

To add support for Spanish (`es`), French (`fr`), or other languages:

1. **Add the Stemmer in `basinrag/indexer/bm25.py`**:
```python
if lang == "pt":
    self._stemmer = RSLPStemmer()
elif lang == "es":
    from nltk.stem.snowball import SnowballStemmer
    self._stemmer = SnowballStemmer("spanish")
elif lang == "fr":
    from nltk.stem.snowball import SnowballStemmer
    self._stemmer = SnowballStemmer("french")
else:
    self._stemmer = SnowballStemmer("english")
```

2. **Dense & Re-ranking**: The default dense model (`paraphrase-multilingual-MiniLM-L12-v2`) and cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) already support Spanish, French, German, Italian, and Chinese out-of-the-box.
