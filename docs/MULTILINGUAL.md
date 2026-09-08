# 🌐 Multilingual & Cross-Lingual Architecture in BasinRAG

BasinRAG provides enterprise-grade out-of-the-box support for the **top 10 global software development and AI languages**:

| Code | Language | Script / Family | Lexical Stemmer | Dense & Re-ranker |
| :---: | :--- | :--- | :--- | :--- |
| **`en`** | English | Latin | `SnowballStemmer("english")` | `MiniLM-L12-v2` + `mMARCO` |
| **`pt`** | Portuguese | Latin (Accented) | `RSLPStemmer` / `Snowball("portuguese")` | `MiniLM-L12-v2` + `mMARCO` |
| **`es`** | Spanish | Latin (Accented) | `SnowballStemmer("spanish")` | `MiniLM-L12-v2` + `mMARCO` |
| **`zh`** | Simplified Chinese | Hanzi (CJK) | Zero-bloat CJK unigrams / pass-through | `MiniLM-L12-v2` + `mMARCO` |
| **`ja`** | Japanese | Kanji + Kana (CJK) | Zero-bloat CJK unigrams / pass-through | `MiniLM-L12-v2` + `mMARCO` |
| **`de`** | German | Latin (Umlauts) | `SnowballStemmer("german")` | `MiniLM-L12-v2` + `mMARCO` |
| **`fr`** | French | Latin (Accented) | `SnowballStemmer("french")` | `MiniLM-L12-v2` + `mMARCO` |
| **`ru`** | Russian | Cyrillic | `SnowballStemmer("russian")` | `MiniLM-L12-v2` + `mMARCO` |
| **`ko`** | Korean | Hangul (CJK) | Zero-bloat CJK unigrams / pass-through | `MiniLM-L12-v2` + `mMARCO` |
| **`it`** | Italian | Latin | `SnowballStemmer("italian")` | `MiniLM-L12-v2` + `mMARCO` |

---

## 1. Architectural Overview

```
                               ┌────────────────────────┐
                               │       User Query       │
                               │  (Any of 10 languages) │
                               └───────────┬────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                        ▼                                     ▼
             ┌─────────────────────┐               ┌─────────────────────┐
             │    Multilingual     │               │  10-Language BM25   │
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

### 2.1 Unicode & CJK Tokenization (`basinrag.core.ids`)
BasinRAG features a unified Unicode tokenizer that parses Western, Cyrillic, and CJK scripts without external C++ dependencies:
- **Latin & Cyrillic**: `[A-Za-zÀ-ÿ0-9_\u0400-\u04FF]{2,}` matches full words and preserves accents.
- **CJK Characters (Chinese, Japanese, Korean)**: `[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]` matches individual Hanzi ideographs, Hiragana/Katakana syllables, and Hangul blocks.
- **Zero-Bloat Advantage**: Eliminates runtime compilation issues associated with heavy native tokenizers (`jieba`, `mecab`), while semantic phrase matching is resolved via dense embeddings.

### 2.2 Dynamic 10-Language Lexical Indexing (BM25 CSR)
At indexing time, `basinrag.indexer.bm25` detects the language of each document and query:
- **Instant Script Detection**: Cyrillic (`ru`), Hangul (`ko`), Kana (`ja`), and Hanzi (`zh`).
- **Diacritics & Morphological Markers**: Spanish (`ñ`), German (`ä, ö, ü, ß`), French (`œ, æ, ù`), Portuguese (`ç, ã, õ`).
- **NLTK Snowball Integration**: Automatically invokes optimized stemmers for English, Portuguese, Spanish, French, German, Italian, and Russian.
- **CJK Pass-Through**: Preserves exact character sequences without destructive stemming.

### 2.3 Shared Semantic Manifold (Dense & Re-ranking)
- **Dense Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` projects queries and documents from all 10 languages into a common 384-dimensional metric space.
- **Cross-Encoder Re-ranking**: `unicamp-dl/mmarco-mMiniLMv2-L12-H384-v1` provides cross-lingual reranking using full cross-attention over query-passage pairs.

### 2.4 Multilingual Intent Router (`basinrag.retriever.router`)
The `IntelligentQueryRouter` inspects question semantics across all 10 languages:
- **Overview & Global Queries**: Recognizes high-level synthesis requests (e.g., *"tema principal"*, *"main theme"*, *"resumen"*, *"vue d'ensemble"*, *"überblick"*, *"главная тема"*, *"核心思想"*, *"主要テーマ"*, *"주요 주제"*) and directs them to global basin summaries.
- **Specific Fact Queries**: Routes narrow factual inquiries directly to the hybrid topological retrieval pipeline.
- **Cross-Lingual Centroids**: Pre-calibrated embedding centroids for global vs hybrid routing.

### 2.5 Language-Aware Agentic Summarizer (`basinrag.indexer.summarizer`)
The L3 community summarizer enforces language preservation during the Draft $\to$ Critique $\to$ Refine cycle, guaranteeing that summaries for Japanese, German, Russian, or Portuguese documentation remain in their native language.

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

# Ingest mixed documentation (e.g. Japanese, German, Portuguese)
rag.ingest("./multilingual_docs/")

# Query in English against Japanese documents
results = rag.query("How does topological basin diffusion work?", top_k=3)

for r in results:
    print(f"Passage: {r.text}")
    print(f"Score: {r.score:.3f}\n")
```

### 3.2 CLI Usage Across Languages

```bash
# Ingest folder with mixed languages
basinrag ingest ./docs/

# Query in Spanish
basinrag query "¿Cuáles son los métodos de optimización topológica?" --top-k 5

# Query in German
basinrag query "Was ist das Hauptthema dieses Systems?" --top-k 5

# Query in Chinese
basinrag query "总结系统的核心架构" --top-k 5
```
