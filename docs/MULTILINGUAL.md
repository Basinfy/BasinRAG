# 🌐 Multilingual behavior in BasinRAG

> Historical note: the previous version of this page made unsupported claims of enterprise-grade ten-language support, calibrated centroids, and guaranteed language-preserving L3. Those are not guarantees of BasinRAG 1.1.0. Encoder and reranker behavior depends on the configured model and must be evaluated for the target language and corpus.

Current contracts: BM25 uses script-aware language signals and language-specific stopword lists only when detection is sufficiently certain; uncertain language means no stopword removal or stemming. Stemming is disabled by default and is enabled only through `bm25_stemming` / `BASINRAG_BM25_STEMMING=true`, recorded in the snapshot manifest. The optional `snowballstemmer` implementation requires no downloaded corpora. L3 is disabled by default, and remote summarization requires explicit egress consent.

## Current behavior

BasinRAG does not claim universal or calibrated cross-lingual quality. Retrieval quality depends on the configured encoder, tokenizer, reranker, corpus, and language pair; evaluate those combinations on representative data before relying on them.

- Dense retrieval uses the configured encoder. The default BAAI/bge-base-en-v1.5 is English-oriented; multilingual use requires choosing and evaluating an appropriate model.
- BM25 uses Unicode tokenization and a small set of language signals and stopwords. If language identification is uncertain, no stopwords are removed.
- Stemming is disabled by default. It can be enabled with bm25_stemming=True or BASINRAG_BM25_STEMMING=true; the setting and stemmer package version are recorded in the snapshot manifest. The optional `snowballstemmer` implementation requires no downloaded corpora.
- hybrid_rrf is the default ranking mode. Experimental topological virtual links are opt-in and are not part of the default ranking path.
- Basin summaries are for briefing and navigation, not evidence of improved retrieval. Background L3 is disabled by default; remote summarization requires both the L3 and remote-egress opt-ins.
- Model revisions must be immutable commit SHAs. The default encoder and reranker are pinned in code; custom models require explicit revisions.

For a custom encoder, configure both its model identifier and a full immutable commit SHA in encoder_revision. Changing the encoder, tokenizer, chunking policy, or stemming configuration requires a rebuild into a new, empty storage root. See the [API reference](API_REFERENCE.md), [ranking guide](RANKING.md), and [deployment guide](DEPLOYMENT.md).

Historical v1.0.3 marketing copy (MiniLM / mMARCO as default) is in [archive/MULTILINGUAL_v1.0.3.md](archive/MULTILINGUAL_v1.0.3.md). It is not this contract.
