"""FAISS index factory: exact IP for small corpora, HNSW when N grows."""
from __future__ import annotations

import numpy as np

# Keep FlatIP longer to protect recall on mid-size BEIR corpora (e.g. SciFact ~5k).
HNSW_THRESHOLD = 20000
HNSW_EF_SEARCH = 256
HNSW_EF_CONSTRUCTION = 80
HNSW_M = 32


def build_ip_index(embeddings: np.ndarray):
    import faiss

    if embeddings is None or len(embeddings) == 0:
        return None
    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    n, d = vecs.shape
    if n == 0 or d == 0:
        return None
    faiss.normalize_L2(vecs)
    if n > HNSW_THRESHOLD:
        index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.add(vecs)
        return index
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index
