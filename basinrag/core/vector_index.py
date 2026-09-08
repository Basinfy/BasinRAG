"""FAISS index factory: exact IP for small corpora, HNSW (L2 on unit vectors) when N grows."""
from __future__ import annotations

import numpy as np

HNSW_THRESHOLD = 4096


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
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 80
        index.add(vecs)
        return index
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index
