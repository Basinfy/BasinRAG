"""CSR-style BM25 (Robertson/Sparck Jones) with optional disk persistence.

k1=1.2, b=0.75 — same knobs as Basinfy's bm25-indexer.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Sequence, Tuple

from ..core.ids import tokenize

K1 = 1.2
B = 0.75

_STOP = {
    "a", "o", "os", "as", "um", "uma", "de", "da", "do", "das", "dos", "e", "ou",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "que", "se", "nao",
    "não", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "be", "as", "at", "by", "an", "this", "that", "it", "from", "with",
}

_stemmer_pt = None
_stemmer_en = None


def _get_stemmers():
    global _stemmer_pt, _stemmer_en
    if _stemmer_pt is None:
        try:
            import nltk
            from nltk.stem import RSLPStemmer, SnowballStemmer
            nltk.download("rslp", quiet=True)
            _stemmer_pt = RSLPStemmer()
            _stemmer_en = SnowballStemmer("english")
        except ImportError:
            _stemmer_pt = False  # Mark as unavailable
    return _stemmer_pt, _stemmer_en


def stem_token(tok: str) -> str:
    pt, en = _get_stemmers()
    if not pt:
        return tok
    import unicodedata
    has_accent = any(
        unicodedata.category(c) == "Mn"
        for c in unicodedata.normalize("NFD", tok)
    )
    if has_accent:
        return pt.stem(tok)
    stemmed_en = en.stem(tok)
    stemmed_pt = pt.stem(tok)
    return min(stemmed_en, stemmed_pt, key=len)


class BM25Index:
    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.n: int = 0
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.build_id: str = ""

    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_len = []
        self.df = {}
        self.postings = {}
        self.n = len(self.doc_ids)
        if self.n == 0:
            self.avgdl = 0.0
            return

        for i, text in enumerate(texts):
            tf: Dict[str, int] = {}
            for tok in tokenize(text):
                if tok in _STOP:
                    continue
                tok = stem_token(tok)
                tf[tok] = tf.get(tok, 0) + 1
            length = sum(tf.values()) or 1
            self.doc_len.append(length)
            for term, count in tf.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.postings.setdefault(term, []).append((i, count))

        self.avgdl = sum(self.doc_len) / self.n

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.n == 0:
            return []
        acc: Dict[int, float] = {}
        for term in tokenize(query):
            if term in _STOP:
                continue
            term = stem_token(term)
            if term not in self.postings:
                continue
            idf = self._idf(term)
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = tf + K1 * (1.0 - B + B * dl / (self.avgdl or 1.0))
                acc[doc_idx] = acc.get(doc_idx, 0.0) + idf * (tf * (K1 + 1.0) / denom)
        ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked]

    def save(self, path: str, build_id: str = "") -> None:
        payload = {
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "n": self.n,
            "df": self.df,
            "postings": {t: pairs for t, pairs in self.postings.items()},
            "buildId": build_id or self.build_id,
        }
        self.build_id = payload["buildId"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.doc_ids = payload["doc_ids"]
        self.doc_len = payload["doc_len"]
        self.avgdl = float(payload["avgdl"])
        self.n = int(payload["n"])
        self.df = {k: int(v) for k, v in payload["df"].items()}
        self.postings = {
            t: [(int(i), int(tf)) for i, tf in pairs]
            for t, pairs in payload["postings"].items()
        }
        self.build_id = payload.get("buildId", "") or ""
        return True
