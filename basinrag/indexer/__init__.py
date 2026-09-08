"""BasinRAG indexer package."""

__all__ = ["BasinIngestor", "BasinSummarizer", "BM25Index"]


def __getattr__(name):
    if name == "BasinIngestor":
        from .ingestor import BasinIngestor
        return BasinIngestor
    if name == "BasinSummarizer":
        from .summarizer import BasinSummarizer
        return BasinSummarizer
    if name == "BM25Index":
        from .bm25 import BM25Index
        return BM25Index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
