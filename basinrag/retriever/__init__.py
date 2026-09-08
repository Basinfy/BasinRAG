"""BasinRAG retrievers."""

__all__ = ["BasinRAGRetriever", "TopologicalLocalSearch", "TopologicalGlobalSearch"]


def __getattr__(name):
    if name == "BasinRAGRetriever":
        from .base import BasinRAGRetriever
        return BasinRAGRetriever
    if name == "TopologicalLocalSearch":
        from .local_search import TopologicalLocalSearch
        return TopologicalLocalSearch
    if name == "TopologicalGlobalSearch":
        from .global_search import TopologicalGlobalSearch
        return TopologicalGlobalSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
