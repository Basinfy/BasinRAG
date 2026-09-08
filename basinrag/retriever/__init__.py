"""BasinRAG retrievers."""

__all__ = ["BasinRAGRetriever", "TopologicalLocalSearch", "TopologicalGlobalSearch", "MetaBasin", "build_meta_basins"]


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
    if name == "MetaBasin":
        from .meta_basins import MetaBasin
        return MetaBasin
    if name == "build_meta_basins":
        from .meta_basins import build_meta_basins
        return build_meta_basins
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

