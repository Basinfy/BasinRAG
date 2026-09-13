from ._version import __version__

__all__ = ["BasinRAG", "BasinRAGConfig", "__version__"]


def __getattr__(name):
    if name in {"BasinRAG", "BasinRAGConfig"}:
        from . import factory
        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
