import sys

# Backward compatibility shim for environments with PyTorch < 2.5 and newer transformers
try:
    import torch
    if hasattr(torch, "distributed"):
        try:
            from torch.distributed.tensor import DTensor  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            try:
                import torch.distributed._tensor as _dt
                import torch.distributed._tensor._utils as _dt_u
                import torch.distributed._tensor.placement_types as _dt_pt
                sys.modules["torch.distributed.tensor"] = _dt
                sys.modules["torch.distributed.tensor._utils"] = _dt_u
                sys.modules["torch.distributed.tensor.placement_types"] = _dt_pt
            except Exception:
                pass
except Exception:
    pass

__version__ = "1.0.0-rc.1"
__all__ = ["BasinRAG"]


def __getattr__(name):
    if name == "BasinRAG":
        from .factory import BasinRAG
        return BasinRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
