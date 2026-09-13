"""Resolve model branches/tags to immutable Hugging Face commit ids."""
from __future__ import annotations

import re
from typing import Optional

_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
DEFAULT_MODEL_REVISIONS = {
    "BAAI/bge-base-en-v1.5": "dd9f42942e0729b6c53632f3c23b0e801f236569",
    "BAAI/bge-reranker-v2-m3": "c4b98d26050227d7b53a54437302be5aa412b70e",
}


def default_huggingface_revision(model_name: str) -> Optional[str]:
    return DEFAULT_MODEL_REVISIONS.get(model_name)


def resolve_huggingface_revision(model_name: str, requested: Optional[str] = None) -> str:
    if requested and _COMMIT.fullmatch(requested):
        return requested.lower()
    if requested:
        raise ValueError(
            f"A revisão de {model_name!r} deve ser o SHA completo de um commit de 40 caracteres; "
            "branches e tags podem mudar."
        )
    pinned = default_huggingface_revision(model_name)
    if pinned:
        return pinned
    raise ValueError(
        f"O modelo customizado {model_name!r} exige revisão explícita: informe o SHA "
        "completo do commit no Hugging Face Hub."
    )
