from types import SimpleNamespace

import pytest

from basinrag.factory import BasinRAG, BasinRAGConfig
from basinrag.indexer.bm25 import current_stemmer_version


def _rag_with_index(*, configured_stemming: bool, stored_stemming: bool, stored_version: str):
    rag = BasinRAG.__new__(BasinRAG)
    rag.config = BasinRAGConfig(
        storage_dir=".basinrag-v3-test",
        encoder_model="unconfigured",
        use_rerank=False,
        bm25_stemming=configured_stemming,
    )
    metadata = {
        "format_version": 3,
        "encoder_model": "unconfigured",
        "encoder_revision": "unresolved",
        "tokenizer": "unconfigured",
        "tokenizer_revision": "unresolved",
        "chunk_policy_version": 1,
        "chunking_mode": "characters",
        "chunk_size": 512,
        "chunk_overlap": 128,
        "bm25_stemming": stored_stemming,
        "bm25_stemmer_version": stored_version,
    }
    rag.engine = SimpleNamespace(
        index_schema_version=3,
        index_metadata=metadata,
        encoder_model="unconfigured",
    )
    rag._ingestor = SimpleNamespace(splitter=SimpleNamespace(tokenizer=None))
    return rag


def test_bm25_default_policy_does_not_require_optional_stemmer():
    assert current_stemmer_version(False) == "disabled"
    rag = _rag_with_index(configured_stemming=False, stored_stemming=False, stored_version="disabled")
    rag._validate_index_compatibility()


def test_snapshot_stemming_policy_change_requires_reindex():
    rag = _rag_with_index(configured_stemming=False, stored_stemming=True, stored_version="snowballstemmer-3.1.1")
    with pytest.raises(ValueError, match="política de stemming"):
        rag._validate_index_compatibility()


def test_snapshot_stemmer_revision_change_requires_reindex():
    rag = _rag_with_index(configured_stemming=True, stored_stemming=True, stored_version="snowballstemmer-0.0.0")
    with pytest.raises(ValueError, match="versão do stemmer"):
        rag._validate_index_compatibility()


def test_enabled_stemmer_version_is_recordable():
    assert current_stemmer_version(True).startswith("snowballstemmer-")
