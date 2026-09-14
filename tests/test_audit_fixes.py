from __future__ import annotations

import subprocess
import sys

import pytest

import basinrag
from basinrag import __version__
from basinrag.core.kv_store import DiskKVStore
from basinrag.core.llm import UniversalLLM


def test_version_alignment():
    assert __version__ == "1.1.0"
    assert basinrag.__version__ == "1.1.0"


def test_default_huggingface_models_are_pinned():
    from basinrag import BasinRAGConfig

    config = BasinRAGConfig()
    assert config.encoder_revision == "dd9f42942e0729b6c53632f3c23b0e801f236569"
    assert config.reranker_revision == "c4b98d26050227d7b53a54437302be5aa412b70e"
    assert config.use_rerank is False


def test_custom_models_require_immutable_commit_revisions():
    from basinrag import BasinRAGConfig

    with pytest.raises(ValueError, match="encoder_revision"):
        BasinRAGConfig(encoder_model="organization/custom-encoder")
    with pytest.raises(ValueError, match="40 caracteres"):
        BasinRAGConfig(
            encoder_model="organization/custom-encoder",
            encoder_revision="main",
        )

    config = BasinRAGConfig(
        encoder_model="organization/custom-encoder",
        encoder_revision="0123456789abcdef0123456789abcdef01234567",
        use_rerank=False,
    )
    assert config.encoder_revision == "0123456789abcdef0123456789abcdef01234567"


def test_cli_version_flag():
    result = subprocess.run(
        [sys.executable, "-m", "basinrag.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "1.1.0" in result.stdout


def test_kv_store_paginated_iterators(tmp_path):
    store = DiskKVStore(str(tmp_path / "items.sqlite3"), "test_items")
    for i in range(25):
        store.set(f"k_{i:02d}", {"value": i})
    assert len(list(store.iter_keys(batch_size=5))) == 25
    assert len(list(store.iter_items(batch_size=5))) == 25
    assert len(list(store.iter_values(batch_size=5))) == 25
    assert list(iter(store))[0] == "k_00"
    store.close()


def test_read_only_kv_store_does_not_create_or_mutate(tmp_path):
    missing = tmp_path / "absent.sqlite3"
    with pytest.raises(FileNotFoundError):
        DiskKVStore(str(missing), "items", read_only=True)
    assert not missing.exists()

    writable = DiskKVStore(str(tmp_path / "present.sqlite3"), "items")
    writable.set("one", 1)
    writable.close()
    readonly = DiskKVStore(str(tmp_path / "present.sqlite3"), "items", read_only=True)
    assert readonly.get("one") == 1
    with pytest.raises(RuntimeError, match="read-only"):
        readonly.set("two", 2)
    assert len(readonly) == 1
    readonly.close()


def test_api_origin_parser_rejects_wildcard_and_paths():
    from basinrag.api.server import _validated_origins

    with pytest.raises(ValueError, match="Wildcard"):
        _validated_origins("*")
    with pytest.raises(ValueError, match="Origem inválida"):
        _validated_origins("https://example.test/path")
    assert _validated_origins("https://example.test") == {"https://example.test"}


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider_raises():
    client = UniversalLLM(provider="invalid_provider")
    with pytest.raises(ValueError, match="não suportado"):
        await client.generate("sys", "user")
