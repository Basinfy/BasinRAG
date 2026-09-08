import os
import shutil
import pytest
import numpy as np
import basinrag
from basinrag import __version__
from basinrag.core.persistence import safe_replace_dir, BasinPersistence
from basinrag.core.kv_store import DiskKVStore
from basinrag.core.llm import UniversalLLM


def test_version_alignment():
    """Valida alinhamento da versão 1.0.2 no __init__ e consistência do pacote."""
    assert __version__ == "1.0.2"
    assert basinrag.__version__ == "1.0.2"


def test_cli_version_flag():
    """Valida que o CLI possui flag --version funcional."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "basinrag.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "1.0.2" in result.stdout or "1.0.2" in result.stderr


def test_safe_replace_dir_preserves_subdirectories(tmp_path):
    """Garante que subdiretórios não são perdidos mesmo em fallback do safe_replace_dir."""
    src = tmp_path / "src_tree"
    dst = tmp_path / "dst_tree"
    
    src.mkdir()
    (src / "file1.txt").write_text("hello", encoding="utf-8")
    sub = src / "sub_folder"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content", encoding="utf-8")
    
    # Executa safe_replace_dir
    safe_replace_dir(str(src), str(dst))
    
    assert dst.exists()
    assert (dst / "file1.txt").read_text(encoding="utf-8") == "hello"
    assert (dst / "sub_folder" / "nested.txt").read_text(encoding="utf-8") == "nested content"
    assert not src.exists()


def test_kv_store_paginated_iterators(tmp_path):
    """Testa os iteradores paginados iter_keys, iter_items, iter_values do DiskKVStore."""
    db_file = str(tmp_path / "test_iter.db")
    store = DiskKVStore(db_file, "iter_test")
    
    # Inserir 25 itens
    for i in range(25):
        store.set(f"k_{i:02d}", {"val": i})
        
    assert len(store) == 25
    
    # Testar iter_keys com batch_size pequeno
    keys = list(store.iter_keys(batch_size=5))
    assert len(keys) == 25
    assert keys[0] == "k_00"
    
    # Testar iter_items
    items = list(store.iter_items(batch_size=5))
    assert len(items) == 25
    assert items[0] == ("k_00", {"val": 0})
    
    # Testar iter_values
    vals = list(store.iter_values(batch_size=5))
    assert len(vals) == 25
    assert vals[0] == {"val": 0}
    
    # Testar __iter__
    iter_keys = list(iter(store))
    assert len(iter_keys) == 25
    
    # Testar compatibilidade de keys() e items()
    assert len(store.keys()) == 25
    assert len(store.items()) == 25
    assert len(store.values()) == 25
    
    store.close()


def test_cors_middleware_wildcard_credentials():
    """Garante que a API não quebra no startup ao usar origins wildcard."""
    from fastapi.testclient import TestClient
    from basinrag.api.server import app
    
    # Instanciar o TestClient valida a árvore de middlewares (não deve levantar AssertionError)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider_raises():
    """Garante que provedor inválido ou falha de inicialização gera exception e não yield de string."""
    client = UniversalLLM(provider="invalid_provider")
    with pytest.raises(ValueError, match="não suportado"):
        await client.generate("sys", "user")
