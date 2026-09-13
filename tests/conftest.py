from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_implicit_engine_storage(monkeypatch, tmp_path, request):
    """Keep low-level engine tests away from any workspace index roots."""
    # The API contract suite uses a mocked RAG and should not import numerical
    # engine dependencies merely to install this isolation shim.
    if request.module.__name__.endswith("test_api"):
        return

    from basinrag.core.topology import BasinTopologyEngine

    original_init = BasinTopologyEngine.__init__

    def isolated_init(self, *args, **kwargs):
        if kwargs.get("storage_dir") is None:
            kwargs["storage_dir"] = str(tmp_path / "implicit-engine-v3")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(BasinTopologyEngine, "__init__", isolated_init)
