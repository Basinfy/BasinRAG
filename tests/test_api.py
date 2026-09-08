import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import basinrag.api.server

@pytest.fixture
def mock_rag():
    with patch("basinrag.api.server.BasinRAG.create") as mock_create:
        mock_instance = MagicMock()
        mock_instance.config.model_name = "test-model"
        
        # Mocking the query method
        mock_doc = MagicMock()
        mock_doc.page_content = "resultado mockado"
        mock_instance.query.return_value = [mock_doc]
        from unittest.mock import AsyncMock
        mock_instance.start_background_summarizer = AsyncMock()
        
        mock_create.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def client(mock_rag):
    # Important to import app after patching BasinRAG.create to ensure lifespan uses mock
    from basinrag.api.server import app
    with TestClient(app) as client:
        yield client

def test_root_returns_online(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"

def test_query_returns_results(client):
    r = client.post("/query", json={"query": "tema principal", "top_k": 2})
    assert r.status_code == 200
    assert "results" in r.json()
    assert r.json()["results"][0] == "resultado mockado"

def test_query_rejects_empty_or_short_query(client):
    r = client.post("/query", json={"query": "a", "top_k": 2})
    assert r.status_code == 422

def test_query_rejects_excessive_top_k(client):
    r = client.post("/query", json={"query": "pergunta valida", "top_k": 500})
    assert r.status_code == 422

def test_api_key_enforcement(client):
    with patch("basinrag.api.server.API_KEY", "secret123"):
        # Without key -> 401
        r = client.post("/query", json={"query": "pergunta valida", "top_k": 5})
        assert r.status_code == 401

        # With correct key -> 200
        r = client.post("/query", json={"query": "pergunta valida", "top_k": 5}, headers={"X-API-KEY": "secret123"})
        assert r.status_code == 200
