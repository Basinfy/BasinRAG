from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


class _StubBasinRAGConfig:
    @staticmethod
    def load_environment():
        return None

    @classmethod
    def from_env(cls):
        return cls()


class _StubBasinRAG:
    @classmethod
    def create(cls):
        raise AssertionError("The API test fixture must provide the mocked RAG")

    def __init__(self, config):
        self.config = config
        self._loaded = False
        self.engine = SimpleNamespace(close_stores=lambda: None)


_factory_stub = ModuleType("basinrag.factory")
_factory_stub.BasinRAG = _StubBasinRAG
_factory_stub.BasinRAGConfig = _StubBasinRAGConfig
_previous_factory = sys.modules.get("basinrag.factory")
sys.modules["basinrag.factory"] = _factory_stub
try:
    server = importlib.import_module("basinrag.api.server")
finally:
    if _previous_factory is None:
        sys.modules.pop("basinrag.factory", None)
    else:
        sys.modules["basinrag.factory"] = _previous_factory


@pytest.fixture
def mock_rag():
    with patch.object(server.BasinRAG, "create") as create:
        class GraphStub:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node_id, **data):
                self.nodes[node_id] = data

            def __contains__(self, node_id):
                return node_id in self.nodes

        rag = SimpleNamespace()
        rag._loaded = True
        rag.config = SimpleNamespace(
            model_name="test-model", enable_background_l3=False,
            provider="local", allow_remote_l3_egress=False,
        )
        rag.engine = SimpleNamespace(build_id="build-v3", graph=GraphStub())
        rag.engine.close_stores = lambda: None
        rag.engine.graph.add_node(
            "node-1", source="/private/corpus/source.md", metadata={"page": 3}
        )
        doc = SimpleNamespace(
            page_content="resultado estruturado",
            metadata={
                "node_id": "node-1",
                "source": "/private/corpus/source.md",
                "page": 3,
                "retrieval_role": "seed",
            },
        )
        rag.aquery = AsyncMock(return_value=[doc])
        rag.validate_query_dependencies = lambda: None
        rag.abrief = AsyncMock(return_value=SimpleNamespace(
            node_ids=["node-1"], hubs=["evidência"], neighbors=[], satellites=[]
        ))

        async def answer_stream(_packet, _question):
            yield "Resposta [ref:node-1]"

        rag.answer_stream = answer_stream
        rag.start_background_summarizer = AsyncMock()
        create.return_value = rag
        yield rag


@pytest.fixture
def client(mock_rag, monkeypatch):
    from basinrag.api import server

    monkeypatch.setenv("BASINRAG_ENV", "local_dev")
    monkeypatch.setenv("BASINRAG_ALLOW_KEYLESS_LOCAL", "true")
    monkeypatch.delenv("BASINRAG_API_KEY", raising=False)
    server._ws_hits.clear()
    server._ws_last_cleanup = 0.0
    with patch.object(server, "API_KEY", None):
        with TestClient(server.app, client=("127.0.0.1", 50000)) as test_client:
            yield test_client


def test_liveness_and_readiness_are_separate(client):
    assert client.get("/v2/livez").json() == {"status": "alive"}
    response = client.get("/v2/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "build_id": "build-v3"}


def test_query_returns_structured_source_without_absolute_path(client):
    response = client.post("/v2/query", json={"query": "tema principal", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["results"] == [{
        "ref_id": "node-1",
        "text": "resultado estruturado",
        "source_label": "source.md",
        "page": 3,
        "rank": 1,
        "role": "seed",
    }]
    assert "private" not in response.text


@pytest.mark.parametrize("body", [
    {"query": " "},
    {"query": "válida", "top_k": 51},
    {"query": "válida", "search_type": "hybrid_typo"},
])
def test_query_rejects_invalid_contract(client, body):
    assert client.post("/v2/query", json=body).status_code == 422


def test_query_returns_503_without_loaded_snapshot(client, mock_rag):
    mock_rag._loaded = False
    assert client.get("/v2/readyz").status_code == 503
    assert client.post("/v2/query", json={"query": "pergunta válida"}).status_code == 503


def test_production_startup_requires_key(monkeypatch, mock_rag):
    monkeypatch.setenv("BASINRAG_ENV", "production")
    monkeypatch.setenv("BASINRAG_ALLOW_KEYLESS_LOCAL", "true")
    with patch.object(server, "API_KEY", None):
        with pytest.raises(RuntimeError, match="BASINRAG_API_KEY"):
            with TestClient(server.app):
                pass


def test_bearer_is_the_only_http_credential(client, monkeypatch):
    monkeypatch.setenv("BASINRAG_API_KEY", "secret-token")
    with patch.object(server, "API_KEY", "secret-token"):
        url_credential = client.post(
            "/v2/query?api_key=secret-token", json={"query": "pergunta válida"}
        )
        legacy_header = client.post(
            "/v2/query", json={"query": "pergunta válida"}, headers={"X-API-Key": "secret-token"}
        )
        bearer = client.post(
            "/v2/query", json={"query": "pergunta válida"},
            headers={"Authorization": "Bearer secret-token"},
        )
    assert url_credential.status_code == 401
    assert legacy_header.status_code == 401
    assert bearer.status_code == 200


def test_origin_allowlist_applies_to_http_and_websocket(client):
    denied = client.post(
        "/v2/query", json={"query": "pergunta válida"}, headers={"Origin": "https://evil.example"}
    )
    assert denied.status_code == 401
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/v2/chat", headers={"Origin": "https://evil.example", "Authorization": "Bearer secret"}
        ):
            pass
    assert exc.value.code == 1008


def test_proxy_ip_does_not_authenticate(monkeypatch):
    monkeypatch.setenv("BASINRAG_TRUSTED_PROXIES", "127.0.0.1/32")
    assert server._rate_client_ip("203.0.113.4", "198.51.100.8") == "203.0.113.4"
    assert server._rate_client_ip("127.0.0.1", "198.51.100.8") == "198.51.100.8"


def test_websocket_emits_validated_json_events_and_citations(client):
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "tema principal", "top_k": 5})
        events = [websocket.receive_json() for _ in range(4)]
    assert [event["type"] for event in events] == ["start", "references", "token", "done"]
    request_id = events[0]["request_id"]
    assert all(event["request_id"] == request_id for event in events)
    assert events[1]["references"][0]["source_label"] == "source.md"
    assert events[2]["text"] == "Resposta [ref:node-1]"


def test_websocket_citation_not_in_retrieved_refs_is_rejected(client, mock_rag):
    async def invalid_answer(_packet, _question):
        yield "Resposta [ref:inventada]"

    mock_rag.answer_stream = invalid_answer
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "tema principal"})
        events = [websocket.receive_json() for _ in range(3)]
    assert [event["type"] for event in events] == ["start", "references", "error"]
    assert events[-1]["code"] == "invalid_citation"


def test_websocket_rejects_oversized_payload(client, monkeypatch):
    monkeypatch.setattr(server, "WS_MAX_MESSAGE_BYTES", 4)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_text("12345")
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_json()
    assert exc.value.code == 1009


@pytest.mark.parametrize("host", [
    None, "", "example.com", "192.0.2.7", "[2001:db8::1]", "[::ffff:192.0.2.1]",
])
def test_loopback_detection_rejects_non_loopback_hosts(host):
    assert not server._is_loopback_host(host)


@pytest.mark.parametrize("host", [
    "localhost", "LOCALHOST.", "service.localhost", "127.0.0.1", "[::1]", "::ffff:127.0.0.1",
])
def test_loopback_detection_accepts_loopback_hosts(host):
    assert server._is_loopback_host(host)


def test_bind_requires_key_except_explicit_loopback_development(monkeypatch):
    monkeypatch.delenv("BASINRAG_API_KEY", raising=False)
    monkeypatch.setenv("BASINRAG_ENV", "local_dev")
    monkeypatch.setenv("BASINRAG_ALLOW_KEYLESS_LOCAL", "yes")
    server.require_api_key_for_public_bind("127.0.0.1")
    with pytest.raises(RuntimeError, match="BASINRAG_API_KEY"):
        server.require_api_key_for_public_bind("0.0.0.0")
    monkeypatch.setenv("BASINRAG_API_KEY", "present")
    server.require_api_key_for_public_bind("0.0.0.0")


@pytest.mark.parametrize("value", [
    "", " , , ", "*", "ftp://example.com", "https://user:pass@example.com",
    "https://example.com/path", "https://example.com?x=1", "https://example.com#frag",
])
def test_origin_configuration_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        server._validated_origins(value)


def test_origin_configuration_normalizes_and_deduplicates():
    assert server._validated_origins(" https://example.com/,https://example.com ") == {
        "https://example.com"
    }


def test_bearer_cookie_and_origin_authentication_rules(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "secret")
    allowed_origin = sorted(server.ALLOWED_ORIGINS)[0]
    assert server._token_from_authorization(None) is None
    assert server._token_from_authorization("Basic secret") is None
    assert server._token_from_authorization("Bearer   ") is None
    assert server._token_from_authorization("bEaReR secret ") == "secret"
    assert server._auth_valid("192.0.2.1", "Bearer secret")
    assert not server._auth_valid("192.0.2.1", "Bearer wrong")
    assert server._auth_valid("192.0.2.1", None, allowed_origin, "secret")
    assert not server._auth_valid("192.0.2.1", None, None, "secret")
    assert not server._auth_valid("192.0.2.1", "Bearer secret", "https://evil.example")
    monkeypatch.setattr(server, "API_KEY", None)
    monkeypatch.setenv("BASINRAG_ENV", "local_dev")
    monkeypatch.setenv("BASINRAG_ALLOW_KEYLESS_LOCAL", "true")
    assert server._auth_valid("127.0.0.1", None)
    assert not server._auth_valid("192.0.2.1", None)


def test_rate_client_ip_parses_trusted_forwarded_chain(monkeypatch):
    monkeypatch.setenv("BASINRAG_TRUSTED_PROXIES", "127.0.0.1/32,10.0.0.0/8")
    assert server._trusted_proxy_networks()
    assert server._rate_client_ip(None) == "unknown"
    assert server._rate_client_ip("not-an-ip") == "not-an-ip"
    assert server._rate_client_ip("203.0.113.4", "198.51.100.8") == "203.0.113.4"
    assert server._rate_client_ip("127.0.0.1", "198.51.100.8, 10.0.0.2") == "198.51.100.8"
    assert server._rate_client_ip("127.0.0.1", "10.0.0.1,10.0.0.2") == "10.0.0.1"
    assert server._rate_client_ip("127.0.0.1", "bad-ip") == "127.0.0.1"


def test_websocket_rate_buckets_expire_and_enforce_limits(monkeypatch):
    server._ws_hits.clear()
    server._ws_last_cleanup = 0
    monkeypatch.setattr(server, "WS_RATE_LIMIT", 2)
    clock = [100.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])
    assert server._ws_rate_allowed("client", "messages")
    assert server._ws_rate_allowed("client", "messages")
    assert not server._ws_rate_allowed("client", "messages")
    clock[0] = 200.0
    assert server._ws_rate_allowed("client", "messages")

    server._ws_hits[("stale", "bucket")] = server.deque([1.0, 150.0])
    server._ws_hits[("empty", "bucket")] = server.deque([1.0])
    server._ws_last_cleanup = 0
    server._cleanup_ws_rate_state(now=200.0)
    assert list(server._ws_hits[("stale", "bucket")]) == [150.0]
    assert ("empty", "bucket") not in server._ws_hits


def test_query_maps_missing_metadata_to_safe_defaults(client, mock_rag):
    mock_rag.aquery.return_value = [SimpleNamespace(page_content="sem metadados", metadata={})]
    response = client.post("/v2/query", json={"query": "pergunta válida"})
    result = response.json()["results"][0]
    assert result["ref_id"] == "1"
    assert result["source_label"] is None
    assert result["page"] is None
    assert result["role"] == "seed"


def test_websocket_rejects_missing_credentials_when_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "secret")
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/chat"):
            pass
    assert exc.value.code == 1008


def test_websocket_rate_limit_and_connection_limit(client, monkeypatch):
    monkeypatch.setattr(server, "_ws_rate_allowed", lambda *_args: False)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/chat"):
            pass
    assert exc.value.code == 1008

    monkeypatch.setattr(server, "_ws_rate_allowed", lambda *_args: True)
    monkeypatch.setattr(server, "_ws_active_connections", server.WS_MAX_CONNECTIONS)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/chat"):
            pass
    assert exc.value.code == 1008
    monkeypatch.setattr(server, "_ws_active_connections", 0)


def test_websocket_not_ready_and_binary_frame_close(client, mock_rag, monkeypatch):
    monkeypatch.setattr(server, "_rag_instance", None)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/chat"):
            pass
    assert exc.value.code == 1013

    monkeypatch.setattr(server, "_rag_instance", mock_rag)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_bytes(b"binary")
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_json()
    assert exc.value.code == 1003


def test_websocket_malformed_message_can_be_followed_by_valid_request(client):
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_text("not-json")
        invalid = websocket.receive_json()
        websocket.send_json({"query": "pergunta válida"})
        events = [websocket.receive_json() for _ in range(4)]
    assert invalid == {"type": "error", "request_id": None, "code": "invalid_request"}
    assert [event["type"] for event in events] == ["start", "references", "token", "done"]


def test_websocket_reports_generation_timeout_and_failure(client, mock_rag, monkeypatch):
    monkeypatch.setattr(server, "WS_GENERATION_TIMEOUT_SECONDS", 0.001)

    async def slow_brief(*_args, **_kwargs):
        await asyncio.sleep(0.02)

    mock_rag.abrief = AsyncMock(side_effect=slow_brief)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "pergunta válida"})
        events = [websocket.receive_json() for _ in range(2)]
    assert events[-1]["code"] == "timeout"

    mock_rag.abrief = AsyncMock(side_effect=RuntimeError("retrieval failed"))
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "pergunta válida"})
        events = [websocket.receive_json() for _ in range(2)]
    assert events[-1]["code"] == "generation_failed"


def test_websocket_cancels_generation_when_client_disconnects(client, mock_rag):
    generation_started = threading.Event()
    generation_cancelled = threading.Event()

    async def wait_for_disconnect(*_args, **_kwargs):
        generation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            generation_cancelled.set()
            raise

    mock_rag.abrief = AsyncMock(side_effect=wait_for_disconnect)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "pergunta válida"})
        assert websocket.receive_json()["type"] == "start"
        assert generation_started.wait(timeout=2)
    assert generation_cancelled.wait(timeout=2)


def test_websocket_rejects_oversized_queued_message_and_queue_overflow(
    client, mock_rag, monkeypatch
):
    monkeypatch.setattr(server, "WS_MAX_MESSAGE_BYTES", 100)

    async def slow_brief(*_args, **_kwargs):
        await asyncio.sleep(1)
        return SimpleNamespace(node_ids=[], hubs=[], neighbors=[], satellites=[])

    mock_rag.abrief = AsyncMock(side_effect=slow_brief)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "pergunta válida"})
        assert websocket.receive_json()["type"] == "start"
        websocket.send_text("x" * 101)
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_json()
    assert exc.value.code == 1009

    monkeypatch.setattr(server, "WS_MAX_MESSAGE_BYTES", 8192)
    with client.websocket_connect("/v2/chat") as websocket:
        websocket.send_json({"query": "pergunta válida"})
        assert websocket.receive_json()["type"] == "start"
        for _ in range(9):
            websocket.send_json({"query": "pergunta válida"})
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_json()
    assert exc.value.code == 1008


def test_lifespan_records_unready_snapshot_and_closes_engine(monkeypatch):
    unready = SimpleNamespace(
        _loaded=False,
        engine=SimpleNamespace(close_stores=Mock()),
        config=SimpleNamespace(enable_background_l3=False),
    )
    with patch.object(server, "API_KEY", "configured"), patch.object(
        server.BasinRAG, "create", return_value=unready
    ):
        with TestClient(server.app):
            assert server._startup_error == "Snapshot v3 não encontrado; execute basinrag reindex"
            with pytest.raises(server.HTTPException, match="Snapshot v3"):
                server._get_rag()
    unready.engine.close_stores.assert_called_once()


def test_lifespan_starts_local_background_summarizer(monkeypatch):
    rag = SimpleNamespace(
        _loaded=True,
        config=SimpleNamespace(
            enable_background_l3=True, provider="local", allow_remote_l3_egress=False
        ),
        engine=SimpleNamespace(build_id="l3-build", close_stores=Mock()),
        validate_query_dependencies=lambda: None,
        start_background_summarizer=AsyncMock(),
    )
    with patch.object(server, "API_KEY", "configured"), patch.object(
        server.BasinRAG, "create", return_value=rag
    ):
        with TestClient(server.app):
            assert server.readyz()["build_id"] == "l3-build"
        rag.start_background_summarizer.assert_awaited_once_with(verbose=True)
    rag.engine.close_stores.assert_called_once()


@pytest.mark.parametrize(("allow_egress", "should_start"), [(False, False), (True, True)])
def test_lifespan_requires_remote_l3_egress_opt_in(allow_egress, should_start):
    rag = SimpleNamespace(
        _loaded=True,
        config=SimpleNamespace(
            enable_background_l3=True, provider="openai", allow_remote_l3_egress=allow_egress
        ),
        engine=SimpleNamespace(build_id="remote-l3", close_stores=Mock()),
        validate_query_dependencies=lambda: None,
        start_background_summarizer=AsyncMock(),
    )
    with patch.object(server, "API_KEY", "configured"), patch.object(
        server.BasinRAG, "create", return_value=rag
    ):
        with TestClient(server.app):
            assert server.readyz()["status"] == "ready"
        assert rag.start_background_summarizer.await_count == int(should_start)
    rag.engine.close_stores.assert_called_once()


def test_lifespan_recovers_failed_factory_as_unready(monkeypatch):
    with patch.object(server, "API_KEY", "configured"), patch.object(
        server.BasinRAG, "create", side_effect=RuntimeError("corrupt snapshot")
    ):
        with TestClient(server.app):
            assert server._startup_error == "corrupt snapshot"
            assert server._rag_instance is not None
            assert server._rag_instance._loaded is False
            with pytest.raises(server.HTTPException) as exc:
                server.readyz()
            assert exc.value.status_code == 503
