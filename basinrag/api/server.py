from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, MutableMapping, cast
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.websockets import WebSocketDisconnect

from ..factory import BasinRAG, BasinRAGConfig
from ..contracts import SearchType
from ..logging_config import setup_logging

logger = setup_logging()

_rag_instance: BasinRAG | None = None
_startup_error: str | None = None
_ws_hits: dict[tuple[str, str], deque[float]] = {}
_ws_last_cleanup = 0.0
_ws_active_connections = 0
_ws_generation_limit: asyncio.Semaphore | None = None

WS_RATE_LIMIT = 30
WS_RATE_WINDOW_SECONDS = 60
WS_MAX_MESSAGE_BYTES = 8192
WS_MAX_CONNECTIONS = 64
WS_MAX_GENERATIONS = 8
WS_GENERATION_TIMEOUT_SECONDS = 120
_DEFAULT_ORIGINS = (
    "http://localhost,http://127.0.0.1,http://localhost:3000,"
    "http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _local_keyless_enabled() -> bool:
    return (
        os.environ.get("BASINRAG_ENV", "production").strip().lower() == "local_dev"
        and os.environ.get("BASINRAG_ALLOW_KEYLESS_LOCAL", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def require_api_key_for_public_bind(host: str) -> None:
    if os.environ.get("BASINRAG_API_KEY"):
        return
    if _local_keyless_enabled() and _is_loopback_host(host):
        return
    raise RuntimeError(
        "A API exige BASINRAG_API_KEY. O modo sem chave só é permitido com "
        "BASINRAG_ENV=local_dev, BASINRAG_ALLOW_KEYLESS_LOCAL=true e bind loopback."
    )


def _validated_origins(value: str) -> set[str]:
    origins = {origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()}
    if not origins:
        raise ValueError("BASINRAG_CORS_ORIGINS deve conter ao menos uma origem")
    if "*" in origins:
        raise ValueError("Wildcard não é permitido para origens de API/WebSocket")
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Origem inválida: {origin!r}")
    return origins


def _token_from_authorization(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if separator and scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _auth_valid(
    client_host: str | None,
    authorization: str | None,
    origin: str | None = None,
    cookie_token: str | None = None,
) -> bool:
    if origin is not None and origin.rstrip("/") not in ALLOWED_ORIGINS:
        return False
    token = _token_from_authorization(authorization)
    if API_KEY:
        token = token or cookie_token
        if not token or not secrets.compare_digest(token.encode(), API_KEY.encode()):
            return False
        # Browser WebSockets cannot set Authorization headers. A session cookie
        # is accepted only for an explicitly allowed same-origin request.
        return authorization is not None or (origin is not None and origin.rstrip("/") in ALLOWED_ORIGINS)
    return _local_keyless_enabled() and _is_loopback_host(client_host)


def _cleanup_ws_rate_state(now: float | None = None) -> None:
    global _ws_last_cleanup
    now = time.monotonic() if now is None else now
    if now - _ws_last_cleanup < WS_RATE_WINDOW_SECONDS:
        return
    cutoff = now - WS_RATE_WINDOW_SECONDS
    for key, hits in list(_ws_hits.items()):
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            _ws_hits.pop(key, None)
    _ws_last_cleanup = now


def _ws_rate_allowed(client: str, bucket: str) -> bool:
    now = time.monotonic()
    _cleanup_ws_rate_state(now)
    key = (client, bucket)
    hits = _ws_hits.setdefault(key, deque())
    cutoff = now - WS_RATE_WINDOW_SECONDS
    while hits and hits[0] <= cutoff:
        hits.popleft()
    if len(hits) >= WS_RATE_LIMIT:
        return False
    hits.append(now)
    return True


def _trusted_proxy_networks():
    networks = []
    for value in os.environ.get("BASINRAG_TRUSTED_PROXIES", "").split(","):
        value = value.strip()
        if value:
            networks.append(ipaddress.ip_network(value, strict=False))
    return networks


def _rate_client_ip(peer: str | None, forwarded_for: str | None = None) -> str:
    """Honor forwarded IPs only when the direct peer is explicitly trusted."""
    if not peer:
        return "unknown"
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if not any(peer_ip in network for network in _trusted_proxy_networks()) or not forwarded_for:
        return str(peer_ip)
    chain = []
    for raw in forwarded_for.split(","):
        try:
            chain.append(ipaddress.ip_address(raw.strip()))
        except ValueError:
            return str(peer_ip)
    for candidate in reversed(chain):
        if not any(candidate in network for network in _trusted_proxy_networks()):
            return str(candidate)
    return str(chain[0]) if chain else str(peer_ip)


def _http_rate_address(request: Request) -> str:
    peer = request.client.host if request.client else None
    return _rate_client_ip(peer, request.headers.get("x-forwarded-for"))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    search_type: SearchType = "auto"
    top_k: int = Field(5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("query must contain at least two non-whitespace characters")
        return value


class SearchResult(BaseModel):
    ref_id: str
    text: str
    source_label: str | None = None
    page: int | str | None = None
    rank: int
    role: Literal["seed", "context"]


class QueryResponse(BaseModel):
    request_id: str
    results: list[SearchResult]


def _get_rag() -> BasinRAG:
    if _rag_instance is None or not _rag_instance._loaded:
        detail = _startup_error or "Nenhum snapshot compatível foi carregado"
        raise HTTPException(status_code=503, detail=detail)
    return _rag_instance


async def _ws_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(WS_RATE_WINDOW_SECONDS)
        _cleanup_ws_rate_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_instance, _startup_error, _ws_generation_limit
    BasinRAGConfig.load_environment()
    if not API_KEY and not _local_keyless_enabled():
        raise RuntimeError("BASINRAG_API_KEY é obrigatório fora do modo local_dev explícito")
    _startup_error = None
    try:
        _rag_instance = BasinRAG.create()
        if _rag_instance._loaded:
            _rag_instance.validate_query_dependencies()
        else:
            _startup_error = "Snapshot não encontrado; execute basinrag reindex"
    except Exception as exc:
        logger.exception("BasinRAG iniciado em estado não pronto")
        _startup_error = str(exc)
        config = BasinRAGConfig.from_env()
        _rag_instance = BasinRAG(config)

    _ws_generation_limit = asyncio.Semaphore(WS_MAX_GENERATIONS)
    cleanup_task = asyncio.create_task(_ws_cleanup_loop())
    l3_task = None
    if _rag_instance and _rag_instance._loaded and _rag_instance.config.enable_background_l3:
        if (
            _rag_instance.config.provider.lower() == "openai"
            and not _rag_instance.config.allow_remote_l3_egress
        ):
            logger.warning("L3 remoto solicitado sem allow_remote_l3_egress; sumarização não iniciada")
        else:
            l3_task = asyncio.create_task(_rag_instance.start_background_summarizer(verbose=True))
    try:
        yield
    finally:
        tasks = [cleanup_task]
        if l3_task is not None:
            l3_task.cancel()
            tasks.append(l3_task)
        cleanup_task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if _rag_instance is not None:
            _rag_instance.engine.close_stores()
        _rag_instance = None
        _ws_generation_limit = None


BasinRAGConfig.load_environment()
API_KEY = os.environ.get("BASINRAG_API_KEY")
ALLOWED_ORIGINS = _validated_origins(os.environ.get("BASINRAG_CORS_ORIGINS", _DEFAULT_ORIGINS))
app = FastAPI(
    title="BasinRAG API",
    description="Hybrid retrieval API with structured references",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
limiter = Limiter(key_func=_http_rate_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))


async def verify_api_key(request: Request, authorization: str | None = Header(default=None)) -> None:
    origin = request.headers.get("origin")
    if not _auth_valid(
        request.client.host if request.client else None,
        authorization,
        origin,
        request.cookies.get("basinrag_session"),
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/livez")
def livez():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    rag = _get_rag()
    return {"status": "ready", "build_id": rag.engine.build_id}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    rag = _get_rag()
    docs = await rag.aquery(body.query, search_type=body.search_type, top_k=body.top_k)
    results = []
    for rank, document in enumerate(docs, 1):
        metadata = document.metadata or {}
        source = metadata.get("source_label") or metadata.get("source") or ""
        source_label = Path(str(source)).name if source else None
        page = metadata.get("page")
        results.append(SearchResult(
            ref_id=str(metadata.get("node_id") or metadata.get("doc_id") or rank),
            text=document.page_content,
            source_label=source_label,
            page=page,
            rank=rank,
            role="context" if metadata.get("retrieval_role") == "context" else "seed",
        ))
    return QueryResponse(request_id=uuid.uuid4().hex, results=results)


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    global _ws_active_connections
    peer = websocket.client.host if websocket.client else "unknown"
    client = _rate_client_ip(peer, websocket.headers.get("x-forwarded-for"))
    origin = websocket.headers.get("origin")
    if not _auth_valid(
        peer,
        websocket.headers.get("authorization"),
        origin,
        websocket.cookies.get("basinrag_session"),
    ):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    if _ws_active_connections >= WS_MAX_CONNECTIONS or not _ws_rate_allowed(client, "connections"):
        await websocket.close(code=1008, reason="Rate limited")
        return
    rag = _rag_instance
    if rag is None or not rag._loaded:
        await websocket.close(code=1013, reason="Index not ready")
        return
    await websocket.accept()
    _ws_active_connections += 1
    pending_messages: deque[MutableMapping[str, Any]] = deque()
    receive_task: asyncio.Task[MutableMapping[str, Any]] | None = asyncio.create_task(
        websocket.receive()
    )
    generation_task: asyncio.Task[None] | None = None

    async def receive_checked():
        nonlocal receive_task
        if pending_messages:
            return pending_messages.popleft()
        if receive_task is None:
            receive_task = asyncio.create_task(websocket.receive())
        message = await receive_task
        receive_task = None
        return message

    async def generate(request_data, request_id):
        try:
            assert _ws_generation_limit is not None
            async with _ws_generation_limit:
                async with asyncio.timeout(WS_GENERATION_TIMEOUT_SECONDS):
                    packet = await rag.abrief(
                        request_data.query,
                        search_type=request_data.search_type,
                        top_k=request_data.top_k,
                    )
                    refs = []
                    allowed_refs = set()
                    for index, ref_id in enumerate(packet.node_ids):
                        if ref_id not in rag.engine.graph:
                            continue
                        node = rag.engine.graph.nodes[ref_id]
                        metadata = node.get("metadata") or {}
                        source = str(node.get("source") or "")
                        allowed_refs.add(str(ref_id))
                        refs.append({
                            "ref_id": str(ref_id),
                            "source_label": Path(source).name if source else None,
                            "page": metadata.get("page"),
                            "role": "seed" if index < len(packet.hubs) else "context",
                        })
                    await websocket.send_json({
                        "type": "references", "request_id": request_id, "references": refs
                    })
                    answer = []
                    async for token in rag.answer_stream(packet, request_data.query):
                        answer.append(token)
                    answer_text = "".join(answer)
                    cited = set(re.findall(r"\[ref:([^\]]+)\]", answer_text))
                    if cited - allowed_refs:
                        await websocket.send_json({
                            "type": "error", "request_id": request_id, "code": "invalid_citation"
                        })
                        return
                    # Buffer until citations have been validated, then emit bounded
                    # chunks so clients never receive a citation the server rejects.
                    for offset in range(0, len(answer_text), 128):
                        await websocket.send_json({
                            "type": "token", "request_id": request_id,
                            "text": answer_text[offset:offset + 128],
                        })
                    await websocket.send_json({"type": "done", "request_id": request_id})
        except TimeoutError:
            await websocket.send_json({"type": "error", "request_id": request_id, "code": "timeout"})
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Falha ao processar geração WebSocket")
            await websocket.send_json({"type": "error", "request_id": request_id, "code": "generation_failed"})

    try:
        while True:
            message = await receive_checked()
            if message.get("type") == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                await websocket.close(code=1003, reason="JSON text frames required")
                return
            if len(raw.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
                await websocket.close(code=1009, reason="Message too large")
                return
            try:
                request_data = QueryRequest.model_validate_json(raw)
            except Exception:
                await websocket.send_json({"type": "error", "request_id": None, "code": "invalid_request"})
                continue
            if not _ws_rate_allowed(client, "messages"):
                await websocket.close(code=1008, reason="Rate limited")
                return
            request_id = uuid.uuid4().hex
            await websocket.send_json({"type": "start", "request_id": request_id})
            generation_task = asyncio.create_task(generate(request_data, request_id))
            if receive_task is None:
                receive_task = asyncio.create_task(websocket.receive())
            while not generation_task.done():
                done, _pending = await asyncio.wait(
                    {generation_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if receive_task in done:
                    next_message = receive_task.result()
                    receive_task = None
                    if next_message.get("type") == "websocket.disconnect":
                        generation_task.cancel()
                        await asyncio.gather(generation_task, return_exceptions=True)
                        return
                    next_text = next_message.get("text")
                    if next_text is not None and len(next_text.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
                        generation_task.cancel()
                        await asyncio.gather(generation_task, return_exceptions=True)
                        await websocket.close(code=1009, reason="Message too large")
                        return
                    if len(pending_messages) >= 8:
                        generation_task.cancel()
                        await asyncio.gather(generation_task, return_exceptions=True)
                        await websocket.close(code=1008, reason="Too many queued messages")
                        return
                    pending_messages.append(next_message)
                    receive_task = asyncio.create_task(websocket.receive())
            await generation_task
            generation_task = None
    except WebSocketDisconnect:
        return
    finally:
        if generation_task is not None and not generation_task.done():
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
        _ws_active_connections = max(0, _ws_active_connections - 1)
