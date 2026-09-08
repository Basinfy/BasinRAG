from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List
import asyncio
from ..logging_config import setup_logging

logger = setup_logging()
from ..factory import BasinRAG
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import os
import secrets

_rag_instance: BasinRAG | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    search_type: str = "auto"
    top_k: int = Field(5, ge=1, le=50)


class QueryResponse(BaseModel):
    results: List[str]


def _get_rag() -> BasinRAG:
    if _rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG not initialized yet")
    return _rag_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_instance
    logger.info("Inicializando BasinRAG...")
    _rag_instance = BasinRAG.create()
    bg_task = asyncio.create_task(_rag_instance.start_background_summarizer(verbose=True))
    logger.info("BasinRAG Pronto!")
    try:
        yield
    finally:
        bg_task.cancel()
        try:
            results = await asyncio.gather(bg_task, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logger.error(f"Erro na task de background summarizer: {res}")
        except Exception as e:
            logger.error(f"Erro inesperado no encerramento da task de background: {e}")
        if _rag_instance:
            _rag_instance.persistence.save_topology(_rag_instance.engine)
        _rag_instance = None


app = FastAPI(
    title="BasinRAG API",
    description="Topological RAG API",
    lifespan=lifespan,
)

cors_origins = [o.strip() for o in os.environ.get("BASINRAG_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=(cors_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = os.environ.get("BASINRAG_API_KEY")

async def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY:
        if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
            raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/")
def read_root():
    rag = _get_rag()
    return {"status": "online", "model": rag.config.model_name}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    rag = _get_rag()
    docs = await rag.aquery(
        body.query,
        search_type=body.search_type,
        top_k=body.top_k,
    )
    return {"results": [d.page_content for d in docs]}



@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket, token: str = Query(None)):
    if API_KEY:
        if not token or not secrets.compare_digest(token, API_KEY):
            await websocket.close(code=1008, reason="Unauthorized")
            return
    await websocket.accept()
    rag = _get_rag()
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            async for token_chunk in rag.chat(data):
                await websocket.send_text(token_chunk)
            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado pelo cliente normalmente.")
    except asyncio.TimeoutError:
        await websocket.close(code=1000, reason="Timeout")
    except Exception:
        logger.exception("WebSocket desconectado com erro")

