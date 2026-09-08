from __future__ import annotations

from typing import AsyncGenerator, Protocol, runtime_checkable, Any, Dict, List, Optional, Sequence


@runtime_checkable
class LLMProviderProtocol(Protocol):
    async def chat_stream(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]: ...
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...


# Alias para retrocompatibilidade
LLMProvider = LLMProviderProtocol


@runtime_checkable
class EncoderProtocol(Protocol):
    def encode(self, text: str | list[str], **kwargs) -> Any: ...


# Alias para retrocompatibilidade
Encoder = EncoderProtocol


@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(self, query: str, candidates: Sequence[str], top_k: int = 5) -> List[str]: ...


@runtime_checkable
class KVStoreProtocol(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def pop(self, key: str, default: Any = None) -> Any: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class TopologyEngineProtocol(Protocol):
    graph: Any
    basins: Dict[str, Any]
    meta_basins: Dict[str, Any]
    def build_graph(self, chunks: List[Dict[str, Any]]) -> None: ...
    def partition_into_basins(self) -> None: ...
    def build_meta_basins(self, similarity_threshold: float = 0.70) -> None: ...


@runtime_checkable
class PersistenceProtocol(Protocol):
    def save_topology(self, engine: Any) -> bool: ...
    def load_topology(self, engine: Any) -> bool: ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    def brief(self, query: str) -> Any: ...
    def invoke(self, query: str) -> List[Any]: ...

