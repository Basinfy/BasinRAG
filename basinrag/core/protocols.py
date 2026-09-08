from typing import AsyncGenerator, Protocol, runtime_checkable, Any

@runtime_checkable
class LLMProvider(Protocol):
    async def chat_stream(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]: ...
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...

@runtime_checkable
class Encoder(Protocol):
    def encode(self, text: str | list[str], **kwargs) -> Any: ...
