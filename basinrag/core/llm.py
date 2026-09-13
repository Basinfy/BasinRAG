import asyncio
import logging
import math
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

class UniversalLLM:
    """Abstração universal para provedores LLM."""
    
    def __init__(self, provider: str = "ollama", model_name: str = "llama3"):
        self.provider = provider.lower()
        self.model_name = model_name
        self._llm: Any = None
        self._tokenizer: Any = None
        self._tokenizer_checked = False

    def count_tokens(self, text: str) -> int:
        """Count with the generator tokenizer when available, else conservatively by script."""
        text = text or ""
        if not text:
            return 0
        if not self._tokenizer_checked:
            self._tokenizer_checked = True
            if self.provider in ("openai", "open-ai"):
                try:
                    import tiktoken
                    try:
                        self._tokenizer = tiktoken.encoding_for_model(self.model_name)
                    except KeyError:
                        self._tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._tokenizer = None
            else:
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                except Exception:
                    self._tokenizer = None
        if self._tokenizer is not None:
            try:
                if hasattr(self._tokenizer, "encode"):
                    try:
                        return len(self._tokenizer.encode(text, add_special_tokens=False))
                    except TypeError:
                        return len(self._tokenizer.encode(text))
                return len(self._tokenizer(text)["input_ids"])
            except Exception:
                pass

        cjk = 0
        latin = 0
        punctuation = 0
        for char in text:
            if char.isspace():
                continue
            if (
                "\u3040" <= char <= "\u30ff"
                or "\u3400" <= char <= "\u9fff"
                or "\uac00" <= char <= "\ud7af"
            ):
                cjk += 1
            elif char.isalnum() or char == "_":
                latin += 1
            else:
                punctuation += 1
        # Roughly 2.5 Latin letters per token is intentionally conservative;
        # CJK ideographs often consume one token each.
        return max(1, cjk + math.ceil(latin / 2.5) + math.ceil(punctuation / 2))

    def _ensure_ollama(self):
        if self._llm is not None:
            return
        try:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(model=self.model_name, temperature=0.3)
        except ImportError:
            raise ImportError(
                "langchain-ollama não instalado. "
                "Instale com: pip install langchain-ollama"
            )
    
    async def chat_stream(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Stream de tokens, agnóstico ao provedor."""
        if self.provider in ("openai", "open-ai"):
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai não instalado. "
                    "Instale com: pip install langchain-openai"
                )
            if self._llm is None:
                self._llm = ChatOpenAI(model=self.model_name, temperature=0.3)
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            async for chunk in self._llm.astream(messages):
                yield chunk.content

        elif self.provider == "ollama":
            self._ensure_ollama()
            if not self._llm:
                raise RuntimeError("Motor Ollama não inicializado.")
                
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            async for chunk in self._llm.astream(messages):
                yield chunk.content
                
        else:
            raise ValueError(f"Provedor {self.provider} não suportado.")

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Versão não-streaming com retry exponencial."""
        for attempt in range(MAX_RETRIES):
            try:
                result = []
                async for token in self.chat_stream(system_prompt, user_prompt):
                    result.append(token)
                return "".join(result)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"LLM falhou após {MAX_RETRIES} tentativas: {e}")
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"LLM tentativa {attempt+1} falhou, retry em {delay}s: {e}")
                await asyncio.sleep(delay)
        raise RuntimeError("Loop de retry LLM terminou inesperadamente")
