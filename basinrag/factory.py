import asyncio
from dataclasses import dataclass
from typing import Optional
from .logging_config import setup_logging

logger = setup_logging()
from .core.topology import BasinTopologyEngine
from .core.persistence import BasinPersistence
from .core.llm import UniversalLLM
from .indexer.ingestor import BasinIngestor
from .indexer.summarizer import BasinSummarizer
from .indexer.bm25 import BM25Index
from .retriever.base import BasinRAGRetriever
from .retriever.briefing import BriefingPacket
from .retriever.prompts import QUERY_PROMPT


@dataclass
class BasinRAGConfig:
    provider: str = "ollama"
    model_name: str = "qwen2.5"
    encoder_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    storage_dir: str = ".basinrag"
    search_type: str = "auto"
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_confidence: float = 0.15
    query_prompt: str = QUERY_PROMPT
    use_rerank: bool = True


class BasinRAG:
    """Facade unica para todo o sistema BasinRAG."""

    def __init__(self, config: Optional[BasinRAGConfig] = None):
        self.config = config or BasinRAGConfig()
        self.engine = BasinTopologyEngine(storage_dir=self.config.storage_dir)
        self.persistence = BasinPersistence(self.config.storage_dir)
        self.ingestor = BasinIngestor(
            self.config.encoder_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.llm = None
        self.summarizer = BasinSummarizer(
            self.engine,
            provider=self.config.provider,
            model_name=self.config.model_name,
        )
        self.retriever: Optional[BasinRAGRetriever] = None
        self._loaded = False
        self._inference_semaphore: Optional[asyncio.Semaphore] = None


    @classmethod
    def create(cls, **kwargs) -> "BasinRAG":
        config = BasinRAGConfig(**kwargs)
        instance = cls(config)
        instance.load()
        return instance

    def load(self) -> bool:
        if self.persistence.load_topology(self.engine):
            self._loaded = True
            self.retriever = None
            stored = getattr(self.engine, "encoder_model", "") or ""
            if stored and stored != self.config.encoder_model:
                raise ValueError(
                    f"Aviso Crítico: O índice vetorial foi gerado com {stored}, "
                    f"mas a configuração atual está tentando ler com {self.config.encoder_model}. Apague a pasta .basinrag e re-ingira o corpus."
                )
        return self._loaded

    def load_or_ingest(self, docs_path: str = "docs") -> bool:
        if self.load():
            return True
        return self.ingest(docs_path) > 0

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = UniversalLLM(self.config.provider, self.config.model_name)
        return self.llm

    def _attach_bm25(self) -> None:
        ids = list(self.engine.graph.nodes)
        texts = [self.engine.graph.nodes[n].get("text", "") for n in ids]
        index = BM25Index()
        index.build(ids, texts)
        self.engine.bm25 = index

    def _load_nodes(self, path: str):
        import os
        if os.path.isdir(path):
            return self.ingestor.ingest_directory(path)
        return self.ingestor.ingest(path)

    def ingest(self, path: str) -> int:
        """Merge new documents into the existing graph. Use reindex() to rebuild from scratch."""
        nodes = self._load_nodes(path)
        if not nodes:
            logger.info("Nenhum documento ingerido; o indice anterior foi preservado.")
            return 0
        self.engine.encoder_model = self.config.encoder_model
        if self.engine.graph.number_of_nodes() == 0:
            self.engine.build_graph(nodes)
        else:
            self.engine.merge_graph(nodes)
        self.engine.partition_into_basins()
        self._attach_bm25()
        self.retriever = None
        self.persistence.save_topology(self.engine)
        logger.info(
            "Resumos L3 LLM serao gerados em background ao iniciar "
            "'basinrag chat' ou 'basinrag serve'. L3 extractivo ja esta no indice."
        )
        return len(nodes)

    def reindex(self, path: str) -> int:
        """Wipe the graph and rebuild from path."""
        nodes = self._load_nodes(path)
        if not nodes:
            logger.info("Nenhum documento ingerido; o indice anterior foi preservado.")
            return 0
        self.engine.encoder_model = self.config.encoder_model
        self.engine.build_graph(nodes)
        self.engine.partition_into_basins()
        self._attach_bm25()
        self.retriever = None
        self.persistence.save_topology(self.engine)
        return len(nodes)

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._inference_semaphore is None:
            self._inference_semaphore = asyncio.Semaphore(2)
        return self._inference_semaphore

    def query(self, question: str, search_type: str = None, top_k: int = None) -> list:
        self._ensure_retriever()
        packet = self.retriever.brief(question, search_type=search_type, top_k=top_k)
        from langchain_core.documents import Document
        docs = []
        for i, text in enumerate(packet.texts_for_rerank()[: (top_k or self.retriever.top_k)]):
            meta = {}
            if i < len(packet.node_ids):
                nid = packet.node_ids[i]
                if nid in self.engine.graph:
                    node_data = self.engine.graph.nodes[nid]
                    meta = dict(node_data.get("metadata") or {})
                    meta["node_id"] = nid
                    meta["source"] = node_data.get("source", "")
                    meta["doc_id"] = meta.get("doc_id") or node_data.get("source", "")
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    async def aquery(self, question: str, search_type: str = None, top_k: int = None) -> list:
        self._ensure_retriever()
        async with self._get_semaphore():
            return await asyncio.to_thread(self.query, question, search_type, top_k)

    def brief(self, question: str, search_type: str = None, top_k: int = None) -> BriefingPacket:
        self._ensure_retriever()
        return self.retriever.brief(question, search_type=search_type, top_k=top_k)

    async def abrief(self, question: str, search_type: str = None, top_k: int = None) -> BriefingPacket:
        self._ensure_retriever()
        async with self._get_semaphore():
            return await asyncio.to_thread(self.retriever.brief, question, search_type, top_k)

    def _ensure_retriever(self, search_type: str = None, top_k: int = None) -> None:
        if self.retriever is None:
            self.retriever = BasinRAGRetriever(
                engine=self.engine,
                encoder=self.ingestor.encoder,
                search_type=self.config.search_type,
                top_k=5,
                reranker_model=self.config.reranker_model,
                query_prompt=self.config.query_prompt,
                use_rerank=self.config.use_rerank,
            )
        if search_type is not None or top_k is not None:
            self.retriever.configure(
                search_type=search_type if search_type is not None else None,
                top_k=top_k if top_k is not None else None,
            )

    async def start_background_summarizer(self, verbose: bool = False):
        await self.summarizer.summarize_missing_background(
            persistence=self.persistence,
            interval=1.0,
            verbose=verbose,
        )

    async def chat(self, question: str, system_prompt: str = None):
        packet = await self.abrief(question)
        context = packet.as_context()
        if not packet.hubs and not packet.neighbors:
            yield (
                "Nenhum trecho recuperado. Confirme que o indice existe "
                "(basinrag ingest <pasta>) e tente outra pergunta."
            )
            return
        if packet.confidence < self.config.min_confidence:
            yield (
                "Nenhum trecho suficientemente relevante para esta pergunta. "
                "Tente termos mais especificos ou outra formulacao."
            )
            return
        sys_prompt = system_prompt or "Responda baseando-se APENAS no contexto fornecido."
        user_prompt = f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
        async for token in self._ensure_llm().chat_stream(sys_prompt, user_prompt):
            yield token

