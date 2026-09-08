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
from .retriever.briefing import BriefingPacket, MIN_CONFIDENCE


@dataclass
class BasinRAGConfig:
    provider: str = "ollama"
    model_name: str = "qwen2.5"
    encoder_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    storage_dir: str = ".basinrag"
    search_type: str = "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    hybrid_alpha: float = 0.55
    rrf_k: int = 60
    similarity_threshold: float = 0.55
    knn_threshold: float = 0.85
    min_confidence: float = 0.15
    hop_lambda: float = 0.35
class BasinRAG:
    """Facade unica para todo o sistema BasinRAG."""

    def __init__(self, config: Optional[BasinRAGConfig] = None):
        self.config = config or BasinRAGConfig()
        self.engine = BasinTopologyEngine(storage_dir=self.config.storage_dir)
        self.persistence = BasinPersistence(self.config.storage_dir)
        self.ingestor = BasinIngestor(self.config.encoder_model)
        self.llm = None
        self.summarizer = BasinSummarizer(
            self.engine,
            provider=self.config.provider,
            model_name=self.config.model_name,
        )
        self.retriever: Optional[BasinRAGRetriever] = None
        self._loaded = False

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

    def ingest(self, path: str) -> int:
        """Ingest a file or directory, replacing any in-memory graph."""
        import os
        if os.path.isdir(path):
            nodes = self.ingestor.ingest_directory(path)
        else:
            nodes = self.ingestor.ingest(path)
        if not nodes:
            logger.info("Nenhum documento ingerido; o indice anterior foi preservado.")
            return 0
        self.engine.encoder_model = self.config.encoder_model
        self.engine.build_graph(nodes)
        self.engine.partition_into_basins()
        self._attach_bm25()
        self.retriever = None
        self.persistence.save_topology(self.engine)
        logger.info(
            "Resumos L3 LLM serao gerados em background ao iniciar "
            "'basinrag chat' ou 'basinrag serve'. L3 extractivo ja esta no indice."
        )
        return len(nodes)

    def query(self, question: str, search_type: str = None, top_k: int = None) -> list:
        self._ensure_retriever(search_type, top_k)
        return self.retriever.invoke(question)

    def brief(self, question: str, search_type: str = None, top_k: int = None) -> BriefingPacket:
        self._ensure_retriever(search_type, top_k)
        return self.retriever.brief(question)

    def _ensure_retriever(self, search_type: str = None, top_k: int = None) -> None:
        resolved_type = search_type if search_type is not None else self.config.search_type
        if self.retriever is None:
            self.retriever = BasinRAGRetriever(
                engine=self.engine,
                encoder=self.ingestor.encoder,
                search_type=resolved_type,
                top_k=top_k if top_k is not None else 5,
                reranker_model=self.config.reranker_model,
            )
            return
        self.retriever.configure(search_type=resolved_type, top_k=top_k)

    async def start_background_summarizer(self, verbose: bool = False):
        await self.summarizer.summarize_missing_background(
            persistence=self.persistence,
            interval=1.0,
            verbose=verbose,
        )

    async def chat(self, question: str, system_prompt: str = None):
        packet = self.brief(question)
        context = packet.as_context()
        if not packet.hubs and not packet.neighbors:
            yield (
                "Nenhum trecho recuperado. Confirme que o indice existe "
                "(basinrag ingest <pasta>) e tente outra pergunta."
            )
            return
        if packet.confidence < MIN_CONFIDENCE:
            yield (
                "Nenhum trecho suficientemente relevante para esta pergunta. "
                "Tente termos mais especificos ou outra formulacao."
            )
            return
        sys_prompt = system_prompt or "Responda baseando-se APENAS no contexto fornecido."
        user_prompt = f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
        async for token in self._ensure_llm().chat_stream(sys_prompt, user_prompt):
            yield token
