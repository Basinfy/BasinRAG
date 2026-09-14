import asyncio
import hashlib
import os
import pickle
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Optional
from .model_revisions import default_huggingface_revision, resolve_huggingface_revision
from .contracts import RankingMode, SearchType
from .logging_config import setup_logging

logger = setup_logging()
from .core.topology import BasinTopologyEngine
from .core.persistence import BasinPersistence
from .core.llm import UniversalLLM
from .indexer.ingestor import BasinIngestor
from .indexer.summarizer import BasinSummarizer
from .indexer.bm25 import BM25Index, current_stemmer_version
from .retriever.base import BasinRAGRetriever
from .retriever.fusion import index_is_flat
from .retriever.briefing import BriefingPacket
from .retriever.prompts import QUERY_PROMPT, build_rag_prompts


@dataclass
class BasinRAGConfig:
    provider: str = "ollama"
    model_name: str = "qwen2.5"
    encoder_model: str = "BAAI/bge-base-en-v1.5"
    encoder_revision: Optional[str] = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: Optional[str] = None
    llm_revision: str = "unresolved"
    storage_dir: str = ".basinrag"
    search_type: SearchType = "auto"
    # Legacy splitter units are characters. Token-aware settings are additive.
    chunk_size: int = 512
    chunk_overlap: int = 128
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    embedding_batch_size: int = 64
    source_encoding: Optional[str] = None
    bm25_stemming: bool = False
    context_window_tokens: int = 8192
    generation_reserve_tokens: int = 1024
    min_confidence: float = 0.15
    query_prompt: str = QUERY_PROMPT
    use_rerank: bool = False
    ranking_mode: RankingMode = "hybrid_rrf"
    enable_background_l3: bool = False
    allow_remote_l3_egress: bool = False

    @staticmethod
    def load_environment() -> None:
        """Load a local .env without overriding variables supplied by the process."""
        from dotenv import load_dotenv

        load_dotenv(override=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> "BasinRAGConfig":
        """Resolve explicit values > process environment/.env > class defaults."""
        import os

        cls.load_environment()
        env = {
            "provider": os.getenv("BASINRAG_LLM_PROVIDER"),
            "model_name": os.getenv("BASINRAG_LLM_MODEL"),
            "encoder_model": os.getenv("BASINRAG_ENCODER_MODEL"),
            "encoder_revision": os.getenv("BASINRAG_ENCODER_REVISION"),
            "reranker_model": os.getenv("BASINRAG_RERANKER_MODEL"),
            "reranker_revision": os.getenv("BASINRAG_RERANKER_REVISION"),
            "llm_revision": os.getenv("BASINRAG_LLM_REVISION"),
            "storage_dir": os.getenv("BASINRAG_STORAGE_DIR"),
            "search_type": os.getenv("BASINRAG_SEARCH_TYPE"),
            "chunk_size": os.getenv("BASINRAG_CHUNK_SIZE"),
            "chunk_overlap": os.getenv("BASINRAG_CHUNK_OVERLAP"),
            "chunk_size_tokens": os.getenv("BASINRAG_CHUNK_SIZE_TOKENS"),
            "chunk_overlap_tokens": os.getenv("BASINRAG_CHUNK_OVERLAP_TOKENS"),
            "embedding_batch_size": os.getenv("BASINRAG_EMBEDDING_BATCH_SIZE"),
            "context_window_tokens": os.getenv("BASINRAG_CONTEXT_WINDOW_TOKENS"),
            "generation_reserve_tokens": os.getenv("BASINRAG_GENERATION_RESERVE_TOKENS"),
            "source_encoding": os.getenv("BASINRAG_SOURCE_ENCODING"),
            "min_confidence": os.getenv("BASINRAG_MIN_CONFIDENCE"),
            "use_rerank": os.getenv("BASINRAG_USE_RERANK"),
            "bm25_stemming": os.getenv("BASINRAG_BM25_STEMMING"),
            "ranking_mode": os.getenv("BASINRAG_RANKING_MODE"),
            "enable_background_l3": os.getenv("BASINRAG_ENABLE_BACKGROUND_L3"),
            "allow_remote_l3_egress": os.getenv("BASINRAG_ALLOW_REMOTE_L3_EGRESS"),
            "query_prompt": os.getenv("BASINRAG_QUERY_PROMPT"),
        }
        values: dict[str, Any] = {key: value for key, value in env.items() if value is not None}
        for key in (
            "chunk_size", "chunk_overlap", "chunk_size_tokens", "chunk_overlap_tokens",
            "embedding_batch_size",
            "context_window_tokens", "generation_reserve_tokens",
        ):
            if key in values:
                values[key] = int(values[key])
        if "min_confidence" in values:
            values["min_confidence"] = float(values["min_confidence"])
        if "use_rerank" in values:
            value = values["use_rerank"].strip().lower()
            if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                raise ValueError("BASINRAG_USE_RERANK must be a boolean value")
            values["use_rerank"] = value in {"1", "true", "yes", "on"}
        for key in ("enable_background_l3", "allow_remote_l3_egress", "bm25_stemming"):
            if key in values:
                value = values[key].strip().lower()
                if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                    raise ValueError(f"{key} must be a boolean value")
                values[key] = value in {"1", "true", "yes", "on"}
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        if self.encoder_revision is None:
            self.encoder_revision = default_huggingface_revision(self.encoder_model)
        if self.encoder_revision is not None:
            self.encoder_revision = resolve_huggingface_revision(
                self.encoder_model, self.encoder_revision
            )
        elif self.encoder_model != "unconfigured":
            raise ValueError(
                "encoder_revision deve ser um SHA completo para modelos customizados"
            )
        if self.reranker_revision is None:
            self.reranker_revision = default_huggingface_revision(self.reranker_model)
        if self.reranker_revision is not None:
            self.reranker_revision = resolve_huggingface_revision(
                self.reranker_model, self.reranker_revision
            )
        elif self.use_rerank and self.reranker_model != "unconfigured":
            raise ValueError(
                "reranker_revision deve ser um SHA completo para rerankers customizados"
            )
        if self.provider.lower() not in {"ollama", "openai"}:
            raise ValueError("provider must be 'ollama' or 'openai'")
        if self.search_type not in {"auto", "local", "global", "hybrid"}:
            raise ValueError("search_type must be one of: auto, local, global, hybrid")
        if self.chunk_size <= 0 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_size must be positive and chunk_overlap must be in [0, chunk_size)")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if self.context_window_tokens <= 0 or not 0 <= self.generation_reserve_tokens < self.context_window_tokens:
            raise ValueError("generation_reserve_tokens must be in [0, context_window_tokens)")
        if self.chunk_size_tokens is not None:
            if self.chunk_size_tokens <= 0:
                raise ValueError("chunk_size_tokens must be positive")
            overlap = self.chunk_overlap_tokens or 0
            if overlap < 0 or overlap >= self.chunk_size_tokens:
                raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")
        elif self.chunk_overlap_tokens is not None:
            raise ValueError("chunk_overlap_tokens requires chunk_size_tokens")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.ranking_mode not in {"hybrid_rrf", "experimental_topology"}:
            raise ValueError("ranking_mode must be 'hybrid_rrf' or 'experimental_topology'")
        if self.allow_remote_l3_egress and not self.enable_background_l3:
            raise ValueError("allow_remote_l3_egress requires enable_background_l3=True")


class BasinRAG:
    """Facade unica para todo o sistema BasinRAG."""

    def __init__(self, config: Optional[BasinRAGConfig] = None):
        self.config = config or BasinRAGConfig.from_env()
        self.engine = BasinTopologyEngine(storage_dir=self.config.storage_dir, open_stores=False)
        self.persistence = BasinPersistence(self.config.storage_dir)
        self._ingestor: Optional[BasinIngestor] = None
        self.llm = None
        self.summarizer = BasinSummarizer(
            self.engine,
            provider=self.config.provider,
            model_name=self.config.model_name,
            storage_dir=self.config.storage_dir,
            enabled=self.config.enable_background_l3,
            allow_remote_egress=self.config.allow_remote_l3_egress,
            model_revision=self.config.llm_revision,
        )
        self.retriever: Optional[BasinRAGRetriever] = None
        self._loaded = False
        self._inference_semaphore: Optional[asyncio.Semaphore] = None
        self._snapshot_lock = threading.RLock()

    @property
    def ingestor(self) -> BasinIngestor:
        if self._ingestor is None:
            self._ingestor = BasinIngestor(
                self.config.encoder_model,
                revision=self.config.encoder_revision,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                chunk_size_tokens=self.config.chunk_size_tokens,
                chunk_overlap_tokens=self.config.chunk_overlap_tokens,
                embedding_batch_size=self.config.embedding_batch_size,
                encoding=self.config.source_encoding,
            )
        return self._ingestor

    @ingestor.setter
    def ingestor(self, value: BasinIngestor) -> None:
        self._ingestor = value


    @classmethod
    def create(cls, *, load_existing_index: bool = True, **kwargs) -> "BasinRAG":
        config = BasinRAGConfig.from_env(**kwargs)
        instance = cls(config)
        if load_existing_index:
            instance.load()
        else:
            # A mandatory major rebuild always targets a new destination.
            instance.engine.build_id = ""
        return instance

    def load(self) -> bool:
        candidate = BasinTopologyEngine(
            section_size=self.engine.section_size,
            storage_dir=self.config.storage_dir,
            open_stores=False,
        )
        if self.persistence.load_topology(candidate):
            self.retriever = None
            stored = getattr(candidate, "encoder_model", "") or ""
            if stored and stored != self.config.encoder_model:
                candidate.close_stores()
                raise ValueError(
                    f"Aviso Crítico: O índice vetorial foi gerado com {stored}, "
                    f"mas a configuração atual seleciona {self.config.encoder_model}. "
                    "Não é seguro misturar vetores de encoders diferentes; execute "
                    "`basinrag reindex <diretorio-do-corpus>` para reconstruir o snapshot."
                )
            previous_engine = self.engine
            self.engine = candidate
            try:
                self._validate_index_compatibility()
            except Exception:
                self.engine = previous_engine
                candidate.close_stores()
                raise
            previous_engine.close_stores()
            self.summarizer.engine = candidate
            self._loaded = True
        else:
            candidate.close_stores()
        return self._loaded

    @staticmethod
    def _normalized_source(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _validate_index_compatibility(self) -> None:
        metadata = getattr(self.engine, "index_metadata", {}) or {}
        schema_version = getattr(self.engine, "index_schema_version", 1)
        if schema_version != 3 or not metadata or int(metadata.get("format_version", 0)) != 3:
            from .core.persistence import IndexRebuildRequired
            raise IndexRebuildRequired(
                "O índice usa um formato incompatível. Informe as fontes e execute "
                "`basinrag reindex <origem> --storage-dir .basinrag`."
            )

        stored_encoder = metadata.get("encoder_model") or self.engine.encoder_model
        if stored_encoder and stored_encoder != self.config.encoder_model:
            raise ValueError(
                f"O snapshot usa o encoder {stored_encoder}, mas a configuração usa "
                f"{self.config.encoder_model}; execute `basinrag reindex <diretorio-do-corpus>`."
            )
        stored_revision = metadata.get("encoder_revision")
        actual_revision = self.config.encoder_revision or getattr(self.ingestor, "model_revision", None)
        if stored_revision not in (None, "unresolved") and actual_revision != stored_revision:
            raise ValueError(
                "A revisão do encoder difere do manifesto do snapshot; execute reindex "
                "em um destino novo."
            )
        tokenizer = getattr(getattr(self.ingestor, "splitter", None), "tokenizer", None)
        current_tokenizer = getattr(tokenizer, "name_or_path", None) or self.config.encoder_model
        stored_tokenizer = metadata.get("tokenizer")
        if stored_tokenizer and stored_tokenizer != current_tokenizer:
            raise ValueError(
                "O tokenizer configurado difere do manifesto do índice; execute "
                "`basinrag reindex <diretorio-do-corpus>`."
            )
        if int(metadata.get("chunk_policy_version", 0)) != 1:
            raise ValueError("A versão da política de chunking mudou; execute reindex.")
        stored_mode = metadata.get("chunking_mode")
        if stored_mode is None:
            stored_mode = "tokens" if metadata.get("chunk_size_tokens") is not None else "characters"
        current_mode = "tokens" if self.config.chunk_size_tokens is not None else "characters"
        if stored_mode != current_mode:
            raise ValueError("O modo de chunking mudou; execute reindex antes de ingerir fontes.")

        if current_mode == "tokens":
            matches = (
                metadata.get("chunk_size_tokens") == self.config.chunk_size_tokens
                and int(metadata.get("chunk_overlap_tokens") or 0)
                == int(self.config.chunk_overlap_tokens or 0)
            )
        else:
            matches = (
                metadata.get("chunk_size") == self.config.chunk_size
                and metadata.get("chunk_overlap") == self.config.chunk_overlap
            )
        if not matches:
            raise ValueError(
                "A política de chunking mudou; execute `basinrag reindex <diretorio-do-corpus>` "
                "antes de ingerir mais fontes."
            )

        stored_stemming = bool(metadata.get("bm25_stemming", False))
        if stored_stemming != self.config.bm25_stemming:
            raise ValueError(
                "A política de stemming BM25 mudou; execute reindex em um destino novo."
            )
        expected_stemmer_version = current_stemmer_version(self.config.bm25_stemming)
        stored_stemmer_version = metadata.get("bm25_stemmer_version", "disabled")
        if stored_stemmer_version != expected_stemmer_version:
            raise ValueError(
                "A versão do stemmer BM25 difere do manifesto; execute reindex em um destino novo."
            )

    def _ensure_write_base(self) -> None:
        current_id = self.persistence.current_build_id()
        if current_id == self.engine.build_id:
            if current_id:
                self._validate_index_compatibility()
            return
        if not self.engine.graph.number_of_nodes() and not self.engine.build_id:
            self.load()
            if self.engine.build_id == current_id:
                return
        raise RuntimeError(
            "O snapshot em memória está desatualizado. Recarregue o índice antes de ingerir fontes."
        )

    def _inventory_sync_sources(self, root: str) -> dict[str, str]:
        supported = {".txt", ".md", ".markdown", ".pdf"}
        skipped = {".git", ".basinrag", "__pycache__", "node_modules", ".venv", "venv"}
        filepaths = []
        errors = []

        def record_error(error):
            errors.append(error)

        for current, dirs, files in os.walk(root, followlinks=False, onerror=record_error):
            dirs[:] = [name for name in dirs if name not in skipped and not name.startswith(".")]
            for filename in files:
                filepath = os.path.abspath(os.path.join(current, filename))
                if os.path.splitext(filename)[1].lower() in supported and not os.path.islink(filepath):
                    filepaths.append(filepath)
        if errors:
            raise OSError(f"Falha ao enumerar fontes sob {root}: {errors[0]}")

        result = {}
        for filepath in sorted(filepaths):
            result[self._normalized_source(filepath)] = self._file_sha256(filepath)
        return result

    def load_or_ingest(self, docs_path: str = "docs") -> bool:
        if self.load():
            return True
        return self.ingest(docs_path) > 0

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = UniversalLLM(self.config.provider, self.config.model_name)
        return self.llm

    def _attach_bm25(self, engine=None) -> None:
        engine = engine or self.engine
        ids = list(engine.graph.nodes)
        texts = [engine.graph.nodes[n].get("text", "") for n in ids]
        index = BM25Index(stemming=self.config.bm25_stemming)
        index.build(ids, texts)
        engine.bm25 = index

    def _load_nodes(self, path: str):
        if os.path.isdir(path):
            return self.ingestor.ingest_directory(path)
        return self.ingestor.ingest(path)

    def _iter_load_nodes(self, path: str):
        """Yield chunks one source at a time when the ingestor supports streaming."""
        if os.path.isdir(path):
            method = getattr(self.ingestor, "iter_ingest_directory", None)
            if callable(method):
                yield from method(path)
            else:
                yield from self.ingestor.ingest_directory(path)
            return
        method = getattr(self.ingestor, "iter_ingest", None)
        if callable(method):
            yield from method(path)
        else:
            yield from self.ingestor.ingest(path)

    @staticmethod
    def _spool_nodes(nodes):
        """Spool a corpus iterator to disk so only one source/batch stays in memory."""
        spool = tempfile.TemporaryFile(mode="w+b")
        count = 0
        sources = set()
        try:
            for node in nodes:
                pickle.dump(node, spool, protocol=pickle.HIGHEST_PROTOCOL)
                count += 1
                if node.get("source"):
                    sources.add(node["source"])
            spool.flush()
            spool.seek(0)
            return spool, count, sources
        except Exception:
            spool.close()
            raise

    @staticmethod
    def _iter_spooled_nodes(spool):
        spool.seek(0)
        while True:
            try:
                yield pickle.load(spool)
            except EOFError:
                return

    @staticmethod
    def _iter_chunks_from_engine(engine, excluded_sources=()):
        excluded_sources = set(excluded_sources)
        for node_id, data in engine.graph.nodes(data=True):
            if data.get("source", "") in excluded_sources:
                continue
            yield {
                "id": node_id,
                "text": data.get("text", ""),
                "embedding": data.get("embedding"),
                "source": data.get("source", ""),
                "chunk_index": data.get("chunk_index", 0),
                "l1": data.get("l1", ""),
                "l2": data.get("l2", ""),
                "l3_summary": data.get("l3_summary", ""),
                "metadata": data.get("metadata") or {},
            }

    @staticmethod
    def _preserved_l3(engine):
        preserved = {}
        for basin_id, basin in engine.basins.items():
            if basin.rho_tree.has_node(basin_id):
                data = basin.rho_tree.nodes[basin_id]
                if data.get("l3_source") == "llm":
                    preserved[basin_id] = {
                        "l3_source": "llm",
                        "l3_summary": data.get("l3_summary", ""),
                    }
        return preserved

    def _publish_snapshot(
        self, chunks, base_engine, *, replace_sources=None, source_file_hashes=None,
        require_ingest_success=False,
    ):
        """Build a new immutable engine and publish it only after persistence succeeds."""
        replace_sources = set(replace_sources or ())
        old_engine = base_engine or self.engine
        with self.persistence.write_transaction(expected_build_id=old_engine.build_id):
            return self._build_and_publish_snapshot(
                chunks,
                old_engine,
                replace_sources=replace_sources,
                source_file_hashes=source_file_hashes,
                require_ingest_success=require_ingest_success,
            )

    def _build_and_publish_snapshot(
        self,
        chunks,
        old_engine,
        *,
        replace_sources,
        source_file_hashes,
        require_ingest_success,
    ):
        build_root = os.path.abspath(self.config.storage_dir)
        os.makedirs(build_root, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=".basinrag-staging-", dir=build_root)
        candidate = BasinTopologyEngine(
            section_size=old_engine.section_size,
            storage_dir=temp_dir,
        )
        candidate.encoder_model = self.config.encoder_model
        incoming_count = 0

        def iter_candidate_chunks():
            nonlocal incoming_count
            yield from self._iter_chunks_from_engine(old_engine, replace_sources)
            for chunk in chunks:
                incoming_count += 1
                yield chunk

        try:
            candidate.build_graph(iter_candidate_chunks())
            failed_sources = [
                source
                for source, status in (getattr(self.ingestor, "last_ingest_status", {}) or {}).items()
                if status == "error"
            ]
            if require_ingest_success and failed_sources:
                examples = ", ".join(sorted(failed_sources)[:3])
                raise RuntimeError(
                    f"Reindex abortado: falha ao processar {len(failed_sources)} fonte(s) ({examples}). "
                    "O snapshot ativo foi preservado."
                )
            candidate.partition_into_basins(
                preserve_l3=self._preserved_l3(old_engine),
                experimental_topology=self.config.ranking_mode == "experimental_topology",
            )
            # build_graph resets engine runtime state, so attach the source
            # generation afterward for the publication compare-and-swap.
            candidate.build_id = getattr(old_engine, "build_id", "") or ""
            splitter = getattr(self.ingestor, "splitter", None)
            tokenizer = getattr(splitter, "tokenizer", None)
            encoder_revision = getattr(self.ingestor, "model_revision", "unresolved")
            if self.config.encoder_revision or encoder_revision not in {None, "", "unresolved", "unknown"}:
                encoder_revision = resolve_huggingface_revision(
                    self.config.encoder_model,
                    self.config.encoder_revision or encoder_revision,
                )
            old_metadata = getattr(old_engine, "index_metadata", {}) or {}
            old_reranker_revision = old_metadata.get("reranker_revision")
            if not self.config.use_rerank:
                reranker_revision = "disabled"
            elif self.config.reranker_revision:
                reranker_revision = resolve_huggingface_revision(
                    self.config.reranker_model, self.config.reranker_revision
                )
            elif (
                old_metadata.get("reranker_model") == self.config.reranker_model
                and old_reranker_revision not in {None, "", "unresolved", "unknown", "disabled"}
            ):
                reranker_revision = old_reranker_revision
            elif hasattr(self.ingestor, "model_revision"):
                reranker_revision = resolve_huggingface_revision(self.config.reranker_model)
            else:
                reranker_revision = "unresolved"
            candidate.index_metadata = {
                "format_version": 3,
                "encoder_model": self.config.encoder_model,
                "encoder_revision": encoder_revision,
                "tokenizer_revision": encoder_revision,
                "reranker_model": self.config.reranker_model if self.config.use_rerank else None,
                "reranker_revision": reranker_revision,
                "tokenizer": getattr(tokenizer, "name_or_path", None) or self.config.encoder_model,
                **self._chunk_policy_metadata(self.config),
                "embedding_batch_size": self.config.embedding_batch_size,
                "ranking_mode": self.config.ranking_mode,
                "bm25_stemming": self.config.bm25_stemming,
                "bm25_stemmer_version": current_stemmer_version(self.config.bm25_stemming),
                "sources": self._source_manifest(self._iter_chunks_from_engine(candidate)),
                "source_file_hashes": (
                    dict(source_file_hashes)
                    if source_file_hashes is not None
                    else self._source_file_manifest(self._iter_chunks_from_engine(candidate))
                ),
            }
            self._attach_bm25(candidate)
            if self.persistence.current_build_id() != old_engine.build_id:
                raise RuntimeError(
                    "O índice ativo mudou durante a operação. Recarregue BasinRAG e tente novamente."
                )
            if not self.persistence.save_topology(candidate):
                raise RuntimeError("Não foi possível publicar o novo build do índice")
            with self._snapshot_lock:
                self.engine = candidate
                self.summarizer.engine = candidate
                self.retriever = None
                self._loaded = True
            return incoming_count
        except Exception:
            candidate.close_stores()
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _chunk_policy_metadata(config):
        if config.chunk_size_tokens is not None:
            return {
                "chunking_mode": "tokens",
                "chunk_policy_version": 1,
                "chunk_size": None,
                "chunk_overlap": None,
                "chunk_size_tokens": config.chunk_size_tokens,
                "chunk_overlap_tokens": config.chunk_overlap_tokens or 0,
            }
        return {
            "chunking_mode": "characters",
            "chunk_policy_version": 1,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunk_size_tokens": None,
            "chunk_overlap_tokens": None,
        }

    @staticmethod
    def _source_manifest(chunks):
        source_ids = {}
        for chunk in chunks:
            source_ids.setdefault(chunk.get("source", ""), []).append(chunk["id"])
        return {
            source: hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
            for source, ids in sorted(source_ids.items())
        }

    @classmethod
    def _source_file_manifest(cls, chunks):
        sources = {chunk.get("source", "") for chunk in chunks if chunk.get("source")}
        manifest = {}
        for source in sorted(sources):
            filepath = cls._source_file_path(source)
            if os.path.isfile(filepath):
                manifest[cls._normalized_source(filepath)] = cls._file_sha256(filepath)
        return manifest

    @staticmethod
    def _source_file_path(source: str) -> str:
        """Resolve JSON document IDs without truncating real paths containing '#'."""
        prefix, marker, suffix = source.rpartition("#")
        if marker and suffix.isdigit() and prefix.lower().endswith(".json"):
            return prefix
        return source

    def ingest(self, path: str) -> int:
        """Merge new documents into the existing graph. Use reindex() to rebuild from scratch."""
        self._ensure_write_base()
        with self.persistence.write_transaction(expected_build_id=self.engine.build_id):
            return self._ingest_locked(path)

    def _ingest_locked(self, path: str) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Caminho de ingestão não encontrado: {path}")
        stored_hashes = dict((self.engine.index_metadata or {}).get("source_file_hashes") or {})
        if os.path.isdir(path):
            before = self._inventory_sync_sources(path)
            changed_existing = sorted(
                source for source, digest in before.items()
                if source in stored_hashes and stored_hashes[source] != digest
            )
            if changed_existing:
                examples = ", ".join(changed_existing[:3])
                raise ValueError(
                    "ingest() é aditivo e encontrou fonte existente alterada "
                    f"({examples}); use sync() para substituir conteúdo."
                )
            new_sources = sorted(source for source in before if source not in stored_hashes)
            if not new_sources:
                logger.info("Nenhuma fonte nova; o snapshot ativo foi mantido.")
                return 0
            spool, node_count, node_sources = self._spool_nodes(
                self._ingest_sync_files(new_sources)
            )
            after = self._inventory_sync_sources(path)
            if before != after:
                spool.close()
                raise RuntimeError(
                    "As fontes mudaram durante ingest(); nenhuma alteração foi publicada."
                )
            statuses = getattr(self.ingestor, "last_ingest_status", {}) or {}
            failed = sorted(source for source in new_sources if statuses.get(source) != "ok")
            if failed:
                spool.close()
                raise RuntimeError(
                    f"Ingestão abortada: {len(failed)} fonte(s) nova(s) falharam; "
                    "nenhuma alteração foi publicada."
                )
            source_file_hashes = dict(stored_hashes)
            source_file_hashes.update({source: after[source] for source in new_sources})
        else:
            source = self._normalized_source(path)
            before_hash = self._file_sha256(path)
            old_hash = stored_hashes.get(source)
            indexed_sources = {
                self._normalized_source(self._source_file_path(data.get("source", "")))
                for _, data in self.engine.graph.nodes(data=True)
                if data.get("source")
            }
            if old_hash == before_hash:
                logger.info("A fonte não mudou; o snapshot ativo foi mantido.")
                return 0
            if old_hash is not None or source in indexed_sources:
                raise ValueError("ingest() não substitui fontes existentes; use sync() para atualizar.")
            spool, node_count, _sources = self._spool_nodes(
                self._ingest_sync_files([os.path.abspath(path)])
            )
            after_hash = self._file_sha256(path)
            if before_hash != after_hash:
                spool.close()
                raise RuntimeError("A fonte mudou durante ingest(); nenhuma alteração foi publicada.")
            statuses = getattr(self.ingestor, "last_ingest_status", {}) or {}
            if statuses.get(os.path.abspath(path)) != "ok":
                spool.close()
                raise RuntimeError("A fonte falhou ao ser processada; nenhuma alteração foi publicada.")
            source_file_hashes = dict(stored_hashes)
            source_file_hashes[source] = after_hash

        try:
            if not node_count and not self.engine.graph.number_of_nodes():
                logger.info("Nenhum documento ingerido; o indice anterior foi preservado.")
                return 0
            published_count = self._publish_snapshot(
                self._iter_spooled_nodes(spool), self.engine,
                source_file_hashes=source_file_hashes,
            )
        finally:
            spool.close()
        logger.info(
            "Resumos L3 LLM serao gerados em background ao iniciar "
            "'basinrag chat' ou 'basinrag serve'. L3 extractivo ja esta no indice."
        )
        return published_count

    def sync(self, path: str) -> int:
        """Synchronize supported sources under a path, replacing changed source chunks."""
        self._ensure_write_base()
        with self.persistence.write_transaction(expected_build_id=self.engine.build_id):
            return self._sync_locked(path)

    def _sync_locked(self, path: str) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Caminho de sincronização não encontrado: {path}")
        old_sources = {data.get("source", "") for _, data in self.engine.graph.nodes(data=True)}
        spool = None
        try:
            if os.path.isdir(path):
                if os.path.islink(path):
                    raise ValueError("sync não segue diretórios simbólicos")
                root = self._normalized_source(path)
                before = self._inventory_sync_sources(path)
                stored_hashes = (self.engine.index_metadata or {}).get("source_file_hashes") or {}
                changed_paths = [
                    os.path.abspath(source_key)
                    for source_key, source_hash in before.items()
                    if stored_hashes.get(source_key) != source_hash
                ]
                spool, node_count, node_sources = self._spool_nodes(
                    self._ingest_sync_files(changed_paths)
                )
                after = self._inventory_sync_sources(path)
                if before != after:
                    raise RuntimeError(
                        "As fontes mudaram durante sync; execute novamente para obter um snapshot consistente."
                    )

                statuses = getattr(self.ingestor, "last_ingest_status", {}) or {}
                successful = {
                    self._normalized_source(source)
                    for source, status in statuses.items()
                    if status == "ok"
                }
                successful.update(
                    self._normalized_source(self._source_file_path(source))
                    for source in node_sources
                )

                def in_scope(source):
                    source_key = self._normalized_source(self._source_file_path(source))
                    try:
                        return os.path.commonpath([root, source_key]) == root
                    except ValueError:
                        return False

                present_sources = set(before)
                source_file_hashes = {
                    key: value for key, value in stored_hashes.items()
                    if not in_scope(key)
                }
                changed_keys = {self._normalized_source(source) for source in changed_paths}
                for source_key, digest in after.items():
                    if source_key not in changed_keys or source_key in successful:
                        source_file_hashes[source_key] = digest
                    elif source_key in stored_hashes:
                        # Keep the old digest so a failed parse is retried on the next sync.
                        source_file_hashes[source_key] = stored_hashes[source_key]
                replace_sources = {
                    source for source in old_sources
                    if source and in_scope(source)
                    and (
                        self._normalized_source(self._source_file_path(source)) in successful
                        or self._normalized_source(self._source_file_path(source)) not in present_sources
                    )
                }
            else:
                exact_source = self._normalized_source(path)
                before_hash = self._file_sha256(path)
                stored_hashes = (self.engine.index_metadata or {}).get("source_file_hashes") or {}
                if stored_hashes.get(exact_source) == before_hash:
                    logger.info("A fonte não mudou; o snapshot ativo foi mantido.")
                    return 0
                spool, node_count, node_sources = self._spool_nodes(
                    self._ingest_sync_files([os.path.abspath(path)])
                )
                after_hash = self._file_sha256(path)
                if after_hash != before_hash:
                    raise RuntimeError(
                        "A fonte mudou durante sync; execute novamente para obter um snapshot consistente."
                    )
                statuses = getattr(self.ingestor, "last_ingest_status", {}) or {}
                successful = {
                    self._normalized_source(source)
                    for source, status in statuses.items()
                    if status == "ok"
                }
                successful.update(
                    self._normalized_source(self._source_file_path(source))
                    for source in node_sources
                )
                replace_sources = {
                    source for source in old_sources
                    if source
                    and self._normalized_source(self._source_file_path(source)) == exact_source
                    and exact_source in successful
                }
                source_file_hashes = dict(stored_hashes)
                if exact_source in successful:
                    source_file_hashes[exact_source] = after_hash

            if not node_count and not replace_sources:
                logger.info("Nenhuma fonte alterada ou removida; o snapshot ativo foi mantido.")
                return 0
            return self._publish_snapshot(
                self._iter_spooled_nodes(spool),
                self.engine,
                replace_sources=replace_sources,
                source_file_hashes=source_file_hashes,
            )
        finally:
            if spool is not None:
                spool.close()

    def _ingest_sync_files(self, filepaths):
        if not filepaths:
            return
        iter_ingest_files = getattr(self.ingestor, "iter_ingest_files", None)
        if callable(iter_ingest_files):
            yield from iter_ingest_files(filepaths)
            return
        ingest_files = getattr(self.ingestor, "ingest_files", None)
        if callable(ingest_files):
            yield from ingest_files(filepaths)
            return
        ingest_one = getattr(self.ingestor, "ingest", None)
        if callable(ingest_one):
            nodes = []
            for filepath in filepaths:
                nodes.extend(ingest_one(filepath))
            yield from nodes
            return
        yield from self._load_nodes(filepaths[0])

    def reindex(self, path: str) -> int:
        """Build a validated snapshot in a new, empty destination; never mutate a legacy root."""
        destination = os.path.abspath(self.config.storage_dir)
        if os.path.exists(destination) and os.listdir(destination):
            raise FileExistsError(
                f"Destino de reindexação não está vazio: {destination}. "
                "Escolha outro storage_dir; o índice existente será preservado."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Origem de reindexação não encontrada: {path}")
        self.engine.build_id = ""
        with self.persistence.write_transaction(expected_build_id=self.engine.build_id):
            return self._reindex_locked(path)

    def _reindex_locked(self, path: str) -> int:
        if hasattr(self.ingestor, "last_ingest_status"):
            self.ingestor.last_ingest_status = {}
        spool, node_count, _sources = self._spool_nodes(self._iter_load_nodes(path))
        try:
            failed_sources = [
                source
                for source, status in (getattr(self.ingestor, "last_ingest_status", {}) or {}).items()
                if status == "error"
            ]
            if failed_sources:
                examples = ", ".join(sorted(failed_sources)[:3])
                raise RuntimeError(
                    f"Reindex abortado: falha ao processar {len(failed_sources)} fonte(s) ({examples}). "
                    "O snapshot ativo foi preservado."
                )
            if not node_count:
                logger.info("Nenhum documento ingerido; o indice anterior foi preservado.")
                return 0
            replace_sources = {
                data.get("source", "") for _, data in self.engine.graph.nodes(data=True)
            }
            return self._publish_snapshot(
                self._iter_spooled_nodes(spool),
                self.engine,
                replace_sources=replace_sources,
                require_ingest_success=True,
            )
        finally:
            spool.close()

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._inference_semaphore is None:
            self._inference_semaphore = asyncio.Semaphore(2)
        return self._inference_semaphore

    def query(
        self, question: str, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> list:
        retriever, engine = self._capture_retriever()
        packet = retriever.brief(question, search_type=search_type, top_k=top_k)
        from langchain_core.documents import Document
        docs = []
        for i, text in enumerate(packet.texts_for_rerank()[: (top_k or retriever.top_k)]):
            meta = {}
            if i < len(packet.node_ids):
                nid = packet.node_ids[i]
                if nid in engine.graph:
                    node_data = engine.graph.nodes[nid]
                    meta = dict(node_data.get("metadata") or {})
            meta["node_id"] = nid
            meta["source"] = node_data.get("source", "")
            meta["source_label"] = os.path.basename(str(meta["source"])) if meta["source"] else None
            meta["doc_id"] = meta.get("doc_id") or node_data.get("source", "")
            meta["retrieval_role"] = "seed" if i < len(packet.hubs) else "context"
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    async def aquery(
        self, question: str, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> list:
        async with self._get_semaphore():
            return await asyncio.to_thread(self.query, question, search_type, top_k)

    def brief(
        self, question: str, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> BriefingPacket:
        retriever, _engine = self._capture_retriever()
        return retriever.brief(question, search_type=search_type, top_k=top_k)

    async def abrief(
        self, question: str, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> BriefingPacket:
        retriever, _engine = self._capture_retriever()
        async with self._get_semaphore():
            return await asyncio.to_thread(retriever.brief, question, search_type, top_k)

    def _ensure_retriever(
        self, search_type: Optional[SearchType] = None, top_k: Optional[int] = None
    ) -> None:
        if self.retriever is None:
            self.retriever = BasinRAGRetriever(
                engine=self.engine,
                encoder=self.ingestor.encoder,
                search_type=self.config.search_type,
                top_k=5,
                reranker_model=self.config.reranker_model,
                query_prompt=self.config.query_prompt,
                use_rerank=self.config.use_rerank,
                ranking_mode=self.config.ranking_mode,
            )
        if search_type is not None or top_k is not None:
            self.retriever.configure(
                search_type=search_type if search_type is not None else None,
                top_k=top_k if top_k is not None else None,
            )

    def _capture_retriever(self) -> tuple[BasinRAGRetriever, BasinTopologyEngine]:
        """Capture a consistent engine/retriever pair across an atomic index swap."""
        with self._snapshot_lock:
            self._ensure_retriever()
            assert self.retriever is not None
            return self.retriever, self.engine

    def validate_query_dependencies(self) -> None:
        if not self._loaded:
            raise RuntimeError("Nenhum snapshot válido foi carregado")
        retriever, _engine = self._capture_retriever()
        if self.config.use_rerank and not index_is_flat(self.engine):
            assert retriever._reranker is not None
            retriever._reranker._load_model()

    async def start_background_summarizer(self, verbose: bool = False):
        if not self.config.enable_background_l3:
            return
        await self.summarizer.summarize_missing_background(
            persistence=self.persistence,
            interval=1.0,
            verbose=verbose,
        )

    async def chat(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        search_type: Optional[SearchType] = None,
        top_k: Optional[int] = None,
    ):
        packet = await self.abrief(question, search_type=search_type, top_k=top_k)
        async for token in self.answer_stream(packet, question, system_prompt):
            yield token

    async def answer_stream(self, packet, question: str, system_prompt: Optional[str] = None):
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
        sys_prompt, user_prompt = self._budgeted_rag_prompts(packet, question, system_prompt)
        async for token in self._ensure_llm().chat_stream(sys_prompt, user_prompt):
            yield token

    def _budgeted_rag_prompts(self, packet, question: str, system_prompt: Optional[str] = None):
        """Fit instructions, question, evidence, headers, and output reserve together."""
        llm = self._ensure_llm()
        input_budget = self.config.context_window_tokens - self.config.generation_reserve_tokens
        selected: dict[str, list[str]] = {"Hubs": [], "Neighbors": [], "Satellites": []}

        def context_text():
            blocks = []
            for label in ("Hubs", "Neighbors", "Satellites"):
                values = selected[label]
                if values:
                    blocks.append(f"[{label}]\n" + "\n\n".join(values))
            return "\n\n".join(blocks)

        def prompts():
            return build_rag_prompts(context_text(), question, system_prompt)

        sys_prompt, user_prompt = prompts()
        if llm.count_tokens(sys_prompt + "\n" + user_prompt) > input_budget:
            raise ValueError("Instruções e pergunta excedem o orçamento de entrada do modelo.")

        evidence: dict[str, list[str]] = {"Hubs": [], "Neighbors": [], "Satellites": []}
        for index, text in enumerate(packet.hubs):
            reference = packet.node_ids[index] if index < len(packet.node_ids) else ""
            evidence["Hubs"].append(f"[ref:{reference}] {text}" if reference else text)
        for index, text in enumerate(packet.neighbors):
            node_index = len(packet.hubs) + index
            reference = packet.node_ids[node_index] if node_index < len(packet.node_ids) else ""
            evidence["Neighbors"].append(f"[ref:{reference}] {text}" if reference else text)
        evidence["Satellites"] = packet.satellites[:4]
        for label in ("Hubs", "Neighbors", "Satellites"):
            for value in evidence[label]:
                item = str(value or "").strip()
                if not item:
                    continue
                if label == "Satellites":
                    item = item[:400].rstrip()
                selected[label].append(item)
                trial_sys, trial_user = prompts()
                if llm.count_tokens(trial_sys + "\n" + trial_user) > input_budget:
                    selected[label].pop()
                    break
                sys_prompt, user_prompt = trial_sys, trial_user
        return sys_prompt, user_prompt

