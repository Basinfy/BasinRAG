from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import sys
from pathlib import Path
import numpy as np
from ..logging_config import setup_logging

logger = setup_logging()

from ..core.ids import make_node_id
from .condensation import node_layers

_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}
_SKIP_DIRS = {".git", ".basinrag", "__pycache__", "node_modules", ".venv", "venv"}
# LlamaIndex sentence-window: retrieve 1–2 sentences, hydrate ±3 at brief time.
SENTENCE_WINDOW_RADIUS = 3
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentence_leaves(
    text: str,
    max_chars: int = 256,
    overlap_chars: int = 32,
    max_sentences: int = 2,
) -> List[str]:
    """Pack 1–2 sentences (or ~128 tokens) into retrieval leaves.

    Overflow longer than ``max_chars`` falls back to the character splitter so
    a methods paragraph still becomes multiple children. Long-doc indexing
    uses this; SciFact stays one node per document.
    """
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    overlap_chars = max(0, int(overlap_chars))
    if max_chars > 1:
        overlap_chars = min(overlap_chars, max_chars - 1)
    max_sentences = max(1, int(max_sentences))

    raw_parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]
    if not raw_parts:
        raw_parts = [text]

    units: List[str] = []
    overflow = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )
    for part in raw_parts:
        if len(part) <= max_chars:
            units.append(part)
        else:
            units.extend(chunk.strip() for chunk in overflow.split_text(part) if chunk.strip())
    if not units:
        return []

    leaves: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def _flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        leaf = " ".join(buf)
        if not leaves or leaves[-1] != leaf:
            leaves.append(leaf)
        if overlap_chars <= 0:
            buf = []
            buf_len = 0
            return
        keep: List[str] = []
        keep_len = 0
        for item in reversed(buf):
            add = len(item) + (1 if keep else 0)
            if keep and keep_len + add > overlap_chars:
                break
            keep.append(item)
            keep_len += add
        buf = list(reversed(keep))
        buf_len = keep_len

    for unit in units:
        extra = len(unit) + (1 if buf else 0)
        if buf and (len(buf) >= max_sentences or buf_len + extra > max_chars):
            _flush()
            extra = len(unit) + (1 if buf else 0)
            if buf and buf_len + extra > max_chars:
                buf = []
                buf_len = 0
                extra = len(unit)
        buf.append(unit)
        buf_len += extra if buf_len else len(unit)
    if buf:
        leaf = " ".join(buf)
        if not leaves or leaves[-1] != leaf:
            leaves.append(leaf)
    return leaves


def _legacy_chunk_length(text: str) -> int:
    """Preserve the legacy character-based meaning of chunk_size/overlap."""
    return len(text)


def _single_sequence(value: Any, *, offset_mapping: bool = False) -> List[Any]:
    """Normalize a tokenizer result for one input string to a flat list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        # Offset maps are already a list of [start, end] pairs for a single
        # string; a batched offset map has one additional nesting level.
        is_batched_offsets = (
            offset_mapping
            and value[0]
            and isinstance(value[0][0], (list, tuple))
        )
        if not offset_mapping or is_batched_offsets:
            value = value[0]
    return list(value or [])


class TokenAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Keep the legacy splitter API and optionally split/cap by encoder tokens."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Any = None,
        max_seq_length: Optional[int] = None,
        chunk_size_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
    ):
        if chunk_size_tokens is not None and int(chunk_size_tokens) <= 0:
            raise ValueError("chunk_size_tokens deve ser maior que zero")
        if chunk_overlap_tokens is not None:
            if chunk_size_tokens is None:
                raise ValueError("chunk_overlap_tokens exige chunk_size_tokens")
            if int(chunk_overlap_tokens) < 0 or int(chunk_overlap_tokens) >= int(chunk_size_tokens):
                raise ValueError("chunk_overlap_tokens deve ser >= 0 e menor que chunk_size_tokens")

        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.chunk_size_tokens = int(chunk_size_tokens) if chunk_size_tokens is not None else None
        self.chunk_overlap_tokens = int(chunk_overlap_tokens or 0)
        special_tokens = self._special_tokens_count()
        self.max_content_tokens = (
            max(1, self.max_seq_length - special_tokens) if self.max_seq_length else None
        )

        super().__init__(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            length_function=_legacy_chunk_length,
            # The empty separator lets the legacy path split CJK and other
            # text that has no whitespace or sentence punctuation.
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        )

    def _special_tokens_count(self) -> int:
        if self.tokenizer is None:
            return 2
        count = getattr(self.tokenizer, "num_special_tokens_to_add", None)
        if callable(count):
            try:
                return max(0, int(count(pair=False)))
            except (TypeError, ValueError):
                pass
        return 2

    def _tokenizer_result(self, text: str, *, offsets: bool = False):
        if self.tokenizer is None:
            return None, None
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=offsets,
            )
            ids = _single_sequence(encoded.get("input_ids"))
            raw_offsets = encoded.get("offset_mapping") if offsets else None
            token_offsets = (
                _single_sequence(raw_offsets, offset_mapping=True)
                if raw_offsets is not None
                else None
            )
            return ids, token_offsets
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            if offsets:
                return None, None
            try:
                return _single_sequence(self.tokenizer.encode(text, add_special_tokens=False)), None
            except (TypeError, ValueError, NotImplementedError, AttributeError):
                return None, None

    @staticmethod
    def _fallback_token_count(text: str) -> int:
        # SentenceTransformer encoders expose a tokenizer. This conservative
        # fallback keeps fake/custom encoders and unspaced CJK bounded too.
        return len(text.encode("utf-8")) if text else 0

    def token_count(self, text: str) -> int:
        ids, _ = self._tokenizer_result(text)
        return len(ids) if ids is not None else self._fallback_token_count(text)

    def _split_with_offsets(self, text: str, limit: int, overlap: int) -> Optional[List[str]]:
        ids, offsets = self._tokenizer_result(text, offsets=True)
        if ids is None or offsets is None or len(ids) != len(offsets):
            return None
        if not ids:
            return [text] if text else []

        chunks: List[str] = []
        start_token = 0
        while start_token < len(ids):
            end_token = min(len(ids), start_token + limit)
            char_start = 0 if start_token == 0 else int(offsets[start_token][0])
            char_end = len(text) if end_token == len(ids) else int(offsets[end_token][0])
            if char_end <= char_start:
                return None
            chunk = text[char_start:char_end]
            if chunk:
                chunks.append(chunk)
            if end_token == len(ids):
                break
            start_token = max(start_token + 1, end_token - overlap)
        return chunks

    def _split_without_offsets(self, text: str, limit: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            low = start + 1
            high = len(text)
            best = low
            while low <= high:
                middle = (low + high) // 2
                if self.token_count(text[start:middle]) <= limit:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            end = best
            chunks.append(text[start:end])
            if end >= len(text):
                break

            next_start = end
            if overlap:
                low = start + 1
                high = end
                while low <= high:
                    middle = (low + high) // 2
                    if self.token_count(text[middle:end]) <= overlap:
                        next_start = middle
                        high = middle - 1
                    else:
                        low = middle + 1
            start = max(start + 1, next_start)
        return chunks

    def _split_by_tokens(self, text: str, limit: int, overlap: int) -> List[str]:
        if not text:
            return []
        limit = max(1, int(limit))
        overlap = min(max(0, int(overlap)), limit - 1)
        chunks = self._split_with_offsets(text, limit, overlap)
        if chunks is None:
            chunks = self._split_without_offsets(text, limit, overlap)
        return chunks

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        if self.chunk_size_tokens is not None:
            size = self.chunk_size_tokens
            if self.max_content_tokens is not None:
                size = min(size, self.max_content_tokens)
            return self._split_by_tokens(text, size, self.chunk_overlap_tokens)

        chunks = super().split_text(text)
        if self.max_content_tokens is None:
            return chunks

        bounded: List[str] = []
        for chunk in chunks:
            if self.token_count(chunk) <= self.max_content_tokens:
                bounded.append(chunk)
            else:
                bounded.extend(self._split_by_tokens(chunk, self.max_content_tokens, 0))
        return bounded

    def split_documents(self, documents):
        from langchain_core.documents import Document

        split = []
        for document in documents:
            metadata = dict(document.metadata or {})
            split.extend(
                Document(page_content=text, metadata=metadata.copy())
                for text in self.split_text(document.page_content)
            )
        return split


def json_doc_source(filepath: str, index: int) -> str:
    return f"{os.path.abspath(filepath)}#{index}"


def _patch_torch_dtensor():
    try:
        import torch.distributed._tensor
        sys.modules.setdefault("torch.distributed.tensor", torch.distributed._tensor)
    except ImportError:
        pass


class BasinIngestor:
    """Load TXT/MD/PDF, chunk, embed. IDs are content-addressed (source+index+text)."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        revision: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        chunk_size_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        embedding_batch_size: int = 64,
        encoding: Optional[str] = None,
    ):
        _patch_torch_dtensor()
        from sentence_transformers import SentenceTransformer
        if revision:
            self.encoder = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)
        else:
            self.encoder = SentenceTransformer(model_name, trust_remote_code=False)
        self.model_revision = self._resolved_model_revision(revision)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if int(embedding_batch_size) <= 0:
            raise ValueError("embedding_batch_size deve ser maior que zero")
        self.embedding_batch_size = int(embedding_batch_size)
        self.encoding = encoding
        self.last_ingest_status: Dict[str, str] = {}
        self._directory_ingesting = False
        tokenizer = getattr(self.encoder, "tokenizer", None)
        max_seq_length = getattr(self.encoder, "max_seq_length", None)
        if tokenizer is None or max_seq_length is None:
            first_module = getattr(self.encoder, "_first_module", None)
            if callable(first_module):
                try:
                    module = first_module()
                    tokenizer = tokenizer or getattr(module, "tokenizer", None)
                    max_seq_length = max_seq_length or getattr(module, "max_seq_length", None)
                except Exception:
                    pass
        self.splitter = TokenAwareTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    def _resolved_model_revision(self, requested: Optional[str]) -> str:
        if requested:
            return requested
        modules = getattr(self.encoder, "_modules", {})
        for module in modules.values() if isinstance(modules, dict) else []:
            auto_model = getattr(module, "auto_model", None)
            config = getattr(auto_model, "config", None)
            commit_hash = getattr(config, "_commit_hash", None)
            if commit_hash:
                return str(commit_hash)
        model_path = str(getattr(self.encoder, "model_name_or_path", ""))
        marker = f"{os.sep}snapshots{os.sep}"
        if marker in model_path:
            return model_path.rsplit(marker, 1)[1].split(os.sep, 1)[0]
        return "unresolved"

    def _nodes_from_texts(
        self,
        texts: List[str],
        source: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        nodes = []
        n = len(texts)
        for start in range(0, len(texts), self.embedding_batch_size):
            batch_texts = texts[start : start + self.embedding_batch_size]
            batch_embeddings = np.asarray(
                self.encoder.encode(
                    batch_texts,
                    batch_size=self.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
            if batch_embeddings.shape[0] != len(batch_texts):
                raise ValueError(
                    "O encoder retornou quantidade de embeddings diferente da quantidade de textos."
                )
            for batch_offset, (txt, emb) in enumerate(zip(batch_texts, batch_embeddings)):
                i = start + batch_offset
                layers = node_layers(txt)
                meta = dict((metadata_list[i] if metadata_list else {}) or {})
                meta["doc_id"] = meta.get("doc_id") or source
                meta["role"] = "child"
                # Sentence-window parent: retrieve the leaf, hydrate ±3 at brief time.
                meta["parent_span"] = [
                    max(0, i - SENTENCE_WINDOW_RADIUS),
                    min(n - 1, i + SENTENCE_WINDOW_RADIUS),
                ]
                meta["parent_doc"] = source
                nodes.append({
                    "id": make_node_id(source, i, txt),
                    "text": txt,
                    "embedding": emb,
                    "source": source,
                    "chunk_index": i,
                    "l1": layers["l1"],
                    "l2": layers["l2"],
                    "metadata": meta,
                })
        return nodes

    def ingest(self, filepath: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in _TEXT_EXTS | _PDF_EXTS:
            logger.warning(f"Extensão não suportada. Ignorando arquivo: {filepath}")
            return []

        source_key = os.path.abspath(filepath)
        if not self._directory_ingesting:
            self.last_ingest_status = {}
        try:
            texts, metadata_list = self._load_chunks(filepath)
            nodes = self._nodes_from_texts(texts, source_key, metadata_list)
            self.last_ingest_status[source_key] = "ok"
            return nodes
        except Exception:
            self.last_ingest_status[source_key] = "error"
            logger.exception(f"Erro ao processar {filepath}")
            return []

    def _load_chunks(self, filepath: str) -> tuple:
        ext = os.path.splitext(filepath)[1].lower()
        docs = []
        if ext in _PDF_EXTS:
            from pypdf import PdfReader
            for page_number, page in enumerate(PdfReader(filepath).pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append((text, {"page": page_number}))
        else:
            raw = Path(filepath).read_bytes()
            if self.encoding:
                text = raw.decode(self.encoding, errors="strict")
            else:
                try:
                    text = raw.decode("utf-8-sig", errors="strict")
                except UnicodeDecodeError:
                    from charset_normalizer import from_bytes
                    match = from_bytes(raw).best()
                    if (
                        match is None
                        or not match.encoding
                        or float(getattr(match, "percent_chaos", 100.0)) > 20.0
                        or float(getattr(match, "percent_coherence", 0.0)) < 20.0
                    ):
                        raise UnicodeError(
                            f"Não foi possível detectar com segurança o encoding de {filepath}; "
                            "configure BASINRAG_SOURCE_ENCODING ou encoding explicitamente."
                        )
                    logger.warning(
                        "Encoding autodetectado para %s: %s (coerência %.1f%%)",
                        filepath,
                        match.encoding,
                        float(getattr(match, "percent_coherence", 0.0)),
                    )
                    text = str(match)
            docs.append((text, {}))

        texts = []
        metadata_list = []
        for raw_text, doc_meta in docs:
            for chunk in self.splitter.split_text(raw_text):
                texts.append(chunk)
                metadata_list.append(dict(doc_meta))
        return texts, metadata_list

    def load_text(self, filepath: str) -> List[str]:
        texts, _ = self._load_chunks(filepath)
        return texts

    def iter_ingest(self, filepath: str):
        """Yield one file's chunks while keeping the list-returning API intact."""
        yield from self.ingest(filepath)

    def iter_ingest_files(self, filepaths):
        """Ingest a file set one source at a time instead of retaining the corpus list."""
        self.last_ingest_status = {}
        self._directory_ingesting = True
        try:
            for filepath in filepaths:
                yield from self.ingest(filepath)
        finally:
            self._directory_ingesting = False

    def ingest_files(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """List-returning compatibility wrapper; streaming callers use iter_ingest_files."""
        return list(self.iter_ingest_files(filepaths))

    def ingest_json_dataset(self, filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List-returning compatibility wrapper for JSON dataset ingestion."""
        return list(self.iter_ingest_json_dataset(filepath, limit=limit))

    def iter_ingest_json_dataset(self, filepath: str, limit: Optional[int] = None):
        """Stream encoded JSON dataset chunks one document at a time."""
        from datasets import load_dataset

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

        dataset = load_dataset("json", data_files=filepath, split="train", streaming=True)
        doc_index = 0
        record_count = 0
        for item in dataset:
            record_count += 1
            documents = item.get("documents", [])
            if isinstance(documents, str):
                documents = [documents]
            for document_text in documents:
                if not document_text or not str(document_text).strip():
                    continue
                chunks = self.splitter.split_text(str(document_text))
                source = json_doc_source(filepath, doc_index)
                metadata = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                }
                metadata_list = [dict(metadata) for _ in chunks]
                yield from self._nodes_from_texts(chunks, source, metadata_list)
                doc_index += 1
            if limit and record_count >= limit:
                break

    def iter_ingest_directory(self, dir_path: str):
        """Yield a directory's chunks one file at a time."""
        if not os.path.exists(dir_path):
            return

        def supported_files():
            for root, dirs, files in os.walk(dir_path, followlinks=False):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
                for filename in files:
                    filepath = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in _TEXT_EXTS | _PDF_EXTS:
                        logger.info(f"Ingerindo arquivo: {filepath}...")
                        yield filepath

        yield from self.iter_ingest_files(supported_files())

    def ingest_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """List-returning compatibility wrapper; BasinRAG uses iter_ingest_directory."""
        if not os.path.exists(dir_path):
            return []
        return list(self.iter_ingest_directory(dir_path))
