from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
from ..logging_config import setup_logging

logger = setup_logging()

from ..core.ids import make_node_id
from .condensation import node_layers

_TEXT_EXTS = {".txt", ".md", ".markdown"}
_PDF_EXTS = {".pdf"}
_SKIP_DIRS = {".git", ".basinrag", "__pycache__", "node_modules", ".venv", "venv"}


def json_doc_source(filepath: str, index: int) -> str:
    return f"{os.path.abspath(filepath)}#{index}"


def group_json_documents(filepath: str, records: List[Dict[str, Any]], splitter) -> List[tuple]:
    """One source per JSON document; each document is split like TXT/PDF."""
    groups = []
    doc_i = 0
    for item in records:
        docs = item.get("documents", [])
        if isinstance(docs, str):
            docs = [docs]
        for doc_text in docs:
            if not doc_text or not str(doc_text).strip():
                continue
            chunks = splitter.split_text(str(doc_text))
            meta = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            metadata_list = [dict(meta) for _ in chunks]
            groups.append((json_doc_source(filepath, doc_i), chunks, metadata_list))
            doc_i += 1
    return groups


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
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        _patch_torch_dtensor()
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def _nodes_from_texts(
        self,
        texts: List[str],
        source: str,
        metadata_list: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        nodes = []
        for i, (txt, emb) in enumerate(zip(texts, embeddings)):
            layers = node_layers(txt)
            meta = (metadata_list[i] if metadata_list else {}) or {}
            meta["doc_id"] = meta.get("doc_id") or source
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

        try:
            texts, metadata_list = self._load_chunks(filepath)
            return self._nodes_from_texts(texts, os.path.abspath(filepath), metadata_list)
        except Exception:
            logger.exception(f"Erro ao processar {filepath}")
            return []

    def _load_chunks(self, filepath: str) -> tuple:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
        ext = os.path.splitext(filepath)[1].lower()
        if ext in _PDF_EXTS:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        else:
            docs = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    loader = TextLoader(filepath, encoding=enc)
                    docs = loader.load()
                    break
                except Exception:
                    continue
            if docs is None:
                loader = TextLoader(filepath, autodetect_encoding=True)
                docs = loader.load()

        chunks = self.splitter.split_documents(docs)
        texts = []
        metadata_list = []
        for c in chunks:
            texts.append(c.page_content)
            meta = dict(c.metadata or {})
            page = meta.get("page")
            if page is None:
                page = meta.get("page_label")
            out = {}
            if page is not None:
                try:
                    out["page"] = int(page)
                except (TypeError, ValueError):
                    out["page"] = page
            metadata_list.append(out)
        return texts, metadata_list

    def load_text(self, filepath: str) -> List[str]:
        texts, _ = self._load_chunks(filepath)
        return texts

    def ingest_json_dataset(self, filepath: str, limit: int = None) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

        dataset = load_dataset("json", data_files=filepath, split="train")
        records = []
        count = 0
        for item in dataset:
            records.append(item)
            count += 1
            if limit and count >= limit:
                break
        groups = group_json_documents(filepath, records, self.splitter)
        all_nodes: List[Dict[str, Any]] = []
        for source, chunks, metadata_list in groups:
            all_nodes.extend(self._nodes_from_texts(chunks, source, metadata_list))
        return all_nodes

    def ingest_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(dir_path):
            return []

        all_nodes: List[Dict[str, Any]] = []
        for root, dirs, files in os.walk(dir_path, followlinks=False):

            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for filename in files:
                filepath = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext not in _TEXT_EXTS | _PDF_EXTS:
                    continue
                logger.info(f"Ingerindo arquivo: {filepath}...")
                try:
                    all_nodes.extend(self.ingest(filepath))
                except Exception:
                    logger.exception(f"Erro ao ler {filename}")
        return all_nodes
