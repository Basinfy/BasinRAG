import hashlib
import os
import json
import re
import shutil
import sqlite3
from pathlib import Path
import time
import uuid
import zipfile
import numpy as np
import networkx as nx
from contextlib import contextmanager
from typing import Any, Dict, Iterator
from filelock import FileLock
from ..logging_config import setup_logging

logger = setup_logging()


def dump_graph(graph) -> Dict[str, Any]:
    try:
        return nx.node_link_data(graph, edges="links")
    except TypeError:
        return nx.node_link_data(graph)


def snapshot_without_embeddings(graph):
    """New node-attr dicts so stripping embeddings cannot mutate the live graph.

    NetworkX Graph.copy() is shallow: deleting attrs on the copy also wipes RAM.
    """
    clone = graph.__class__()
    for node_id, attrs in graph.nodes(data=True):
        clone.add_node(node_id, **{k: v for k, v in attrs.items() if k != "embedding"})
    for source, target, attrs in graph.edges(data=True):
        clone.add_edge(source, target, **dict(attrs))
    return clone


def load_graph(data: Dict[str, Any]):
    payload = dict(data)
    payload.pop("_meta", None)
    edges_key = "edges" if "edges" in payload else "links"
    try:
        return nx.node_link_graph(payload, edges=edges_key)
    except TypeError:
        return nx.node_link_graph(payload)


def basin_filename(basin_id: str) -> str:
    digest = hashlib.sha256(str(basin_id).encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_EXPECTED_UNSET = object()
_BUILD_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class IndexRebuildRequired(RuntimeError):
    """Raised when a snapshot predates the current major-release format."""


class BasinPersistence:
    """Split persistence: metadata JSON, embeddings NPZ, BM25 JSON, atomic current.json pointer."""

    def __init__(self, storage_dir: str = ".basinrag-v3"):
        self.storage_dir = storage_dir
        self._write_lock = FileLock(
            os.path.join(self.storage_dir, "index.write.lock"),
            timeout=90,
        )

    def _ensure_storage(self) -> None:
        os.makedirs(os.path.join(self.storage_dir, "builds"), exist_ok=True)

    def active_dir(self) -> str:
        pointer = os.path.join(self.storage_dir, "current.json")
        if os.path.exists(pointer):
            try:
                with open(pointer, "r", encoding="utf-8") as f:
                    data = json.load(f)
                build_id = data.get("build_id") or data.get("buildId") or ""
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Ponteiro current.json inválido: {exc}") from exc
            if not isinstance(build_id, str) or not _BUILD_ID_PATTERN.fullmatch(build_id):
                raise RuntimeError("Ponteiro current.json contém um build_id inválido")
            candidate = os.path.realpath(os.path.join(self.storage_dir, "builds", build_id))
            builds_root = os.path.realpath(os.path.join(self.storage_dir, "builds"))
            if os.path.commonpath([builds_root, candidate]) != builds_root or not os.path.isdir(candidate):
                raise RuntimeError(f"Snapshot ativo ausente ou fora da pasta de builds: {build_id}")
            return candidate
        return self.storage_dir

    def current_build_id(self) -> str:
        """Return the active generation id, including legacy root-format indexes."""
        pointer = os.path.join(self.storage_dir, "current.json")
        if os.path.exists(pointer):
            root = self.active_dir()
            with open(pointer, "r", encoding="utf-8") as f:
                payload = json.load(f)
            build_id = payload.get("build_id") or payload.get("buildId") or ""
            meta_path = os.path.join(root, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (meta.get("buildId") or "") != build_id:
                    raise RuntimeError("O build_id do ponteiro não corresponde ao manifesto do snapshot")
            return build_id

        meta_path = os.path.join(self.storage_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return str(meta.get("buildId") or "")
            except (OSError, json.JSONDecodeError):
                raise RuntimeError("Manifesto do índice legado inválido")
        return ""

    @contextmanager
    def write_transaction(self, expected_build_id=_EXPECTED_UNSET) -> Iterator[None]:
        """Serialize writers and reject a build created from an obsolete snapshot."""
        self._ensure_storage()
        with self._write_lock:
            current = self.current_build_id()
            if expected_build_id is not _EXPECTED_UNSET and current != expected_build_id:
                raise RuntimeError(
                    "O índice ativo mudou durante a construção "
                    f"(esperado={expected_build_id!r}, atual={current!r}); recarregue e tente novamente."
                )
            yield

    @property
    def basins_dir(self) -> str:
        return os.path.join(self.active_dir(), "basins")

    def save_topology(self, engine) -> bool:
        if not engine.graph:
            logger.error("Snapshot vazio recusado; nenhum índice será publicado")
            return False
        self._ensure_storage()
        with self._write_lock:
            expected_build_id = getattr(engine, "build_id", "") or ""
            try:
                current_build_id = self.current_build_id()
            except (OSError, RuntimeError, ValueError):
                logger.exception("Não foi possível validar o snapshot ativo antes de salvar")
                return False
            if current_build_id != expected_build_id:
                logger.warning(
                    "Save recusado: engine obsoleto (esperado=%r, ativo=%r)",
                    expected_build_id,
                    current_build_id,
                )
                return False
            return self._save_topology_inner(engine)

    def _save_topology_inner(self, engine) -> bool:
        logger.info(f"Salvando indice em {self.storage_dir}...")
        build_id = uuid.uuid4().hex
        build_dir = os.path.join(self.storage_dir, "builds", build_id)
        previous_engine_build_id = getattr(engine, "build_id", "")
        published = False
        try:
            previous_build_id = self.current_build_id()
            os.makedirs(os.path.join(build_dir, "basins"), exist_ok=True)

            nodes = list(engine.graph.nodes(data=True))
            node_ids = [n[0] for n in nodes]
            embeddings = np.array(
                [n[1].get("embedding", np.zeros(1, dtype=np.float32)) for n in nodes],
                dtype=np.float32,
            )
            if not nodes:
                embeddings = np.empty((0, 0), dtype=np.float32)
            if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
                raise ValueError("Snapshot contém embeddings inválidos ou não finitos")
            with open(os.path.join(build_dir, "node_ids.json"), "w", encoding="utf-8") as f:
                json.dump(node_ids, f)
            np.savez_compressed(
                os.path.join(build_dir, "embeddings.npz"),
                vectors=embeddings,
            )

            graph_data = dump_graph(snapshot_without_embeddings(engine.graph))
            with open(os.path.join(build_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False)

            index_metadata = dict(getattr(engine, "index_metadata", {}) or {})
            if not index_metadata:
                # Direct engine persistence is useful for core consumers and
                # tests. Production builds set exact provenance in BasinRAG.
                index_metadata = {
                    "format_version": 3,
                    "encoder_model": getattr(engine, "encoder_model", "") or "unconfigured",
                    "encoder_revision": "unresolved",
                    "tokenizer": "unconfigured",
                    "tokenizer_revision": "unresolved",
                    "reranker_revision": "unresolved",
                    "chunking_mode": "characters",
                    "chunk_policy_version": 1,
                    "chunk_size": 512,
                    "chunk_overlap": 0,
                    "chunk_size_tokens": None,
                    "chunk_overlap_tokens": None,
                    "sources": {},
                    "source_file_hashes": {},
                }
                engine.index_metadata = index_metadata
            engine.index_schema_version = 3
            meta = {
                "buildId": build_id,
                "index_schema_version": 3,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "section_size": getattr(engine, "section_size", 20),
                "encoder_model": getattr(engine, "encoder_model", ""),
                "node_count": len(node_ids),
                "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
                "index_metadata": index_metadata,
            }

            if hasattr(engine.successor, "backup_to"):
                engine.successor.backup_to(os.path.join(build_dir, "successor.db"))
            else:
                from .kv_store import DiskKVStore
                store = DiskKVStore(os.path.join(build_dir, "successor.db"), "successor")
                try:
                    store.set_many(dict(engine.successor.items()))
                finally:
                    store.close()

            if hasattr(engine.attractor_of, "backup_to"):
                engine.attractor_of.backup_to(os.path.join(build_dir, "attractor.db"))
            else:
                from .kv_store import DiskKVStore
                store = DiskKVStore(os.path.join(build_dir, "attractor.db"), "attractor_of")
                try:
                    store.set_many(dict(engine.attractor_of.items()))
                finally:
                    store.close()

            if hasattr(engine, "close_stores"):
                engine.close_stores()

            basins_tmp = os.path.join(build_dir, "basins")
            for basin_id, basin in engine.basins.items():
                basin_data = dump_graph(snapshot_without_embeddings(basin.rho_tree))
                basin_data["_meta"] = {
                    "basin_id": basin_id,
                    "source": getattr(basin, "source", ""),
                    "cohesion": getattr(basin, "cohesion", 1.0),
                }
                path = os.path.join(basins_tmp, basin_filename(basin_id))
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(basin_data, f, ensure_ascii=False)

            if getattr(engine, "bm25", None) is None:
                from ..indexer.bm25 import BM25Index
                engine.bm25 = BM25Index()
                engine.bm25.build(
                    node_ids,
                    [str(data.get("text", "")) for _, data in nodes],
                )
            engine.bm25.save(os.path.join(build_dir, "bm25.json"), build_id=build_id)

            artifact_paths = [
                "node_ids.json", "embeddings.npz", "graph.json",
                "successor.db", "attractor.db", "bm25.json",
            ] + [
                os.path.join("basins", name)
                for name in sorted(os.listdir(basins_tmp))
                if name.endswith(".json")
            ]
            meta["artifact_sha256"] = {
                name.replace(os.sep, "/"): self._sha256_file(os.path.join(build_dir, name))
                for name in artifact_paths
            }
            with open(os.path.join(build_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, sort_keys=True)

            self._validate_build(build_dir, node_ids, embeddings.shape, engine)
            if hasattr(engine, "reopen_stores"):
                engine.reopen_stores(build_dir, read_only=True)

            _atomic_write_json(
                os.path.join(self.storage_dir, "current.json"),
                {"build_id": build_id},
            )
            published = True
            engine.build_id = build_id
            self._prune_builds(keep={build_id, previous_build_id})
            return True
        except (IOError, OSError, ValueError, RuntimeError):
            logger.exception("Erro de I/O ao salvar")
            if published:
                engine.build_id = build_id
                if hasattr(engine, "reopen_stores"):
                    try:
                        engine.reopen_stores(build_dir, read_only=True)
                    except Exception:
                        logger.warning("Snapshot foi publicado, mas a reabertura das stores falhou", exc_info=True)
                return True
            shutil.rmtree(build_dir, ignore_errors=True)
            engine.build_id = previous_engine_build_id
            if hasattr(engine, "reopen_stores"):
                try:
                    active = self.active_dir()
                    if os.path.isfile(os.path.join(active, "successor.db")):
                        engine.reopen_stores(active, read_only=True)
                except Exception:
                    pass
            return False
        except Exception:
            logger.exception("Erro inesperado ao salvar topologia")
            if published:
                engine.build_id = build_id
                if hasattr(engine, "reopen_stores"):
                    try:
                        engine.reopen_stores(build_dir, read_only=True)
                    except Exception:
                        logger.warning("Snapshot foi publicado, mas a reabertura das stores falhou", exc_info=True)
                return True
            shutil.rmtree(build_dir, ignore_errors=True)
            engine.build_id = previous_engine_build_id
            if hasattr(engine, "reopen_stores"):
                try:
                    active = self.active_dir()
                    if os.path.isfile(os.path.join(active, "successor.db")):
                        engine.reopen_stores(active, read_only=True)
                except Exception:
                    pass
            return False

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_artifact_checksums(build_dir: str, meta=None) -> None:
        if meta is None:
            with open(os.path.join(build_dir, "meta.json"), "r", encoding="utf-8") as stream:
                meta = json.load(stream)
        artifact_hashes = meta.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("Manifesto v3 sem checksums de artefatos")
        basins_dir = os.path.join(build_dir, "basins")
        if not os.path.isdir(basins_dir):
            raise ValueError("Snapshot v3 sem diretório de bacias")
        expected_artifacts = {
            "node_ids.json", "embeddings.npz", "graph.json", "successor.db",
            "attractor.db", "bm25.json",
        }
        expected_artifacts.update(
            os.path.join("basins", name).replace(os.sep, "/")
            for name in os.listdir(basins_dir)
            if name.endswith(".json")
        )
        if set(artifact_hashes) != expected_artifacts:
            raise ValueError("Manifesto v3 não cobre exatamente todos os artefatos do snapshot")
        for relative, expected_hash in artifact_hashes.items():
            path = os.path.join(build_dir, *str(relative).split("/"))
            if not os.path.isfile(path) or BasinPersistence._sha256_file(path) != expected_hash:
                raise ValueError(f"Checksum ausente ou inválido: {relative}")

    @staticmethod
    def _sqlite_rows(path: str, table: str) -> Dict[str, Any]:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise ValueError(f"SQLite store ausente ou vazio: {os.path.basename(path)}")
        uri = f"{Path(path).resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ValueError(f"SQLite integrity_check falhou: {path}")
            found = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not found:
                raise ValueError(f"Tabela {table!r} ausente em {path}")
            rows = connection.execute(f'SELECT key, value FROM "{table}"').fetchall()
            return {str(key): json.loads(value) for key, value in rows}
        finally:
            connection.close()

    @staticmethod
    def _validate_build(build_dir: str, node_ids, embedding_shape, engine=None) -> None:
        required = (
            "node_ids.json", "embeddings.npz", "graph.json", "meta.json",
            "successor.db", "attractor.db", "bm25.json",
        )
        missing = [name for name in required if not os.path.isfile(os.path.join(build_dir, name))]
        if missing:
            raise ValueError(f"Build incompleto; faltam artefatos: {', '.join(missing)}")
        if not node_ids or len(embedding_shape) != 2 or embedding_shape[1] <= 0:
            raise ValueError("Snapshot v3 deve conter nós e embeddings não vazios")

        with open(os.path.join(build_dir, "meta.json"), "r", encoding="utf-8") as stream:
            meta = json.load(stream)
        BasinPersistence._validate_artifact_checksums(build_dir, meta)

        embedding_path = os.path.join(build_dir, "embeddings.npz")
        with zipfile.ZipFile(embedding_path) as archive:
            corrupt_entry = archive.testzip()
            if corrupt_entry:
                raise ValueError(f"Embedding compactado corrompido: {corrupt_entry}")
            with archive.open("vectors.npy") as vectors_file:
                version = np.lib.format.read_magic(vectors_file)
                if version == (1, 0):
                    saved_shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(vectors_file)
                else:
                    saved_shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(vectors_file)
        if tuple(saved_shape) != tuple(embedding_shape):
            raise ValueError("Dimensão serializada dos embeddings difere da matriz em memória")

        with open(os.path.join(build_dir, "node_ids.json"), "r", encoding="utf-8") as stream:
            saved_ids = json.load(stream)
        with open(os.path.join(build_dir, "graph.json"), "r", encoding="utf-8") as stream:
            graph_data = json.load(stream)
        graph_nodes = {str(node.get("id")) for node in graph_data.get("nodes", [])}
        if len(saved_ids) != len(set(saved_ids)) or set(saved_ids) != graph_nodes:
            raise ValueError("IDs do grafo e do arquivo node_ids.json não correspondem")
        if set(map(str, saved_ids)) != set(map(str, node_ids)):
            raise ValueError("IDs do snapshot diferem dos IDs serializados")
        if len(embedding_shape) != 2 or embedding_shape[0] != len(saved_ids):
            raise ValueError("Dimensão dos embeddings não corresponde à quantidade de nós")
        edges = graph_data.get("links", graph_data.get("edges", []))
        for edge in edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source not in graph_nodes or target not in graph_nodes:
                raise ValueError("Grafo contém aresta que aponta para nó ausente")
        if meta.get("buildId") != os.path.basename(build_dir):
            raise ValueError("buildId do manifesto não corresponde à pasta do snapshot")
        if int(meta.get("index_schema_version", 0)) != 3:
            raise IndexRebuildRequired("Snapshot incompatível; execute reindex para gerar formato v3")
        BasinPersistence._validate_index_metadata(meta.get("index_metadata"))
        if int(meta.get("node_count", -1)) != len(saved_ids):
            raise ValueError("Contagem de nós do manifesto inválida")
        if int(meta.get("embedding_dimension", -1)) != int(embedding_shape[1]):
            raise ValueError("Dimensão de embedding do manifesto inválida")

        graph_ids = set(map(str, saved_ids))
        successors = BasinPersistence._sqlite_rows(os.path.join(build_dir, "successor.db"), "successor")
        attractors = BasinPersistence._sqlite_rows(os.path.join(build_dir, "attractor.db"), "attractor_of")
        if set(successors) != graph_ids or set(attractors) != graph_ids:
            raise ValueError("Stores funcionais não cobrem todos os nós do snapshot")
        if any(value is not None and str(value) not in graph_ids for value in successors.values()):
            raise ValueError("Store successor contém referência a nó ausente")
        if any(str(value) not in graph_ids for value in attractors.values()):
            raise ValueError("Store attractor contém referência a nó ausente")

        basin_ids: set[str] = set()
        basin_members: set[str] = set()
        basins_dir = os.path.join(build_dir, "basins")
        for name in os.listdir(basins_dir):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(basins_dir, name), "r", encoding="utf-8") as stream:
                basin = json.load(stream)
            extra = basin.get("_meta") or {}
            basin_id = str(extra.get("basin_id") or "")
            members = {str(node.get("id")) for node in basin.get("nodes", [])}
            if not basin_id or not members or basin_id in basin_ids:
                raise ValueError(f"Artefato de bacia inválido: {name}")
            if basin_members.intersection(members):
                raise ValueError(f"Nós duplicados entre bacias: {name}")
            basin_ids.add(basin_id)
            basin_members.update(members)
        if basin_members != graph_ids:
            raise ValueError("As bacias não particionam exatamente os nós do grafo")
        if set(map(str, attractors.values())) != basin_ids:
            raise ValueError("IDs de atrator não correspondem aos artefatos de bacia")

        with open(os.path.join(build_dir, "bm25.json"), "r", encoding="utf-8") as stream:
            bm25 = json.load(stream)
        bm25_ids = list(map(str, bm25.get("doc_ids", [])))
        if (
            bm25.get("buildId") != meta["buildId"]
            or len(bm25_ids) != len(set(bm25_ids))
            or set(bm25_ids) != graph_ids
            or int(bm25.get("n", -1)) != len(graph_ids)
            or len(bm25.get("doc_len", [])) != len(graph_ids)
        ):
            raise ValueError("Índice BM25 não corresponde ao build ativo")
        postings = bm25.get("postings") or {}
        if any(
            int(doc_index) < 0 or int(doc_index) >= len(graph_ids) or int(tf) <= 0
            for pairs in postings.values() for doc_index, tf in pairs
        ):
            raise ValueError("Postings BM25 contêm posição ou frequência inválida")

    @staticmethod
    def _validate_index_metadata(metadata) -> None:
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Manifesto v3 sem index_metadata")
        if int(metadata.get("format_version", 0)) != 3:
            raise ValueError("Versão de index_metadata não suportada")
        if not isinstance(metadata.get("encoder_model"), str) or not metadata["encoder_model"]:
            raise ValueError("Manifesto v3 sem encoder_model")
        if not isinstance(metadata.get("tokenizer"), str) or not metadata["tokenizer"]:
            raise ValueError("Manifesto v3 sem tokenizer")
        for revision_field in ("encoder_revision", "tokenizer_revision", "reranker_revision"):
            if not isinstance(metadata.get(revision_field), str) or not metadata[revision_field]:
                raise ValueError(f"Manifesto v3 sem revisão imutável: {revision_field}")
        if int(metadata.get("chunk_policy_version", 0)) != 1:
            raise ValueError("Política de chunking ausente ou não suportada")
        for key in ("sources", "source_file_hashes"):
            if not isinstance(metadata.get(key), dict):
                raise ValueError(f"Manifesto v3 sem proveniência de fontes: {key}")

        mode = metadata.get("chunking_mode")
        if mode is None:
            mode = "tokens" if metadata.get("chunk_size_tokens") is not None else "characters"
        if mode == "tokens":
            size = metadata.get("chunk_size_tokens")
            overlap = metadata.get("chunk_overlap_tokens")
        elif mode == "characters":
            size = metadata.get("chunk_size")
            overlap = metadata.get("chunk_overlap")
        else:
            raise ValueError(f"Modo de chunking inválido: {mode!r}")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Manifesto v2 sem limite de chunk válido")
        if overlap is not None and (not isinstance(overlap, int) or overlap < 0):
            raise ValueError("Manifesto v2 com overlap inválido")

    def _prune_builds(self, keep) -> None:
        """Retain the active and immediately previous snapshots after successful publish."""
        try:
            builds_root = os.path.join(self.storage_dir, "builds")
            if not os.path.isdir(builds_root):
                return
            retained = set(keep)
            for name in os.listdir(builds_root):
                path = os.path.join(builds_root, name)
                if os.path.isdir(path) and _BUILD_ID_PATTERN.fullmatch(name) and name not in retained:
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        logger.warning("Não foi possível remover snapshot antigo %s", path, exc_info=True)
        except Exception:
            # Pruning happens after pointer commit and can never invalidate a published build.
            logger.warning("Falha ao limpar snapshots antigos após a publicação", exc_info=True)

    def load_topology(self, engine) -> bool:
        pointer_path = os.path.join(self.storage_dir, "current.json")
        if not os.path.exists(pointer_path):
            legacy_meta = os.path.join(self.storage_dir, "meta.json")
            if os.path.isfile(legacy_meta):
                with open(legacy_meta, "r", encoding="utf-8") as stream:
                    legacy = json.load(stream)
                raise IndexRebuildRequired(
                    f"Snapshot v{legacy.get('index_schema_version', 1)} é legado; "
                    "execute reindex em um storage_dir v3 novo."
                )
            if any(
                os.path.isfile(os.path.join(self.storage_dir, name))
                for name in ("graph.json", "embeddings.npy", "basin_data.json", "node_ids.json")
            ):
                raise IndexRebuildRequired(
                    "Índice sem ponteiro v3 detectado; execute reindex em um storage_dir novo."
                )
            return False

        root = self.active_dir()
        meta_path = os.path.join(root, "meta.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError("Snapshot inválido: meta.json ausente")
        with open(meta_path, "r", encoding="utf-8") as stream:
            meta = json.load(stream)
        schema_version = int(meta.get("index_schema_version", 0))
        if schema_version != 3:
            raise IndexRebuildRequired(
                f"Snapshot v{schema_version} não é aceito por esta release; "
                "execute reindex em um storage_dir v3 novo."
            )
        active_id = self.current_build_id()
        if meta.get("buildId") != active_id or os.path.basename(root) != active_id:
            raise RuntimeError("Snapshot inválido: ponteiro e manifesto não correspondem")

        self._validate_artifact_checksums(root, meta)

        with open(os.path.join(root, "node_ids.json"), "r", encoding="utf-8") as stream:
            saved_node_ids = json.load(stream)
        with np.load(os.path.join(root, "embeddings.npz"), allow_pickle=False) as archive:
            saved_shape = archive["vectors"].shape
        self._validate_build(root, saved_node_ids, saved_shape)

        graph_path = os.path.join(root, "graph.json")
        with open(graph_path, "r", encoding="utf-8") as stream:
            graph = load_graph(json.load(stream))
        with np.load(os.path.join(root, "embeddings.npz"), allow_pickle=False) as archive:
            vectors = archive["vectors"]
        if vectors.shape != tuple(saved_shape) or not np.isfinite(vectors).all():
            raise ValueError("Snapshot contém matriz de embeddings inválida")
        node_ids = saved_node_ids
        if len(node_ids) != len(graph):
            raise RuntimeError("Snapshot inválido: quantidade de nós não confere")
        for node_id, vector in zip(node_ids, vectors):
            graph.nodes[node_id]["embedding"] = vector

        from .topology import TopologicalBasin
        basins = {}
        for filename in os.listdir(os.path.join(root, "basins")):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(root, "basins", filename), "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            extra = payload.pop("_meta")
            basin_id = str(extra["basin_id"])
            basin = TopologicalBasin(basin_id)
            basin.rho_tree = load_graph(payload)
            basin.source = extra.get("source", "")
            basin.cohesion = extra.get("cohesion", 1.0)
            for node_id in basin.rho_tree.nodes:
                basin.rho_tree.nodes[node_id]["embedding"] = graph.nodes[node_id]["embedding"]
            basins[basin_id] = basin

        from ..indexer.bm25 import BM25Index
        bm25 = BM25Index()
        if not bm25.load(os.path.join(root, "bm25.json")) or bm25.build_id != active_id:
            raise RuntimeError("Snapshot inválido: índice BM25 ausente ou de outro build")

        from .kv_store import DiskKVStore
        successor = DiskKVStore(os.path.join(root, "successor.db"), "successor", read_only=True)
        try:
            attractor_of = DiskKVStore(os.path.join(root, "attractor.db"), "attractor_of", read_only=True)
        except Exception:
            successor.close()
            raise

        engine.close_stores()
        engine.graph = graph
        engine.basins = basins
        engine.successor = successor
        engine.attractor_of = attractor_of
        engine._kv_dir = root
        engine.section_size = int(meta.get("section_size", 20))
        engine.encoder_model = str(meta.get("encoder_model", ""))
        engine.build_id = active_id
        engine.index_schema_version = 3
        engine.index_metadata = meta.get("index_metadata", {})
        engine.gate_cache_tag = engine.index_metadata.get("gate_cache_tag", "")
        engine.bm25 = bm25
        return True
