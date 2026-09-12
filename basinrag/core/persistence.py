import hashlib
import os
import json
import shutil
import time
import uuid
import numpy as np
import networkx as nx
from typing import Any, Dict, Optional
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


def safe_replace_dir(src_dir: str, dst_dir: str, retries: int = 5, delay: float = 0.1) -> None:
    """Replace a directory by rename. Never merge with copytree (that ghosts stale basins)."""
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Diretório de origem não existe: {src_dir}")

    if not os.path.exists(dst_dir):
        os.rename(src_dir, dst_dir)
        return

    dead_dir = f"{dst_dir}.dead.{uuid.uuid4().hex[:8]}"
    last_error: Optional[OSError] = None
    for attempt in range(retries):
        try:
            os.rename(dst_dir, dead_dir)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(delay * (2 ** attempt))
    if last_error is not None:
        raise RuntimeError(f"Falha ao recuar o diretório ativo: {last_error}") from last_error

    try:
        os.rename(src_dir, dst_dir)
    except Exception as exc:
        try:
            os.rename(dead_dir, dst_dir)
        except Exception:
            pass
        raise RuntimeError(f"Falha crítica ao ativar novo diretório: {exc}") from exc

    for attempt in range(retries):
        try:
            shutil.rmtree(dead_dir, ignore_errors=False)
            break
        except OSError:
            if attempt == retries - 1:
                shutil.rmtree(dead_dir, ignore_errors=True)
            time.sleep(delay * (2 ** attempt))


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


class BasinPersistence:
    """Split persistence: metadata JSON, embeddings NPZ, BM25 JSON, atomic current.json pointer."""

    def __init__(self, storage_dir: str = ".basinrag"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "builds"), exist_ok=True)

    def active_dir(self) -> str:
        pointer = os.path.join(self.storage_dir, "current.json")
        if os.path.exists(pointer):
            try:
                with open(pointer, "r", encoding="utf-8") as f:
                    data = json.load(f)
                build_id = data.get("build_id") or data.get("buildId") or ""
                if build_id:
                    cand = os.path.join(self.storage_dir, "builds", str(build_id))
                    if os.path.isdir(cand):
                        return cand
            except (OSError, json.JSONDecodeError):
                pass
        return self.storage_dir

    @property
    def basins_dir(self) -> str:
        return os.path.join(self.active_dir(), "basins")

    def save_topology(self, engine) -> bool:
        logger.info(f"Salvando indice em {self.storage_dir}...")
        build_id = uuid.uuid4().hex
        build_dir = os.path.join(self.storage_dir, "builds", build_id)
        try:
            os.makedirs(os.path.join(build_dir, "basins"), exist_ok=True)

            nodes = list(engine.graph.nodes(data=True))
            node_ids = [n[0] for n in nodes]
            embeddings = np.array(
                [n[1].get("embedding", np.zeros(1, dtype=np.float32)) for n in nodes],
                dtype=np.float32,
            )
            with open(os.path.join(build_dir, "node_ids.json"), "w", encoding="utf-8") as f:
                json.dump(node_ids, f)
            np.savez_compressed(
                os.path.join(build_dir, "embeddings.npz"),
                vectors=embeddings,
            )

            graph_data = dump_graph(snapshot_without_embeddings(engine.graph))
            with open(os.path.join(build_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False)

            engine.build_id = build_id
            meta = {
                "buildId": build_id,
                "section_size": getattr(engine, "section_size", 20),
                "encoder_model": getattr(engine, "encoder_model", ""),
            }

            if hasattr(engine.successor, "backup_to"):
                engine.successor.backup_to(os.path.join(build_dir, "successor.db"))
            else:
                meta["successor"] = dict(engine.successor.items()) if hasattr(engine.successor, "items") else engine.successor

            if hasattr(engine.attractor_of, "backup_to"):
                engine.attractor_of.backup_to(os.path.join(build_dir, "attractor.db"))
            else:
                meta["attractor_of"] = dict(engine.attractor_of.items()) if hasattr(engine.attractor_of, "items") else engine.attractor_of

            if hasattr(engine, "close_stores"):
                engine.close_stores()

            with open(os.path.join(build_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

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

            if getattr(engine, "bm25", None) is not None:
                engine.bm25.save(os.path.join(build_dir, "bm25.json"), build_id=build_id)

            _atomic_write_json(
                os.path.join(self.storage_dir, "current.json"),
                {"build_id": build_id},
            )
            if hasattr(engine, "reopen_stores"):
                engine.reopen_stores(build_dir)
            return True
        except (IOError, OSError):
            logger.exception("Erro de I/O ao salvar")
            shutil.rmtree(build_dir, ignore_errors=True)
            if hasattr(engine, "reopen_stores"):
                try:
                    engine.reopen_stores(self.active_dir())
                except Exception:
                    pass
            return False
        except Exception:
            logger.exception("Erro inesperado ao salvar topologia")
            shutil.rmtree(build_dir, ignore_errors=True)
            if hasattr(engine, "reopen_stores"):
                try:
                    engine.reopen_stores(self.active_dir())
                except Exception:
                    pass
            return False

    def load_topology(self, engine) -> bool:
        root = self.active_dir()
        graph_path = os.path.join(root, "graph.json")
        emb_path = os.path.join(root, "embeddings.npz")

        if not os.path.exists(graph_path):
            return False
        if not os.path.exists(emb_path):
            logger.info("Erro ao carregar topologia: embeddings.npz ausente.")
            return False

        logger.info("Carregando memoria topologica...")
        try:
            if hasattr(engine, "reopen_stores"):
                engine.reopen_stores(root)

            emb_map = {}
            ids_path = os.path.join(root, "node_ids.json")
            if os.path.exists(ids_path):
                with open(ids_path, "r", encoding="utf-8") as f:
                    node_ids_list = json.load(f)
                with np.load(emb_path, allow_pickle=False) as data:
                    for nid, vec in zip(node_ids_list, data["vectors"]):
                        emb_map[str(nid)] = vec
            else:
                with np.load(emb_path, allow_pickle=False) as data:
                    for nid, vec in zip(data["ids"], data["vectors"]):
                        emb_map[str(nid)] = vec

            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            engine.graph = load_graph(graph_data)

            missing_emb = [n for n in engine.graph.nodes if n not in emb_map]
            if missing_emb:
                logger.info(
                    f"Erro ao carregar topologia: {len(missing_emb)} nos sem embedding no NPZ."
                )
                return False

            for node_id in engine.graph.nodes:
                engine.graph.nodes[node_id]["embedding"] = emb_map[node_id]

            meta_path = os.path.join(root, "meta.json")
            meta_build_id = ""
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                engine.section_size = meta.get("section_size", 20)
                engine.encoder_model = meta.get("encoder_model", "")
                meta_build_id = meta.get("buildId", "") or ""
            else:
                meta = {}
                engine.encoder_model = ""

            succ_db_path = os.path.join(root, "successor.db")
            if os.path.exists(succ_db_path) and hasattr(engine.successor, "restore_from"):
                engine.successor.restore_from(succ_db_path)
            elif "successor" in meta:
                succ_data = meta.get("successor", {})
                if hasattr(engine.successor, "clear"):
                    engine.successor.clear()
                if hasattr(engine.successor, "set_many"):
                    engine.successor.set_many(succ_data)
                else:
                    for k, v in succ_data.items():
                        engine.successor[k] = v if v is not None else None
            else:
                if hasattr(engine.successor, "clear"):
                    engine.successor.clear()
                else:
                    engine.successor = {}

            attr_db_path = os.path.join(root, "attractor.db")
            if os.path.exists(attr_db_path) and hasattr(engine.attractor_of, "restore_from"):
                engine.attractor_of.restore_from(attr_db_path)
            elif "attractor_of" in meta:
                attr_data = meta.get("attractor_of", {})
                if hasattr(engine.attractor_of, "clear"):
                    engine.attractor_of.clear()
                if hasattr(engine.attractor_of, "set_many"):
                    engine.attractor_of.set_many(attr_data)
                else:
                    for k, v in attr_data.items():
                        engine.attractor_of[k] = v
            else:
                if hasattr(engine.attractor_of, "clear"):
                    engine.attractor_of.clear()
                else:
                    engine.attractor_of = {}

            engine.build_id = meta_build_id

            from .topology import TopologicalBasin

            engine.basins = {}
            preserve_l3: Dict[str, Dict[str, Any]] = {}
            basins_dir = os.path.join(root, "basins")
            if os.path.isdir(basins_dir):
                for fname in os.listdir(basins_dir):
                    if not fname.endswith(".json"):
                        continue
                    with open(os.path.join(basins_dir, fname), "r", encoding="utf-8") as f:
                        b_data = json.load(f)
                    extra = b_data.pop("_meta", {}) or {}
                    basin_id = extra.get("basin_id")
                    if not basin_id:
                        stem = fname[:-5]
                        if ".." in stem or "/" in stem or "\\" in stem:
                            continue
                        basin_id = stem
                    basin = TopologicalBasin(basin_id)
                    basin.rho_tree = load_graph(b_data)
                    basin.source = extra.get("source", "")
                    basin.cohesion = extra.get("cohesion", 1.0)
                    for nid in basin.rho_tree.nodes:
                        if nid in emb_map:
                            basin.rho_tree.nodes[nid]["embedding"] = emb_map[nid]
                    engine.basins[basin_id] = basin
                    if basin.rho_tree.has_node(basin_id):
                        node = basin.rho_tree.nodes[basin_id]
                        if node.get("l3_source") == "llm":
                            preserve_l3[basin_id] = {
                                "l3_source": "llm",
                                "l3_summary": node.get("l3_summary", ""),
                            }

            expected = set(engine.attractor_of.values()) if engine.attractor_of else set()
            if not expected:
                expected = {
                    str(d.get("basin_id"))
                    for _, d in engine.graph.nodes(data=True)
                    if d.get("basin_id")
                }
            if expected - set(engine.basins) or not engine.basins:
                engine.partition_into_basins(preserve_l3=preserve_l3)

            bm25_path = os.path.join(root, "bm25.json")
            from ..indexer.bm25 import BM25Index

            engine.bm25 = BM25Index()
            if not engine.bm25.load(bm25_path):
                engine.bm25 = None
            elif (
                engine.bm25.build_id != meta_build_id
                or set(engine.bm25.doc_ids) != set(engine.graph.nodes)
            ):
                engine.bm25 = None

            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.info(f"Erro ao carregar topologia: {e}")
            return False
