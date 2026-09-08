import os
import json
import shutil
import uuid
import numpy as np
import networkx as nx
from typing import Any, Dict
from ..logging_config import setup_logging

logger = setup_logging()

from ..indexer.bm25 import BM25Index


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


class BasinPersistence:
    """Split persistence: metadata JSON, embeddings NPZ, BM25 JSON, buildId."""

    def __init__(self, storage_dir: str = ".basinrag"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.basins_dir = os.path.join(storage_dir, "basins")
        os.makedirs(self.basins_dir, exist_ok=True)

    def save_topology(self, engine) -> bool:
        logger.info(f"Salvando indice em {self.storage_dir}...")
        tmp = self.storage_dir.rstrip("\\/") + ".tmp-" + uuid.uuid4().hex[:8]
        try:
            os.makedirs(tmp, exist_ok=True)
            basins_tmp = os.path.join(tmp, "basins")
            os.makedirs(basins_tmp, exist_ok=True)

            nodes = list(engine.graph.nodes(data=True))
            node_ids = [n[0] for n in nodes]
            embeddings = np.array(
                [n[1].get("embedding", np.zeros(1, dtype=np.float32)) for n in nodes],
                dtype=np.float32,
            )
            with open(os.path.join(tmp, "node_ids.json"), "w", encoding="utf-8") as f:
                json.dump(node_ids, f)
            np.savez_compressed(
                os.path.join(tmp, "embeddings.npz"),
                vectors=embeddings,
            )

            graph_data = dump_graph(snapshot_without_embeddings(engine.graph))
            with open(os.path.join(tmp, "graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False)

            build_id = uuid.uuid4().hex
            engine.build_id = build_id
            meta = {
                "buildId": build_id,
                "successor": dict(engine.successor.items()) if hasattr(engine.successor, 'items') else engine.successor,
                "attractor_of": dict(engine.attractor_of.items()) if hasattr(engine.attractor_of, 'items') else engine.attractor_of,
                "section_size": getattr(engine, "section_size", 20),
                "encoder_model": getattr(engine, "encoder_model", ""),
            }
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

            for basin_id, basin in engine.basins.items():
                basin_data = dump_graph(snapshot_without_embeddings(basin.rho_tree))
                basin_data["_meta"] = {
                    "source": getattr(basin, "source", ""),
                    "cohesion": getattr(basin, "cohesion", 1.0),
                }
                path = os.path.join(basins_tmp, f"{basin_id}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(basin_data, f, ensure_ascii=False)

            if getattr(engine, "bm25", None) is not None:
                engine.bm25.save(os.path.join(tmp, "bm25.json"), build_id=build_id)

            os.makedirs(self.storage_dir, exist_ok=True)
            for name in ("embeddings.npz", "node_ids.json", "graph.json", "meta.json", "bm25.json"):
                src = os.path.join(tmp, name)
                if os.path.exists(src):
                    os.replace(src, os.path.join(self.storage_dir, name))

            live_basins = self.basins_dir
            old_basins = live_basins + ".old"
            if os.path.isdir(old_basins):
                shutil.rmtree(old_basins, ignore_errors=True)
            if os.path.isdir(live_basins):
                os.replace(live_basins, old_basins)
            os.replace(basins_tmp, live_basins)
            if os.path.isdir(old_basins):
                shutil.rmtree(old_basins, ignore_errors=True)

            shutil.rmtree(tmp, ignore_errors=True)
            return True
        except (IOError, OSError):
            logger.exception("Erro de I/O ao salvar")
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        except Exception:
            logger.exception("Erro inesperado ao salvar topologia")
            shutil.rmtree(tmp, ignore_errors=True)
            return False

    def load_topology(self, engine) -> bool:
        graph_path = os.path.join(self.storage_dir, "graph.json")
        emb_path = os.path.join(self.storage_dir, "embeddings.npz")

        if not os.path.exists(graph_path):
            return False
        if not os.path.exists(emb_path):
            logger.info("Erro ao carregar topologia: embeddings.npz ausente.")
            return False

        logger.info("Carregando memoria topologica...")
        try:
            emb_map = {}
            ids_path = os.path.join(self.storage_dir, "node_ids.json")
            if os.path.exists(ids_path):
                with open(ids_path, "r", encoding="utf-8") as f:
                    node_ids_list = json.load(f)
                data = np.load(emb_path, allow_pickle=False)
                for nid, vec in zip(node_ids_list, data["vectors"]):
                    emb_map[str(nid)] = vec
            else:
                # Fallback for legacy format (pre-migration)
                data = np.load(emb_path, allow_pickle=True)
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

            meta_path = os.path.join(self.storage_dir, "meta.json")
            meta_build_id = ""
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                succ_data = meta.get("successor", {})
                for k, v in succ_data.items():
                    engine.successor[k] = v if v is not None else None
                attr_data = meta.get("attractor_of", {})
                for k, v in attr_data.items():
                    engine.attractor_of[k] = v
                engine.section_size = meta.get("section_size", 20)
                engine.encoder_model = meta.get("encoder_model", "")
                meta_build_id = meta.get("buildId", "") or ""
            else:
                engine.successor = {}
                engine.attractor_of = {}
                engine.encoder_model = ""
            engine.build_id = meta_build_id

            from .topology import TopologicalBasin

            engine.basins = {}
            preserve_l3: Dict[str, Dict[str, Any]] = {}
            if os.path.exists(self.basins_dir):
                for fname in os.listdir(self.basins_dir):
                    if not fname.endswith(".json"):
                        continue
                    basin_id = fname[:-5]
                    with open(os.path.join(self.basins_dir, fname), "r", encoding="utf-8") as f:
                        b_data = json.load(f)
                    extra = b_data.pop("_meta", {})
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

            bm25_path = os.path.join(self.storage_dir, "bm25.json")
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
