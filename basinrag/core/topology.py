import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional
from ..logging_config import setup_logging

logger = setup_logging()

from .functional_graph import (
    DEFAULT_SECTION_SIZE,
    detect_attractors,
    members_by_attractor,
    reverse_hops,
    rho_parent_child,
    sequential_successor,
    sequential_successor_adaptive,
)
from .vector_index import build_ip_index


class TopologicalBasin:
    def __init__(self, attractor_id: str):
        self.attractor_id = attractor_id
        self.rho_tree = nx.DiGraph()
        self.source: str = ""
        self.cohesion: float = 1.0

    def add_node(self, node_id: str, hops: int, data: Dict):
        payload = {k: v for k, v in data.items() if k != "embedding"}
        payload["hops"] = hops
        self.rho_tree.add_node(node_id, **payload)
        if "embedding" in data:
            self.rho_tree.nodes[node_id]["embedding"] = data["embedding"]


class BasinTopologyEngine:
    """Undirected proximity graph + functional successor Ï† for basins."""

    def __init__(self, section_size: int = DEFAULT_SECTION_SIZE, max_hops: int = 50, storage_dir: Optional[str] = None):
        self.graph = nx.Graph()
        self.basins: Dict[str, TopologicalBasin] = {}
        
        self.storage_dir = storage_dir or ".basinrag"
        import os
        from .kv_store import DiskKVStore
        os.makedirs(self.storage_dir, exist_ok=True)
        self._kv_dir = self.storage_dir
        self.successor = DiskKVStore(os.path.join(self._kv_dir, "successor.db"), "successor")
        self.attractor_of = DiskKVStore(os.path.join(self._kv_dir, "attractor.db"), "attractor_of")
        
        self.section_size = section_size
        self.max_hops = max_hops
        self.bm25 = None
        self.encoder_model: str = ""
        self.build_id: str = ""

    def close_stores(self) -> None:
        for store in (getattr(self, "successor", None), getattr(self, "attractor_of", None)):
            if store is not None and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass

    def reopen_stores(self, directory: Optional[str] = None) -> None:
        import os
        from .kv_store import DiskKVStore
        self.close_stores()
        self._kv_dir = directory or self.storage_dir
        os.makedirs(self._kv_dir, exist_ok=True)
        self.successor = DiskKVStore(os.path.join(self._kv_dir, "successor.db"), "successor")
        self.attractor_of = DiskKVStore(os.path.join(self._kv_dir, "attractor.db"), "attractor_of")

    def reset(self) -> None:
        self.graph.clear()
        self.basins = {}
        if hasattr(self, 'successor') and hasattr(self.successor, 'clear'):
            self.successor.clear()
        if hasattr(self, 'attractor_of') and hasattr(self.attractor_of, 'clear'):
            self.attractor_of.clear()
        self.bm25 = None
        self.build_id = ""

    def build_graph(self, chunks: List[Dict[str, Any]]):
        """Rebuild from chunks. Always clears first so re-ingest cannot append."""
        self.reset()
        self.merge_graph(chunks)

    def merge_graph(self, chunks: List[Dict[str, Any]]):
        """Add or update chunks without wiping existing nodes, then rebuild φ and edges."""
        if not chunks:
            return

        for chunk in chunks:
            node_id = chunk["id"]
            text = chunk.get("text", "")
            self.graph.add_node(
                node_id,
                text=text,
                embedding=chunk["embedding"],
                source=chunk.get("source", ""),
                chunk_index=int(chunk.get("chunk_index", 0)),
                l1=chunk.get("l1", ""),
                l2=chunk.get("l2", ""),
                l3_summary=chunk.get("l3_summary", ""),
                metadata=chunk.get("metadata") or {},
                hops=0,
                basin_id="",
            )

        self.graph.remove_edges_from(list(self.graph.edges()))
        by_source: Dict[str, List[str]] = defaultdict(list)
        indexed: Dict[str, List] = defaultdict(list)
        for nid, data in self.graph.nodes(data=True):
            indexed[data.get("source") or "_anon"].append(
                (int(data.get("chunk_index", 0)), nid)
            )
        for src, pairs in indexed.items():
            by_source[src] = [nid for _, nid in sorted(pairs)]

        for ids in by_source.values():
            for i in range(len(ids) - 1):
                self.graph.add_edge(ids[i], ids[i + 1], weight=1.0, type="sequential")

        nodes = list(self.graph.nodes(data=True))
        n_chunks = len(nodes)
        if n_chunks > 1:
            embeddings = np.array([n[1]["embedding"] for n in nodes], dtype=np.float32)
            index = build_ip_index(embeddings)
            k = min(6, n_chunks)
            query = np.ascontiguousarray(embeddings, dtype=np.float32)
            import faiss
            faiss.normalize_L2(query)
            _D, I = index.search(query, k)
            for i in range(n_chunks):
                n1_id = nodes[i][0]
                for j in range(1, k):
                    neighbor_idx = int(I[i][j])
                    if neighbor_idx < 0:
                        continue
                    n2_id = nodes[neighbor_idx][0]
                    if n1_id == n2_id or self.graph.has_edge(n1_id, n2_id):
                        continue
                    sim = float(_D[i][j])
                    if sim > 0.85:
                        self.graph.add_edge(n1_id, n2_id, weight=sim, type="virtual-edge")

        emb_map = {nid: data["embedding"] for nid, data in self.graph.nodes(data=True)
                   if "embedding" in data}
        successor_dict = sequential_successor_adaptive(
            by_source, emb_map,
            fallback_section_size=self.section_size,
        )
        if hasattr(self.successor, "set_many"):
            self.successor.set_many(successor_dict)
        else:
            for k, v in successor_dict.items():
                self.successor[k] = v

    def partition_into_basins(self, preserve_l3: Optional[Dict[str, Dict[str, Any]]] = None):
        """Partition along Ï† (sequential successor), not PageRank."""
        logger.info("Criando bacias de atracao (grafo funcional)...")
        if not self.graph.nodes:
            self.basins = {}
            return

        if not self.successor:
            by_source: Dict[str, List[str]] = defaultdict(list)
            indexed: Dict[str, List] = defaultdict(list)
            for nid, data in self.graph.nodes(data=True):
                indexed[data.get("source") or "_anon"].append(
                    (int(data.get("chunk_index", 0)), nid)
                )
            for src, pairs in indexed.items():
                by_source[src] = [nid for _, nid in sorted(pairs)]
            succ_dict = sequential_successor(by_source, self.section_size)
            if hasattr(self.successor, "set_many"):
                self.successor.set_many(succ_dict)
            else:
                for k, v in succ_dict.items():
                    self.successor[k] = v

        succ_map = dict(self.successor.items()) if hasattr(self.successor, "items") else dict(self.successor)
        attr_dict = detect_attractors(succ_map)
        if hasattr(self.attractor_of, "set_many"):
            self.attractor_of.set_many(attr_dict)
        else:
            for k, v in attr_dict.items():
                self.attractor_of[k] = v
        hops = reverse_hops(succ_map, attr_dict)
        groups = members_by_attractor(attr_dict)
        preserve_l3 = preserve_l3 or {}

        for nid, data in self.graph.nodes(data=True):
            attr = attr_dict.get(nid, nid)
            data["basin_id"] = attr
            data["hops"] = hops.get(nid, 0)

        self.basins = {}
        for attr, members in groups.items():
            valid_members_for_src = [n for n in members if self.graph.has_node(n)]
            if not valid_members_for_src:
                continue
            basin = TopologicalBasin(attr)
            srcs = {self.graph.nodes[n].get("source", "") for n in valid_members_for_src}
            basin.source = next(iter(srcs)) if len(srcs) == 1 else ""
            
            valid_members = []
            for nid in valid_members_for_src:
                hop = hops.get(nid, 0)
                self.graph.nodes[nid]["hops"] = hop
                valid_members.append(nid)
                basin.add_node(nid, hop, dict(self.graph.nodes[nid]))
                
            for parent, child in rho_parent_child(succ_map, valid_members):
                basin.rho_tree.add_edge(parent, child)
            basin.cohesion = self._cohesion(valid_members)
            if basin.rho_tree.has_node(attr):
                from ..indexer.condensation import basin_l3_label
                texts = [self.graph.nodes[n].get("text", "") for n in valid_members[:8]]
                basin.rho_tree.nodes[attr]["l3_label"] = basin_l3_label(
                    texts, basin.source
                )
                if basin.rho_tree.nodes[attr].get("l3_source") != "llm":
                    basin.rho_tree.nodes[attr]["l3_source"] = "extractive"
                saved = preserve_l3.get(attr) or {}
                if saved.get("l3_source") == "llm":
                    basin.rho_tree.nodes[attr]["l3_summary"] = saved.get("l3_summary", "")
                    basin.rho_tree.nodes[attr]["l3_source"] = "llm"
            self.basins[attr] = basin

        logger.info(f"  Atratores identificados: {len(self.basins)}")

    def _cohesion(self, members: List[str]) -> float:
        member_set = set(members)
        if len(member_set) <= 1:
            return 1.0
        internal = 0.0
        total = 0.0
        for nid in member_set:
            for nbr in self.graph.neighbors(nid):
                w = float(self.graph.edges[nid, nbr].get("weight", 1.0))
                total += w
                if nbr in member_set:
                    internal += w
        return 1.0 if total == 0 else round(internal / total, 2)

    def hops_of(self, node_id: str) -> int:
        return int(self.graph.nodes[node_id].get("hops", 0)) if node_id in self.graph else 0

    def basin_id_of(self, node_id: str) -> str:
        return str(self.graph.nodes[node_id].get("basin_id", "")) if node_id in self.graph else ""
