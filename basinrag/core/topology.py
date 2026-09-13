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

    def __init__(
        self,
        section_size: int = DEFAULT_SECTION_SIZE,
        storage_dir: Optional[str] = None,
        *,
        open_stores: bool = True,
    ):
        self.graph = nx.Graph()
        self.basins: Dict[str, TopologicalBasin] = {}
        
        self.storage_dir = storage_dir or ".basinrag"
        import os
        self._kv_dir = self.storage_dir
        self.successor: Any
        self.attractor_of: Any
        if open_stores:
            from .kv_store import DiskKVStore
            os.makedirs(self.storage_dir, exist_ok=True)
            self.successor = DiskKVStore(os.path.join(self._kv_dir, "successor.db"), "successor")
            self.attractor_of = DiskKVStore(os.path.join(self._kv_dir, "attractor.db"), "attractor_of")
        else:
            self.successor = {}
            self.attractor_of = {}
        
        self.section_size = section_size
        self.bm25: Any = None
        self.encoder_model: str = ""
        self.build_id: str = ""
        self.index_metadata: Dict[str, Any] = {}

    def close_stores(self) -> None:
        for store in (getattr(self, "successor", None), getattr(self, "attractor_of", None)):
            if store is not None and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass

    def reopen_stores(self, directory: Optional[str] = None, *, read_only: bool = False) -> None:
        import os
        from .kv_store import DiskKVStore
        self.close_stores()
        self._kv_dir = directory or self.storage_dir
        if not read_only:
            os.makedirs(self._kv_dir, exist_ok=True)
        self.successor = DiskKVStore(
            os.path.join(self._kv_dir, "successor.db"), "successor", read_only=read_only
        )
        self.attractor_of = DiskKVStore(
            os.path.join(self._kv_dir, "attractor.db"), "attractor_of", read_only=read_only
        )

    def reset(self) -> None:
        self.graph.clear()
        self.basins = {}
        if hasattr(self, 'successor') and hasattr(self.successor, 'clear'):
            self.successor.clear()
        if hasattr(self, 'attractor_of') and hasattr(self.attractor_of, 'clear'):
            self.attractor_of.clear()
        self.bm25 = None
        self.build_id = ""
        self.index_metadata = {}

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

        # Note: virtual-edge linking is deferred to _link_semantic_neighbors(),
        # called after partition_into_basins() to enable across-basin filtering.

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

    MAX_SEMANTIC_DEGREE = 5
    SEMANTIC_SIM_THRESHOLD = 0.85

    def _link_semantic_neighbors(self):
        """Inject kNN virtual edges across different basins only.

        Adapted from BasinMind's cross-basin semantic synapses:
        - Only creates edges between nodes in DIFFERENT basins
        - Caps virtual-edge degree at MAX_SEMANTIC_DEGREE per node
        - Requires cosine similarity >= SEMANTIC_SIM_THRESHOLD
        """
        import faiss

        # Remove existing virtual edges
        virtual_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "virtual-edge"]
        self.graph.remove_edges_from(virtual_edges)

        nodes = list(self.graph.nodes(data=True))
        n_chunks = len(nodes)
        if n_chunks <= 1:
            return

        embeddings = np.array([n[1]["embedding"] for n in nodes], dtype=np.float32)
        index = build_ip_index(embeddings)
        query = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(query)
        virtual_degree = {node_id: 0 for node_id, _ in nodes}
        search_rows = np.arange(n_chunks, dtype=np.int64)
        k = min(n_chunks, 32)
        for _pass in range(3):
            if len(search_rows) == 0 or k <= 1:
                break
            distances, indices = index.search(query[search_rows], k)
            candidates = {}
            for row, source_idx in enumerate(search_rows):
                n1_id = nodes[int(source_idx)][0]
                n1_basin = self.graph.nodes[n1_id].get("basin_id", "")
                for neighbor_idx, score in zip(indices[row], distances[row]):
                    neighbor_idx = int(neighbor_idx)
                    if neighbor_idx < 0 or neighbor_idx == int(source_idx):
                        continue
                    n2_id = nodes[neighbor_idx][0]
                    if self.graph.has_edge(n1_id, n2_id):
                        continue
                    sim = float(score)
                    if sim < self.SEMANTIC_SIM_THRESHOLD:
                        continue
                    n2_basin = self.graph.nodes[n2_id].get("basin_id", "")
                    if n1_basin and n2_basin and n1_basin == n2_basin:
                        continue
                    pair = tuple(sorted((int(source_idx), neighbor_idx)))
                    candidates[pair] = max(sim, candidates.get(pair, -1.0))

            ordered = sorted(
                candidates.items(),
                key=lambda item: (-item[1], str(nodes[item[0][0]][0]), str(nodes[item[0][1]][0])),
            )
            additions = 0
            for (i, j), similarity in ordered:
                n1_id, n2_id = nodes[i][0], nodes[j][0]
                if self.graph.has_edge(n1_id, n2_id):
                    continue
                if (
                    virtual_degree[n1_id] >= self.MAX_SEMANTIC_DEGREE
                    or virtual_degree[n2_id] >= self.MAX_SEMANTIC_DEGREE
                ):
                    continue
                self.graph.add_edge(n1_id, n2_id, weight=similarity, type="virtual-edge")
                virtual_degree[n1_id] += 1
                virtual_degree[n2_id] += 1
                additions += 1
            if additions == 0:
                break
            search_rows = np.asarray(
                [i for i, node in enumerate(nodes)
                 if virtual_degree[node[0]] < self.MAX_SEMANTIC_DEGREE],
                dtype=np.int64,
            )
            if k >= min(n_chunks, 128) or not len(search_rows):
                break
            k = min(n_chunks, 128, k * 2)

    def partition_into_basins(
        self,
        preserve_l3: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        experimental_topology: bool = False,
    ):
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

        # Virtual links are expensive and experimental; hybrid RRF does not need them.
        virtual_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data.get("type") == "virtual-edge"]
        self.graph.remove_edges_from(virtual_edges)
        if experimental_topology:
            self._link_semantic_neighbors()

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
