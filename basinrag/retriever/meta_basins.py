"""Meta-Basins Level 2: Cross-document attractor communities using NetworkX."""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


class MetaBasin:
    """Nível 2: Comunidade de atratores que une bacias entre múltiplos documentos."""

    def __init__(self, meta_id: str, attractor_ids: List[str]):
        self.meta_id = meta_id
        self.attractor_ids = attractor_ids
        self.l3_cross_summary: str = ""
        self.medoid_id: str = attractor_ids[0] if attractor_ids else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta_id": self.meta_id,
            "attractor_ids": self.attractor_ids,
            "l3_cross_summary": self.l3_cross_summary,
            "medoid_id": self.medoid_id,
        }


def build_meta_basins(
    engine: Any,
    similarity_threshold: float = 0.70,
) -> Dict[str, MetaBasin]:
    """Descobre Meta-Bacias através de modularidade nativa do NetworkX sem scikit-learn."""
    if not hasattr(engine, "basins") or len(engine.basins) <= 1:
        return {}

    attractors = list(engine.basins.keys())
    valid_attrs = []
    embs = []

    for attr in attractors:
        if attr in engine.graph and "embedding" in engine.graph.nodes[attr]:
            vec = engine.graph.nodes[attr]["embedding"]
            if vec is not None:
                vec = np.asarray(vec, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    valid_attrs.append(attr)
                    embs.append(vec / norm)

    if len(valid_attrs) <= 1:
        return {}

    # Constrói grafo de afinidade entre atratores
    g_attr = nx.Graph()
    for attr in valid_attrs:
        g_attr.add_node(attr)

    n = len(valid_attrs)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embs[i], embs[j]))
            if sim >= similarity_threshold:
                g_attr.add_edge(valid_attrs[i], valid_attrs[j], weight=sim)

    # Detecção de comunidades nativa via Modularidade
    try:
        communities = list(greedy_modularity_communities(g_attr, weight="weight"))
    except Exception:
        # Fallback: componentes conexos
        communities = list(nx.connected_components(g_attr))

    meta_basins: Dict[str, MetaBasin] = {}
    for idx, comm in enumerate(communities):
        comm_attrs = list(comm)
        meta_id = f"meta_{idx}"
        meta = MetaBasin(meta_id, comm_attrs)
        # Calcula o medoid (atrator com maior similaridade média com os outros da comunidade)
        if len(comm_attrs) > 1:
            attr_indices = [valid_attrs.index(a) for a in comm_attrs if a in valid_attrs]
            comm_embs = [embs[i] for i in attr_indices]
            centroid = np.mean(comm_embs, axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            best_sim = -1.0
            best_attr = comm_attrs[0]
            for a, v in zip(comm_attrs, comm_embs):
                sim = float(np.dot(centroid, v))
                if sim > best_sim:
                    best_sim = sim
                    best_attr = a
            meta.medoid_id = best_attr
        meta_basins[meta_id] = meta

    return meta_basins
