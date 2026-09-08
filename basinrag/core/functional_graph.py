"""Functional-graph collapse for document chunks.

Each chunk has one successor (φ): the previous chunk in the same source
section. Semantic kNN edges stay virtual and never define φ — the same
rule Basinfy uses for import vs HNSW synapses.

Attractors are section-start sinks. Hops are reverse-BFS distance along φ.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_SECTION_SIZE = 20


def sequential_successor(
    ordered_ids_by_source: Dict[str, List[str]],
    section_size: int = DEFAULT_SECTION_SIZE,
) -> Dict[str, Optional[str]]:
    """Drain each section toward its first chunk (sink / attractor)."""
    successor: Dict[str, Optional[str]] = {}
    size = max(1, int(section_size))
    for ids in ordered_ids_by_source.values():
        for i, nid in enumerate(ids):
            section_start = (i // size) * size
            successor[nid] = None if i == section_start else ids[i - 1]
    return successor


def adaptive_section_breaks(
    ordered_ids: List[str],
    embeddings_by_id: Dict[str, any],
    similarity_threshold: float = 0.55,
    max_section: int = 30,
    min_section: int = 3,
) -> List[int]:
    """Detect section breaks by cosine similarity drops between consecutive chunks."""
    import numpy as np

    breaks = [0]
    for i in range(1, len(ordered_ids)):
        if i - breaks[-1] >= max_section:
            breaks.append(i)
            continue
        if i - breaks[-1] < min_section:
            continue
        emb_prev = embeddings_by_id.get(ordered_ids[i - 1])
        emb_curr = embeddings_by_id.get(ordered_ids[i])
        if emb_prev is None or emb_curr is None:
            continue
        emb_prev = np.asarray(emb_prev, dtype=np.float32)
        emb_curr = np.asarray(emb_curr, dtype=np.float32)
        dot = float(np.dot(emb_prev, emb_curr))
        norm = float(np.linalg.norm(emb_prev) * np.linalg.norm(emb_curr))
        sim = dot / norm if norm > 0 else 0.0
        if sim < similarity_threshold:
            breaks.append(i)
    return breaks


def sequential_successor_adaptive(
    ordered_ids_by_source: Dict[str, List[str]],
    embeddings_by_id: Dict[str, any],
    fallback_section_size: int = DEFAULT_SECTION_SIZE,
    similarity_threshold: float = 0.55,
) -> Dict[str, Optional[str]]:
    """Use semantic breaks when embeddings are available; fixed-size fallback otherwise."""
    successor: Dict[str, Optional[str]] = {}
    for ids in ordered_ids_by_source.values():
        if embeddings_by_id:
            breaks = adaptive_section_breaks(
                ids,
                embeddings_by_id,
                similarity_threshold=similarity_threshold,
                max_section=fallback_section_size + 10,
            )
        else:
            breaks = list(range(0, len(ids), max(1, fallback_section_size)))
        break_set = set(breaks)
        for i, nid in enumerate(ids):
            if i in break_set:
                successor[nid] = None
            else:
                successor[nid] = ids[i - 1]
    return successor


def detect_attractors(successor: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Map every node to its attractor (sink or cycle representative) using O(V) 3-state DFS."""
    assigned: Dict[str, str] = {}
    state: Dict[str, int] = {k: 0 for k in successor}  # 0=unvisited, 1=path, 2=settled
    
    for start in successor:
        if state.get(start, 0) == 2:
            continue
            
        curr: Optional[str] = start
        path: List[str] = []
        
        while curr is not None and state.get(curr, 0) == 0:
            state[curr] = 1
            path.append(curr)
            curr = successor.get(curr)
            
        if curr is None:
            attr = path[-1] if path else start
        elif state.get(curr, 0) == 2:
            attr = assigned.get(curr, curr)
        elif state.get(curr, 0) == 1:
            # Cycle detected
            cycle_idx = path.index(curr)
            cycle = path[cycle_idx:]
            attr = min(cycle)
        else:
            attr = curr
            
        for nid in path:
            state[nid] = 2
            assigned[nid] = attr
            
    return assigned

def compute_trapping_bounds(hops: int, max_hops: int = 5) -> bool:
    """Evaluate if a node falls within the trapping bound (False if peripheral)."""
    return hops <= max_hops

def reverse_hops(
    successor: Dict[str, Optional[str]],
    attractor_of: Dict[str, str],
) -> Dict[str, int]:
    preds: Dict[str, List[str]] = defaultdict(list)
    for nid, nxt in successor.items():
        if nxt is not None:
            preds[nxt].append(nid)

    hops: Dict[str, int] = {}
    queue: deque[str] = deque()
    for attr in set(attractor_of.values()):
        hops[attr] = 0
        queue.append(attr)

    while queue:
        curr = queue.popleft()
        for pred in preds.get(curr, ()):
            if pred not in hops:
                hops[pred] = hops[curr] + 1
                queue.append(pred)

    for nid in successor:
        hops.setdefault(nid, 0)
    return hops


def members_by_attractor(attractor_of: Dict[str, str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for nid, attr in attractor_of.items():
        groups[attr].append(nid)
    return dict(groups)


def rho_parent_child(
    successor: Dict[str, Optional[str]],
    members: Iterable[str],
) -> List[Tuple[str, str]]:
    """Directed tree edges parent → child (φ inverse inside the basin)."""
    member_set = set(members)
    edges: List[Tuple[str, str]] = []
    for child, parent in successor.items():
        if child in member_set and parent in member_set:
            edges.append((parent, child))
    return edges
