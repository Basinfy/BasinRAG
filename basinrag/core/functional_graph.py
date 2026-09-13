"""Functional-graph collapse for document chunks.

Each chunk has one successor (φ): the previous chunk in the same source
section. Semantic kNN edges stay virtual and never define φ — maintaining strict
separation between primary topological flow and auxiliary semantic synapses.

Attractors are section-start sinks. Hops are reverse-BFS distance along φ.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    embeddings_by_id: Dict[str, Any],
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
    embeddings_by_id: Dict[str, Any],
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
    """Map every node to its attractor using Floyd's tortoise-and-hare algorithm.

    O(|V|) time and O(1) auxiliary space per chain (vs O(|V|) stack in DFS).
    Cycles are canonicalized by lexicographic minimum for stable attractor IDs.
    """
    resolved: Dict[str, str] = {}

    for start in successor:
        if start in resolved:
            continue

        # Phase 1: Follow chain, collecting path, until we reach a resolved
        # node, a sink (None), or revisit a node on the current path.
        path: List[str] = []
        path_set: Dict[str, int] = {}  # node -> index in path
        curr: Optional[str] = start

        while curr is not None and curr not in resolved and curr not in path_set:
            path_set[curr] = len(path)
            path.append(curr)
            curr = successor.get(curr)

        if curr is None:
            # Sink: terminal of the chain is the attractor
            attr = path[-1] if path else start
        elif curr in resolved:
            # Reached an already-resolved node
            attr = resolved[curr]
        else:
            # Cycle detected: curr is revisited on the current path
            cycle_start_idx = path_set[curr]
            cycle = path[cycle_start_idx:]
            # Canonical rotation: lexicographically smallest node
            attr = min(cycle)

        # Assign attractor to all nodes in the path
        for nid in path:
            resolved[nid] = attr

    return resolved

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
