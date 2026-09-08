from typing import Optional
import numpy as np
import re


class IntelligentQueryRouter:
    """Overview → global; everything else hybrid. Substring traps like 'quais são' do not fire."""

    OVERVIEW_PHRASES = (
        "tema principal",
        "visão geral",
        "visao geral",
        "ideia central",
        "main theme",
        "central idea",
        "key topics",
        "principais tópicos",
        "principais topicos",
    )
    OVERVIEW_WORDS = {"resumo", "resuma", "summarize", "summary", "overview"}
    COMPARE_PHRASES = (
        "em comum",
        "in common",
        "diferenças",
        "diferencas",
        "differences",
        "compare the",
    )
    FILLER = {"deste", "desta", "this", "that", "book", "texto", "text", "the"}

    @classmethod
    def route(cls, query: str) -> str:
        q = query.lower().strip()
        q_bare = q.strip("?.! ")
        tokens = q.split()
        n = len(tokens)

        if q_bare in cls.OVERVIEW_WORDS or q_bare in cls.OVERVIEW_PHRASES:
            return "global"
        if n <= 3:
            for phrase in cls.OVERVIEW_PHRASES:
                if phrase in q:
                    return "global"
            return "hybrid"

        overview_hit = any(p in q for p in cls.OVERVIEW_PHRASES)
        word_hit = any(re.search(rf"\b{re.escape(w)}\b", q) for w in cls.OVERVIEW_WORDS)
        compare_hit = any(p in q for p in cls.COMPARE_PHRASES) or bool(
            re.search(r"\bcompare\b", q)
        )
        if not (overview_hit or word_hit or compare_hit):
            return "hybrid"

        rest = q
        for phrase in cls.OVERVIEW_PHRASES + cls.COMPARE_PHRASES:
            rest = rest.replace(phrase, " ")
        for word in cls.OVERVIEW_WORDS | {"compare"}:
            rest = re.sub(rf"\b{re.escape(word)}\b", " ", rest)
        content = [
            t for t in rest.split() if len(t) > 3 and t not in cls.FILLER
        ]
        return "global" if len(content) <= 3 else "hybrid"

    _encoder = None
    _global_centroid: Optional[np.ndarray] = None
    _hybrid_centroid: Optional[np.ndarray] = None

    @classmethod
    def train_centroids(cls, encoder):
        """Train centroids with representative examples (call once at startup)."""
        cls._encoder = encoder
        global_examples = [
            "qual é o tema principal", "visão geral do documento",
            "resuma o conteúdo", "summarize the main ideas",
            "quais são os principais tópicos", "overview of the book",
        ]
        hybrid_examples = [
            "qual é a idade do autor", "when was the experiment conducted",
            "quem descobriu o elétron", "what is the melting point",
            "quantos participantes no estudo", "where did the battle happen",
        ]
        g_embs = encoder.encode(global_examples, normalize_embeddings=True)
        h_embs = encoder.encode(hybrid_examples, normalize_embeddings=True)
        cls._global_centroid = np.mean(g_embs, axis=0)
        cls._hybrid_centroid = np.mean(h_embs, axis=0)

    @classmethod
    def route_with_embeddings(cls, query: str) -> str:
        """Regex fast-path; if ambiguous, classify by embedding distance."""
        regex_result = cls.route(query)
        if cls._encoder is None or cls._global_centroid is None:
            return regex_result
        if len(query.split()) < 4:
            return regex_result
        q_emb = cls._encoder.encode(query, normalize_embeddings=True)
        sim_global = float(np.dot(q_emb, cls._global_centroid))
        sim_hybrid = float(np.dot(q_emb, cls._hybrid_centroid))
        if sim_global > sim_hybrid + 0.05:
            return "global"
        return "hybrid"
