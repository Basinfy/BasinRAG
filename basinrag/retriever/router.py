from typing import Optional
import numpy as np
import re


class IntelligentQueryRouter:
    """Overview → global; detail/context → local; everything else hybrid."""

    OVERVIEW_PHRASES = (
        # PT
        "tema principal", "visão geral", "visao geral", "ideia central", "principais tópicos", "principais topicos",
        # EN
        "main theme", "central idea", "key topics", "main points", "overview of", "executive summary",
        # ES
        "visión general", "vision general", "idea central", "tema central", "temas clave", "puntos principales",
        # FR
        "thème principal", "theme principal", "vue d'ensemble", "idée centrale", "idee centrale", "points clés", "points cles",
        # DE
        "hauptthema", "überblick", "ueberblick", "zentrale idee", "hauptpunkte", "kernaussage",
        # IT
        "tema principale", "panoramica", "punti principali", "argomento principale",
        # RU
        "главная тема", "обзор", "центральная идея", "основные темы", "ключевые моменты", "краткое содержание",
        # ZH
        "主要议题", "主要主题", "核心思想", "中心思想", "概览", "全局概览", "总结概述", "要点概括", "全文总结",
        # JA
        "主なテーマ", "中心的な考え", "全体の概要", "主なトピック", "全体のまとめ",
        # KO
        "주요 주제", "중심 생각", "전체 개요", "주요 요점", "핵심 내용", "전체 요약",
    )
    OVERVIEW_WORDS = {
        # PT
        "resumo", "resuma",
        # EN
        "summarize", "summary", "overview", "synopsis",
        # ES
        "resumen", "resume", "resumir",
        # FR
        "résumé", "résumer", "resumer", "aperçu", "apercu",
        # DE
        "zusammenfassung", "abstrakt",
        # IT
        "riassunto", "riassumere",
        # RU
        "резюме", "обобщение", "обобщить", "конспект",
        # ZH
        "总结", "概述", "概括", "综述",
        # JA
        "要約", "まとめ", "概略",
        # KO
        "요약", "개요", "줄거리",
    }
    COMPARE_PHRASES = (
        # PT
        "em comum", "diferenças", "diferencas", "comparar",
        # EN
        "in common", "differences", "compare the",
        # ES
        "en común", "en comun", "diferencias",
        # FR
        "en commun", "différences", "differences",
        # DE
        "gemeinsam", "unterschiede", "im vergleich",
        # IT
        "in comune", "differenze",
        # RU
        "в общем", "различия", "разница", "сравнить",
        # ZH
        "共同点", "相同点", "区别", "不同点", "对比",
        # JA
        "共通点", "違い", "比較", "対比",
        # KO
        "공통점", "차이점", "비교",
    )
    # Detail / neighborhood questions → local basin+PPR path
    LOCAL_PHRASES = (
        "neste parágrafo", "neste capitulo", "neste capítulo", "nessa seção", "nessa secao",
        "contexto imediato", "trecho sobre", "passagem sobre", "ao redor de",
        "in this paragraph", "in this section", "nearby context", "surrounding text",
        "in this chapter", "local context", "passage about",
    )
    FILLER = {
        # PT
        "deste", "desta", "livro", "texto", "artigo", "documento",
        # EN
        "this", "that", "book", "text", "the", "article", "document",
        # ES
        "este", "esta", "articulo",
        # FR
        "ce", "cet", "cette", "texte",
        # DE
        "dieses", "dieser", "buch",
        # IT
        "questo", "questa", "testo",
        # RU
        "этот", "эта", "это", "книга", "текст", "статья", "документ",
    }

    @staticmethod
    def _word_in_query(w: str, q: str) -> bool:
        """Matches word in query, supporting CJK without whitespace boundaries."""
        if any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af" for c in w):
            return w in q
        return bool(re.search(rf"\b{re.escape(w)}\b", q))

    @classmethod
    def route(cls, query: str) -> str:
        q = query.lower().strip()
        q_bare = q.strip("?.! ")
        tokens = q.split()
        n = len(tokens)

        if q_bare in cls.OVERVIEW_WORDS or q_bare in cls.OVERVIEW_PHRASES:
            return "global"

        if any(p in q for p in cls.LOCAL_PHRASES):
            return "local"

        is_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af" for c in q)
        if n <= 3 or (is_cjk and len(q) <= 12):
            for phrase in cls.OVERVIEW_PHRASES:
                if phrase in q:
                    return "global"
            for word in cls.OVERVIEW_WORDS:
                if cls._word_in_query(word, q):
                    return "global"
            return "hybrid"

        overview_hit = any(p in q for p in cls.OVERVIEW_PHRASES)
        word_hit = any(cls._word_in_query(w, q) for w in cls.OVERVIEW_WORDS)
        compare_hit = any(p in q for p in cls.COMPARE_PHRASES) or bool(
            re.search(r"\b(compare|comparar|vergleichen|confrontare|сравнить)\b", q)
        )
        if not (overview_hit or word_hit or compare_hit):
            return "hybrid"

        rest = q
        for phrase in cls.OVERVIEW_PHRASES + cls.COMPARE_PHRASES:
            rest = rest.replace(phrase, " ")
        for word in cls.OVERVIEW_WORDS:
            if any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af" for c in word):
                rest = rest.replace(word, " ")
            else:
                rest = re.sub(rf"\b{re.escape(word)}\b", " ", rest)
        rest = re.sub(r"\b(compare|comparar|vergleichen|confrontare|сравнить)\b", " ", rest)
        content = [
            t for t in rest.split() if len(t) > 3 and t not in cls.FILLER
        ]
        return "global" if len(content) <= 3 else "hybrid"

    _encoder = None
    _global_centroid: Optional[np.ndarray] = None
    _hybrid_centroid: Optional[np.ndarray] = None
    _local_centroid: Optional[np.ndarray] = None

    @classmethod
    def train_centroids(cls, encoder):
        """Train centroids with representative examples across 10 languages (call once at startup)."""
        cls._encoder = encoder
        global_examples = [
            # PT / EN
            "qual é o tema principal", "visão geral do documento",
            "resuma o conteúdo", "summarize the main ideas",
            "quais são os principais tópicos", "overview of the book",
            # ES / FR / DE / IT
            "cuál es la idea central del texto", "quel est le thème principal de ce livre",
            "was ist das hauptthema dieses dokuments", "qual è l'idea centrale di questo testo",
            # RU / ZH / JA / KO
            "в чем главная тема этой статьи", "这篇文档的核心思想是什么",
            "このドキュメントの主なテーマは何ですか", "이 문서의 주요 주제는 무엇입니까",
        ]
        hybrid_examples = [
            # PT / EN
            "qual é a idade do autor", "when was the experiment conducted",
            "quem descobriu o elétron", "what is the melting point",
            "quantos participantes no estudo", "where did the battle happen",
            # ES / FR / DE / IT
            "cuándo se realizó el experimento", "qui a découvert l'électron",
            "wie hoch ist der schmelzpunkt von kupfer", "quanti partecipanti hanno preso parte allo studio",
            # RU / ZH / JA / KO
            "в каком году произошло это событие", "钠的熔点是多少度",
            "電子を発見したのは誰ですか", "구리의 녹는점은 얼마입니까",
        ]
        local_examples = [
            "neste parágrafo o que significa", "contexto imediato deste trecho",
            "in this section what does it say about", "passage about the method nearby",
            "ao redor desta menção qual é o argumento", "local context of this claim",
        ]
        g_embs = encoder.encode(global_examples, normalize_embeddings=True)
        h_embs = encoder.encode(hybrid_examples, normalize_embeddings=True)
        l_embs = encoder.encode(local_examples, normalize_embeddings=True)
        cls._global_centroid = np.mean(g_embs, axis=0)
        cls._global_centroid /= np.linalg.norm(cls._global_centroid) + 1e-10
        cls._hybrid_centroid = np.mean(h_embs, axis=0)
        cls._hybrid_centroid /= np.linalg.norm(cls._hybrid_centroid) + 1e-10
        cls._local_centroid = np.mean(l_embs, axis=0)
        cls._local_centroid /= np.linalg.norm(cls._local_centroid) + 1e-10

    @classmethod
    def route_with_embeddings(cls, query: str) -> str:
        """Regex fast-path; if ambiguous, classify by embedding distance."""
        regex_result = cls.route(query)
        if cls._encoder is None or cls._global_centroid is None:
            return regex_result
        if len(query.split()) < 4:
            return regex_result
        if regex_result in ("global", "local"):
            return regex_result
        q_emb = cls._encoder.encode(query, normalize_embeddings=True)
        sim_global = float(np.dot(q_emb, cls._global_centroid))
        sim_hybrid = float(np.dot(q_emb, cls._hybrid_centroid))
        sim_local = float(np.dot(q_emb, cls._local_centroid)) if cls._local_centroid is not None else -1.0
        if sim_global > sim_hybrid + 0.05 and sim_global >= sim_local:
            return "global"
        if sim_local > sim_hybrid + 0.05 and sim_local > sim_global:
            return "local"
        return "hybrid"
