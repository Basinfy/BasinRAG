"""Tests for multilingual support across the top 10 global development languages:
1. English (EN)
2. Portuguese (PT)
3. Spanish (ES)
4. Simplified Chinese (ZH)
5. Japanese (JA)
6. German (DE)
7. French (FR)
8. Russian (RU)
9. Korean (KO)
10. Italian (IT)
"""
import os
import tempfile
from basinrag.core.ids import tokenize
from basinrag.indexer.bm25 import detect_language, stem_token, BM25Index
from basinrag.retriever.router import IntelligentQueryRouter


SAMPLE_TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog in the sunny park.",
    "pt": "A rápida raposa marrom pula sobre o cão preguiçoso no parque.",
    "es": "El rápido zorro marrón salta sobre el perro perezoso en el parque.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux dans le parc.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund im Park.",
    "it": "La rapida volpe marrone salta sopra il cane pigro nel parco.",
    "ru": "Быстрая коричневая лиса прыгает через ленивую собаку в парке.",
    "zh": "敏捷的棕色狐狸在阳光明媚的公园里跳过懒狗。",
    "ja": "素早い茶色のキツネは日当たりの良い公園で怠惰な犬を飛び越えます。",
    "ko": "빠른 갈색 여우가 햇살 가득한 공원에서 게으른 개를 뛰어넘습니다.",
}


def test_tokenize_all_10_languages():
    """Verify that every language generates non-empty, correct tokens."""
    for lang, text in SAMPLE_TEXTS.items():
        tokens = tokenize(text)
        assert len(tokens) >= 3, f"Tokenization failed or was too sparse for {lang}: {tokens}"

    # Verify specific scripts
    assert "привет" in tokenize("Привет, мир!")
    assert "мир" in tokenize("Привет, мир!")
    assert "总" in tokenize("总结文档")
    assert "结" in tokenize("总结文档")
    assert "キ" in tokenize("キツネ") or "ツ" in tokenize("キツネ")
    assert "여" in tokenize("여우") and "우" in tokenize("여우")


def test_bm25_language_detection():
    """Verify language detection across all 10 languages."""
    for lang, text in SAMPLE_TEXTS.items():
        detected = detect_language(text)
        assert detected == lang, f"Language mismatch for {lang}: detected {detected}"


def test_bm25_stemming():
    """Verify stemmers for European languages and pass-through for CJK."""
    # Portuguese
    assert stem_token("computadores", "pt") != "computadores"
    # English
    assert stem_token("running", "en") == "run"
    # Spanish
    assert stem_token("computadoras", "es") != "computadoras"
    # French
    assert stem_token("chanteurs", "fr") != "chanteurs"
    # German
    assert stem_token("häuser", "de") != "häuser"
    # Italian
    assert stem_token("ragazzi", "it") != "ragazzi"
    # Russian
    assert stem_token("программирование", "ru") != "программирование"

    # CJK pass-through (no mutilation)
    assert stem_token("总结", "zh") == "总结"
    assert stem_token("キツネ", "ja") == "キツネ"
    assert stem_token("여우", "ko") == "여우"


def test_bm25_multilingual_indexing_and_scoring():
    """Verify indexing and retrieving documents in all 10 languages."""
    index = BM25Index()
    doc_ids = list(SAMPLE_TEXTS.keys())
    texts = list(SAMPLE_TEXTS.values())
    index.build(doc_ids, texts)

    # Test queries in each language
    queries = {
        "en": "brown fox lazy dog",
        "pt": "raposa marrom cão",
        "es": "zorro marrón perro",
        "fr": "renard brun chien",
        "de": "braune Fuchs Hund",
        "it": "volpe marrone cane",
        "ru": "коричневая лиса собаку",
        "zh": "棕色 狐狸 懒狗",
        "ja": "茶色 キツネ 犬",
        "ko": "갈색 여우 개",
    }

    for lang, query in queries.items():
        results = index.score(query, top_k=3)
        assert len(results) > 0, f"No results for query '{query}' in {lang}"
        top_doc, score = results[0]
        assert top_doc == lang, f"Expected {lang} to be top-1 for query '{query}', got {top_doc} (score: {score})"


def test_router_multilingual_overview_and_hybrid():
    """Verify routing logic for global overview vs specific facts across 10 languages."""
    overview_queries = [
        ("qual é o tema principal do livro", "global"),
        ("summarize the main ideas of this document", "global"),
        ("cuál es la idea central de este documento", "global"),
        ("donnez-moi un résumé de ce texte", "global"),
        ("was ist das hauptthema dieses dokuments", "global"),
        ("qual è il tema principale di questo testo", "global"),
        ("в чем главная тема этой книги", "global"),
        ("总结这篇文档的核心思想", "global"),
        ("このドキュメントの主なテーマは何ですか", "global"),
        ("이 문서의 주요 주제는 무엇입니까", "global"),
    ]

    hybrid_queries = [
        ("quando nasceu Dom Pedro II", "hybrid"),
        ("what is the boiling point of water", "hybrid"),
        ("en qué año comenzó la segunda guerra", "hybrid"),
        ("qui a peint la Joconde", "hybrid"),
        ("wann wurde Albert Einstein geboren", "hybrid"),
        ("chi ha inventato il telefono", "hybrid"),
        ("когда родился Пушкин", "hybrid"),
        ("水在标准大气压下的沸点是多少", "hybrid"),
        ("アインシュタインはいつ生まれましたか", "hybrid"),
        ("물의 끓는점은 몇 도입니까", "hybrid"),
    ]

    for q, expected in overview_queries + hybrid_queries:
        res = IntelligentQueryRouter.route(q)
        assert res == expected, f"Query '{q}' routed to '{res}', expected '{expected}'"


def test_bm25_save_and_load_multilingual():
    """Verify that BM25 index with multilingual tokens survives JSON serialization."""
    index = BM25Index()
    doc_ids = ["zh_doc", "ru_doc"]
    texts = ["敏捷的棕色狐狸", "Быстрая коричневая лиса"]
    index.build(doc_ids, texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "bm25_test.json")
        index.save(save_path, build_id="test-build-123")

        loaded = BM25Index()
        assert loaded.load(save_path) is True
        assert loaded.build_id == "test-build-123"
        assert loaded.n == 2
        assert "zh_doc" in loaded.doc_ids
        assert "ru_doc" in loaded.doc_ids

        zh_res = loaded.score("狐狸", top_k=1)
        assert len(zh_res) == 1 and zh_res[0][0] == "zh_doc"

        ru_res = loaded.score("лиса", top_k=1)
        assert len(ru_res) == 1 and ru_res[0][0] == "ru_doc"
