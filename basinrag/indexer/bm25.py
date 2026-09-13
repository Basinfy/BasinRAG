"""CSR-style BM25 (Robertson/Sparck Jones) with optional disk persistence.

Calibrated hyperparameters: k1=1.2, b=0.75.
"""
from __future__ import annotations

import json
import math
import os
from importlib.metadata import PackageNotFoundError, version as package_version
from threading import local
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.ids import tokenize

K1 = 1.2
B = 0.75

_STOP = {
    # Portuguese
    "a", "o", "os", "as", "um", "uma", "de", "da", "do", "das", "dos", "e", "ou",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "que", "se", "nao", "não",
    # English
    "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "as", "at", "by", "an", "this", "that", "it", "from", "with",
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", "en", "para",
    "por", "con", "que", "se", "no", "y", "o", "pero",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "pour", "par", "avec",
    "que", "qui", "ne", "pas", "et", "ou", "sur", "ce", "cette",
    # German
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "einen",
    "und", "oder", "in", "im", "für", "mit", "von", "zu", "ist", "sind", "nicht", "auf",
    # Italian
    "il", "la", "lo", "i", "gli", "le", "un", "uno", "una", "di", "del", "della",
    "in", "nel", "nella", "per", "con", "che", "non", "e", "ed", "sono",
    # Russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "до", "вас", "там", "потом", "себя", "для",
    # CJK single particles / stop tokens
    "的", "了", "和", "是", "就", "都", "而", "及", "與", "着",
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "이", "그", "저", "것", "수", "등", "들", "및", "에", "와", "과",
}

_STOP_BY_LANG = {
    "pt": set("a o os as um uma de da do das dos e ou em no na nos nas para por com que se nao não".split()),
    "en": set("the and or of to in on for is are was were be as at by an this that it from with".split()),
    "es": set("el la los las un una unos unas del al en para por con que se no y o pero".split()),
    "fr": set("le la les un une des du de dans pour par avec que qui ne pas et ou sur ce cette".split()),
    "de": set("der die das den dem des ein eine einer einem einen und oder in im für mit von zu ist sind nicht auf".split()),
    "it": set("il la lo i gli le un uno una di del della in nel nella per con che non e ed sono".split()),
    "ru": set("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только".split()),
    "zh": set("的 了 和 是 就 都 而 及 與 着".split()),
    "ja": set("の に は を た が で て と し れ さ".split()),
    "ko": set("이 그 저 것 수 등 들 및 에 와 과".split()),
}

_LANG_TO_SNOWBALL = {
    "en": "english",
    "pt": "portuguese",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ru": "russian",
}

_STEMMERS = local()


def current_stemmer_version(enabled: bool) -> str:
    if not enabled:
        return "disabled"
    try:
        return f"snowballstemmer-{package_version('snowballstemmer')}"
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "BM25 stemming requires the optional dependency; install basinrag[nlp]."
        ) from exc


def _get_stemmer(lang: str):
    cache = getattr(_STEMMERS, "cache", None)
    if cache is None:
        cache = {}
        _STEMMERS.cache = cache
    if lang in cache:
        return cache[lang]
    if lang in ("zh", "ja", "ko"):
        cache[lang] = None
        return None
    try:
        import snowballstemmer

        snowball_name = _LANG_TO_SNOWBALL.get(lang)
        if snowball_name:
            stemmer = snowballstemmer.stemmer(snowball_name)
            cache[lang] = stemmer
            return stemmer
    except Exception:
        pass
    cache[lang] = False
    return False


def _apply_stemmer(stemmer: Any, token: str) -> str:
    stem_word = getattr(stemmer, "stemWord", None)
    if callable(stem_word):
        return str(stem_word(token))
    stem = getattr(stemmer, "stem", None)
    if callable(stem):
        return str(stem(token))
    return token


_PT_INDICATORS = {
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "que", "se", "nao", "não", "um", "uma",
    "os", "as", "ou", "como", "mais", "mas", "ao", "aos", "este",
    "esta", "esse", "essa", "isso", "isto", "pelo", "pela", "sobre",
}
_EN_INDICATORS = {
    "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "as", "at", "by", "an", "this",
    "that", "it", "from", "with", "which", "there", "have", "has",
}
_ES_INDICATORS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del",
    "al", "en", "para", "por", "con", "que", "como", "pero", "este",
    "esta", "estos", "estas", "sobre", "entre", "también", "tambien",
}
_FR_INDICATORS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "dans",
    "pour", "par", "avec", "sur", "est", "sont", "cette", "ce",
    "ces", "qui", "que", "mais", "aussi",
}
_DE_INDICATORS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "einem", "einen", "und", "oder", "in", "im", "für", "mit",
    "von", "zu", "ist", "sind", "nicht", "auf", "auch",
}
_IT_INDICATORS = {
    "il", "la", "lo", "i", "gli", "le", "un", "uno", "una", "del",
    "della", "dei", "degli", "delle", "in", "nel", "nella", "per",
    "con", "che", "sono", "questo", "questa", "anche",
}


def detect_language(text: str, default: Optional[str] = None) -> Optional[str]:
    """Detect language across 10 supported languages (EN, PT, ES, ZH, JA, DE, FR, RU, KO, IT)."""
    if not text:
        return default
    lower = text.lower()

    # 1. Non-Latin scripts
    if any("\u0400" <= c <= "\u04ff" for c in lower):
        return "ru"
    if any(("\uac00" <= c <= "\ud7af") or ("\u1100" <= c <= "\u11ff") for c in lower):
        return "ko"
    if any(("\u3040" <= c <= "\u309f") or ("\u30a0" <= c <= "\u30ff") for c in lower):
        return "ja"
    if any("\u4e00" <= c <= "\u9fff" for c in lower):
        return "zh"

    # 2. Distinctive diacritics
    if any(c in lower for c in "ñ¿¡"):
        return "es"
    if any(c in lower for c in "äöüß"):
        return "de"
    if any(c in lower for c in "œæëïù"):
        return "fr"
    if any(c in lower for c in "çãõ"):
        return "pt"

    # 3. Lexical indicators for Latin scripts
    words = set(tokenize(lower))
    scores = {
        "pt": len(words & _PT_INDICATORS),
        "en": len(words & _EN_INDICATORS),
        "es": len(words & _ES_INDICATORS),
        "fr": len(words & _FR_INDICATORS),
        "de": len(words & _DE_INDICATORS),
        "it": len(words & _IT_INDICATORS),
    }
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score > 0:
        return best_lang

    if any(c in lower for c in "áéíóúâêôà"):
        return "pt"

    return default


def stem_token(tok: str, lang: Optional[str] = None) -> str:
    if not tok:
        return ""
    tok_lower = tok.lower()
    if lang in ("zh", "ja", "ko"):
        return tok_lower
    if len(tok) <= 3:
        return tok_lower

    if lang:
        stemmer = _get_stemmer(lang)
        if stemmer:
            try:
                return _apply_stemmer(stemmer, tok_lower)
            except Exception:
                return tok_lower
        return tok_lower

    # Uncertain language means no language-specific stemming.
    return tok_lower



class BM25Index:
    def __init__(self, *, stemming: bool = False) -> None:
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.n: int = 0
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.build_id: str = ""
        self.corpus_lang: str = "pt"
        self.stemming = bool(stemming)
        self.stemmer_version = current_stemmer_version(self.stemming)

    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_len = []
        self.df = {}
        self.postings = {}
        self.n = len(self.doc_ids)
        if self.n == 0:
            self.avgdl = 0.0
            return

        lang_counts: Dict[str, int] = {}
        for i, text in enumerate(texts):
            lang = detect_language(text)
            if lang is not None:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            tf: Dict[str, int] = {}
            for tok in tokenize(text):
                if lang is not None and tok in _STOP_BY_LANG.get(lang, set()):
                    continue
                if self.stemming and lang is not None:
                    tok = stem_token(tok, lang=lang)
                tf[tok] = tf.get(tok, 0) + 1
            length = sum(tf.values()) or 1
            self.doc_len.append(length)
            for term, count in tf.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.postings.setdefault(term, []).append((i, count))

        if lang_counts:
            self.corpus_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
        else:
            self.corpus_lang = "pt"
        self.avgdl = sum(self.doc_len) / self.n

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.n == 0:
            return []
        acc: Dict[int, float] = {}
        default_lang = getattr(self, "corpus_lang", "pt")
        lang = detect_language(query)
        if lang is None and default_lang:
            # Uncertain language means no language-specific filtering or stemming.
            lang = None
        for term in tokenize(query):
            if lang is not None and term in _STOP_BY_LANG.get(lang, set()):
                continue
            if self.stemming and lang is not None:
                term = stem_token(term, lang=lang)
            if term not in self.postings:
                continue
            idf = self._idf(term)
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = tf + K1 * (1.0 - B + B * dl / (self.avgdl or 1.0))
                acc[doc_idx] = acc.get(doc_idx, 0.0) + idf * (tf * (K1 + 1.0) / denom)
        ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked]

    def save(self, path: str, build_id: str = "") -> None:
        payload = {
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "n": self.n,
            "df": self.df,
            "postings": {t: pairs for t, pairs in self.postings.items()},
            "buildId": build_id or self.build_id,
            "corpus_lang": getattr(self, "corpus_lang", "pt"),
            "stemming": self.stemming,
            "stemmer_version": self.stemmer_version,
        }
        self.build_id = str(payload["buildId"])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.doc_ids = payload.pop("doc_ids")
        self.doc_len = payload.pop("doc_len")
        self.avgdl = float(payload.pop("avgdl"))
        self.n = int(payload.pop("n"))
        self.df = {k: int(v) for k, v in payload.pop("df").items()}
        raw_postings = payload.pop("postings")
        self.postings = {
            t: [(int(i), int(tf)) for i, tf in pairs]
            for t, pairs in raw_postings.items()
        }
        del raw_postings
        self.build_id = payload.get("buildId", "") or ""
        self.corpus_lang = payload.get("corpus_lang", "pt")
        self.stemming = bool(payload.get("stemming", False))
        self.stemmer_version = payload.get("stemmer_version", "disabled")
        del payload
        return True
