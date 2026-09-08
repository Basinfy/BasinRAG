import re
import spacy
from typing import Dict, List, Set, Tuple, Optional
import math
from collections import defaultdict

class GraphRAGBaseline:
    """
    Baseline que simula os fundamentos do GraphRAG / HippoRAG:
    Extração de entidades de conhecimento e símbolos de código,
    linking em um Knowledge Graph bipartido (Entidade <-> Documento),
    e recuperação via Graph Walk com propagação de ativação (multi-hop).
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self.doc_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self.entity_df: Dict[str, int] = defaultdict(int)
        self.N: int = 0
        self._stopwords = {
            "def", "class", "return", "import", "from", "self", "none", "true", "false",
            "and", "or", "not", "for", "while", "with", "as", "if", "elif", "else",
            "try", "except", "finally", "pass", "lambda", "yield", "raise", "in", "is"
        }

    def _extract_entities(self, text: str) -> List[str]:
        entities: Set[str] = set()

        # 1. Entidades de Código e Símbolos (CamelCase, snake_case, identificadores)
        code_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
        for tok in code_tokens:
            tok_lower = tok.lower()
            if tok_lower not in self._stopwords:
                entities.add(tok_lower)
                # Decomposição de camelCase e snake_case para capturar conceitos semânticos
                parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", tok)
                for part in parts:
                    p_low = part.lower()
                    if len(p_low) >= 3 and p_low not in self._stopwords:
                        entities.add(p_low)

        # 2. Entidades Nomeadas e Noun Chunks via SpaCy (texto natural, docstrings, comentários)
        # Limita tamanho do texto para evitar estouro de memória no SpaCy
        sample_text = text[:1500]
        try:
            doc = self.nlp(sample_text)
            for ent in doc.ents:
                if ent.label_ not in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
                    lemma = ent.lemma_.lower().strip()
                    if len(lemma) >= 3 and lemma not in self._stopwords:
                        entities.add(lemma)
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ != "PRON":
                    lemma = chunk.lemma_.lower().strip()
                    if len(lemma) >= 3 and lemma not in self._stopwords:
                        entities.add(lemma)
        except Exception:
            pass

        return list(entities)

    def build_graph(self, corpus: Dict[str, str]):
        """Constrói o Grafo Bipartido de Conhecimento Entidades <-> Documentos/Chunks."""
        self.entity_to_docs.clear()
        self.doc_to_entities.clear()
        self.entity_df.clear()
        self.N = len(corpus)

        total = len(corpus)
        for idx, (docid, text) in enumerate(corpus.items(), 1):
            if idx % 1000 == 0 or idx == 1 or idx == total:
                print(f"[GraphRAG Build] Indexando documento {idx}/{total} ({idx/total*100:.0f}%)...", flush=True)
            entities = self._extract_entities(text)
            for e in entities:
                self.entity_to_docs[e].add(docid)
                self.doc_to_entities[docid].add(e)

        for e, docs in self.entity_to_docs.items():
            self.entity_df[e] = len(docs)

    def _idf(self, entity: str) -> float:
        df = self.entity_df.get(entity, 0)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search_single(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Executa Graph Walk para uma query:
        1. Localiza entidades da query no grafo e computa ativação direta via IDF.
        2. Realiza expansão 2-hop por co-ocorrência de entidades (Knowledge Graph Walk).
        """
        q_entities = self._extract_entities(query_text)
        direct_docs: Dict[str, float] = defaultdict(float)

        # 1-Hop: Ativação direta dos nós de documentos
        for qe in q_entities:
            weight = self._idf(qe)
            if weight > 0:
                for docid in self.entity_to_docs.get(qe, set()):
                    direct_docs[docid] += weight

        if not direct_docs:
            return []

        doc_scores = defaultdict(float, direct_docs)

        # 2-Hop: Propagação de ativação sobre o grafo de co-ocorrência de entidades
        for docid, direct_score in list(direct_docs.items()):
            if direct_score <= 0:
                continue
            for neighbor_entity in self.doc_to_entities.get(docid, set()):
                ent_weight = self._idf(neighbor_entity)
                if ent_weight > 0.5:
                    decay = 0.05 * (ent_weight / (1.0 + self.entity_df.get(neighbor_entity, 1)))
                    for co_doc in self.entity_to_docs.get(neighbor_entity, set()):
                        if co_doc != docid:
                            doc_scores[co_doc] += direct_score * decay

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_beir(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """Retorna formato oficial do BEIR: {qid: {doc_id: score}}."""
        results = {}
        total = len(queries)
        for idx, (qid, qtext) in enumerate(queries.items(), 1):
            if idx % 5 == 0 or idx == 1 or idx == total:
                print(f"[GraphRAG BEIR] Processando query {idx}/{total} ({idx/total*100:.0f}%)...", flush=True)
            ranked = self.search_single(qtext, top_k=top_k)
            results[qid] = {docid: float(score) for docid, score in ranked}
        return results

    def search(self, queries: Dict[str, str], corpus: Dict[str, str], top_k: int = 10) -> Dict[str, List[str]]:
        """Compatibilidade com o protocolo de avaliação do BEIR."""
        results = {}
        for qid, qtext in queries.items():
            ranked = self.search_single(qtext, top_k=top_k)
            results[qid] = [docid for docid, _ in ranked]
        return results
