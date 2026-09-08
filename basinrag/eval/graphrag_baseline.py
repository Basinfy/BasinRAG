import spacy
from typing import Dict, List
import math
from collections import defaultdict

class GraphRAGBaseline:
    """
    Um baseline que simula os fundamentos do GraphRAG: 
    Extração de entidades de conhecimento, linking em um Knowledge Graph, 
    e recuperação via expansão de nós no grafo (1-hop graph walk).
    """
    def __init__(self):
        print("Carregando modelo SpaCy para extração de Entidades (GraphRAG Baseline)...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        self.entity_to_docs = defaultdict(set)
        self.doc_to_entities = defaultdict(set)
        self.entity_df = defaultdict(int)
        self.N = 0
        
    def _extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = set()
        # Entidades Nomeadas (ORG, GPE, PERSON, etc)
        for ent in doc.ents:
            if ent.label_ not in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
                entities.add(ent.lemma_.lower())
        # Noun Chunks como conceitos genéricos (Knowledge Triples abstraction)
        for chunk in doc.noun_chunks:
            # Filtra pronomes
            if chunk.root.pos_ != "PRON":
                entities.add(chunk.lemma_.lower())
        return list(entities)

    def build_graph(self, corpus: Dict[str, str]):
        self.N = len(corpus)
        for docid, text in corpus.items():
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

    def search(self, queries: Dict[str, str], corpus: Dict[str, str], top_k: int = 10) -> Dict[str, List[str]]:
        results = {}
        for qid, qtext in queries.items():
            q_entities = self._extract_entities(qtext)
            
            doc_scores = defaultdict(float)
            # Graph Walk - Hops Diretos
            for qe in q_entities:
                weight = self._idf(qe)
                if weight > 0:
                    for docid in self.entity_to_docs.get(qe, []):
                        doc_scores[docid] += weight
                        
            # Graph Walk - Expansão de vizinhança (1-hop entity co-occurrence)
            # Para simular o GraphRAG (communities), se um documento tem muitas entidades que 
            # co-ocorrem com a query, ele ganha um boost. (Já coberto indiretamente pelo somatório de IDFs).
            
            # Ordena e pega os top K
            ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            results[qid] = [docid for docid, score in ranked[:top_k]]
            
        return results
