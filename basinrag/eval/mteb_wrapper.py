import os
import time
from typing import Dict, List, Any
import numpy as np

class BasinRAGMTEBWrapper:
    """
    Wrapper para o MTEB que encapsula o BasinRAG.
    O MTEB verifica se o modelo possui o método 'search'. Se sim, ele não tentará extrair embeddings
    manualmente, e sim repassará o controle de busca (corpus, queries) para este método.
    """
    def __init__(self, rag_instance, search_type: str = "hybrid"):
        self.rag = rag_instance
        self.search_type = search_type

    def index(self, corpus: Dict[str, Dict[str, str]], *args, **kwargs):
        print(f"\n[MTEB Wrapper] Iniciando indexação de {len(corpus)} documentos no BasinRAG...")
        
        # 1. Indexar corpus no BasinRAG
        from basinrag.indexer.condensation import node_layers
        chunks = []
        if isinstance(corpus, dict):
            doc_ids = list(corpus.keys())
            doc_texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip() for did in doc_ids]
        else:
            # HuggingFace Dataset format
            doc_ids = corpus["id"] if "id" in corpus.column_names else corpus["_id"]
            titles = corpus["title"] if "title" in corpus.column_names else [""] * len(doc_ids)
            doc_texts = [f"{t} {x}".strip() for t, x in zip(titles, corpus["text"])]
        
        batch_size = 32
        print(f"Gerando embeddings para {len(doc_texts)} documentos...")
        
        # O BasinRAG usará o Qwen3-Embedding (que foi passado na config)
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            # Aqui geramos embeddings diretamente do encoder do BasinRAG
            embs = self.rag.ingestor.encoder.encode(batch_texts)
            for j, (text, doc_id, emb) in enumerate(zip(batch_texts, batch_ids, embs)):
                layers = node_layers(text)
                chunks.append({
                    "id": doc_id,
                    "text": text,
                    "embedding": emb,
                    "source": "mteb",
                    "chunk_index": 0,
                    "l1": layers["l1"],
                    "l2": layers["l2"],
                    "metadata": {"id": doc_id}
                })
        
        print("[MTEB Wrapper] Construindo Bacias de Atração...")
        self.rag.engine.build_graph(chunks)
        self.rag.engine.partition_into_basins()
        self.rag._attach_bm25()
        self.text_to_id = {text: docid for docid, text in zip(doc_ids, doc_texts)}

    def search(
        self, 
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        queries = kwargs.get("queries") or args[0]
        if not isinstance(queries, dict):
            q_ids = queries["id"] if "id" in queries.column_names else queries["_id"]
            queries = {q: t for q, t in zip(q_ids, queries["text"])}
        top_k = kwargs.get("top_k", 10)
        print(f"[MTEB Wrapper] Executando {len(queries)} queries ({self.search_type})...")
        results = {}
        for qid, qtext in queries.items():
            docs = self.rag.query(qtext, search_type=self.search_type, top_k=top_k)
            doc_scores = {}
            for rank, d in enumerate(docs):
                found_id = getattr(self, "text_to_id", {}).get(d.page_content, "")
                if found_id:
                    doc_scores[found_id] = 1.0 / (rank + 1.0)
            results[qid] = doc_scores
            
        return results

    @property
    def mteb_model_meta(self):
        import mteb
        return mteb.get_model_meta("BAAI/bge-small-en-v1.5")

class DenseOnlyMTEBWrapper:
    """Wrapper para rodar o modelo Dense (Qwen) PURO sem grafos no MTEB."""
    def __init__(self, encoder_model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(encoder_model_name, trust_remote_code=True)
        
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        return self.model.encode(sentences, **kwargs)

    @property
    def mteb_model_meta(self):
        from mteb.models.model_meta import ModelMeta
        return ModelMeta(name="DenseBaseline", languages=["eng"])
