from typing import List
from ..logging_config import setup_logging

logger = setup_logging()


class CrossEncoderReranker:
    """Multilingual MiniLM cross-encoder; skip when the model cannot load."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, trust_remote_code=True)
            except Exception as e:
                logger.info(f"Aviso: Falha ao carregar o reranker '{self.model_name}' ({e}). Fazendo fallback para ms-marco-MiniLM...")
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                except Exception:
                    logger.info("Aviso: CrossEncoder indisponivel. Reranker desativado.")
                    self._model = "disabled"

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if not documents:
            return []
        self._load_model()
        if self._model == "disabled":
            return documents[:top_k]
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_score_pairs[:top_k]]
