from typing import List, Sequence, Tuple
from ..logging_config import setup_logging

logger = setup_logging()


class CrossEncoderReranker:
    """Cross-encoder reranker (defaults to BAAI/bge-reranker-v2-m3); skip when the model cannot load."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is None:
            import sys
            try:
                import torch.distributed._tensor as _t
                sys.modules.setdefault("torch.distributed.tensor", _t)
            except Exception:
                pass

            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, max_length=self.max_length, trust_remote_code=True)
            except Exception as e:
                logger.info(f"Aviso: Falha ao carregar o reranker '{self.model_name}' ({e}). Fazendo fallback para ms-marco-MiniLM...")
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=self.max_length)
                except Exception:
                    logger.info("Aviso: CrossEncoder indisponivel. Reranker desativado.")
                    self._model = "disabled"

    def predict_scores(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        self._load_model()
        if self._model == "disabled":
            return [float(1.0 / (idx + 1)) for idx in range(len(documents))]
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        scored = self.rerank_with_scores(query, documents, top_k=top_k)
        return [doc for doc, _ in scored]

    def rerank_items(
        self,
        query: str,
        items: Sequence[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, str]]:
        """Rerank (node_id, text) pairs by cross-encoder score; ids stay attached."""
        if not items:
            return []
        texts = [text for _, text in items]
        scores = self.predict_scores(query, texts)
        ranked = sorted(zip(items, scores), key=lambda x: float(x[1]), reverse=True)
        return [item for item, _ in ranked[:top_k]]

    def rerank_with_scores(self, query: str, documents: List[str], top_k: int = 5) -> List[tuple[str, float]]:
        if not documents:
            return []
        scores = self.predict_scores(query, documents)
        doc_score_pairs = [(doc, float(s)) for doc, s in zip(documents, scores)]
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]
