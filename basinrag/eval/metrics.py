import math
from typing import List, Set, Dict

def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            # Assuming binary relevance (1 for relevant, 0 for not)
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(k, len(relevant_ids))):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    for rid in retrieved_ids[:k]:
        if rid in relevant_ids:
            return 1.0
    return 0.0

def evaluate_retrieval(qrels: Dict[str, Set[str]], results: Dict[str, List[str]], k: int = 10) -> Dict[str, float]:
    total = len(qrels)
    if total == 0:
        return {"mrr": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
        
    mrr_sum = 0.0
    ndcg_sum = 0.0
    hit_sum = 0.0
    
    for qid, rel_ids in qrels.items():
        retrieved = results.get(qid, [])
        mrr_sum += mrr_at_k(retrieved, rel_ids, k)
        ndcg_sum += ndcg_at_k(retrieved, rel_ids, k)
        hit_sum += hit_rate_at_k(retrieved, rel_ids, k)
        
    return {
        "mrr": mrr_sum / total,
        "ndcg": ndcg_sum / total,
        "hit_rate": hit_sum / total
    }
