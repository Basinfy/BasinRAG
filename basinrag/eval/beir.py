import os
import json
import urllib.request
import zipfile
from typing import Dict, Tuple, List

BEIR_DATASETS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"

def download_and_unzip_beir_dataset(dataset_name: str, out_dir: str = ".basinrag/eval_cache") -> str:
    """Baixa um dataset padrao do BEIR e o descompacta."""
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"{dataset_name}.zip")
    extract_dir = os.path.join(out_dir, dataset_name)
    
    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            url = f"{BEIR_DATASETS_URL}{dataset_name}.zip"
            print(f"Baixando dataset BEIR '{dataset_name}' de {url} ...")
            with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            
        print(f"Descompactando {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            dest_abs = os.path.abspath(out_dir)
            for member in zip_ref.infolist():
                target = os.path.abspath(os.path.join(out_dir, member.filename))
                if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
                    raise ValueError(f"zip-slip rejeitado: {member.filename}")
            zip_ref.extractall(out_dir)
            
    return extract_dir

def load_beir_dataset(dataset_dir: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    """Carrega o corpus, queries e qrels de um diretorio BEIR descompactado."""
    corpus = {}
    with open(os.path.join(dataset_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            # Concatena titulo e texto (padrao BEIR)
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            corpus[doc["_id"]] = text
            
    queries = {}
    with open(os.path.join(dataset_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
            
    qrels = {}
    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")
    if os.path.exists(qrels_path):
        with open(qrels_path, "r", encoding="utf-8") as f:
            next(f)  # Pula o header (query-id \t corpus-id \t score)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, docid, score = parts[0], parts[1], int(parts[2])
                    if score > 0:
                        if qid not in qrels:
                            qrels[qid] = set()
                        qrels[qid].add(docid)
                        
    return corpus, queries, qrels

class BasinRAGBEIRAdapter:
    """Adaptador para encapsular o BasinRAG na API oficial do BEIR."""
    def __init__(self, rag_instance):
        self.rag = rag_instance

    def search_beir(self, queries: Dict[str, str], top_k: int = 10, search_type: str = "hybrid") -> Dict[str, Dict[str, float]]:
        """Retorna formato oficial do BEIR: {qid: {doc_id: score}}."""
        results = {}
        total = len(queries)
        for i, (qid, qtext) in enumerate(queries.items(), 1):
            if i % 5 == 0 or i == 1 or i == total:
                print(f"[BasinRAG BEIR] Processando query {i}/{total} ({i/total*100:.0f}%)...", flush=True)
            docs = self.rag.query(qtext, search_type=search_type, top_k=top_k * 2)
            retrieved_scores = {}
            for rank, d in enumerate(docs):
                meta = getattr(d, "metadata", {}) or {}
                found_id = meta.get("doc_id") or meta.get("source") or meta.get("node_id") or ""
                if found_id and found_id not in retrieved_scores:
                    # Score decrescente calibrado por rank se score não estiver disponível
                    score = float(getattr(d, "score", 0.0) or (1.0 / (rank + 1)))
                    retrieved_scores[found_id] = score
                if len(retrieved_scores) >= top_k:
                    break
            results[qid] = retrieved_scores
        return results

    def search(self, queries: Dict[str, str], corpus: Dict[str, str], search_type: str = "hybrid", top_k: int = 10) -> Dict[str, List[str]]:
        results = {}
        for qid, qtext in queries.items():
            docs = self.rag.query(qtext, search_type=search_type, top_k=top_k * 2)
            retrieved_ids = []
            seen_docs = set()
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                found_id = meta.get("doc_id") or meta.get("source") or meta.get("node_id") or ""
                if found_id and found_id in corpus and found_id not in seen_docs:
                    seen_docs.add(found_id)
                    retrieved_ids.append(found_id)
                if len(retrieved_ids) >= top_k:
                    break
            results[qid] = retrieved_ids
        return results


def ingest_beir_corpus(rag, corpus: Dict[str, str], batch_size: int = 128) -> int:
    """Ingere um corpus arbitrário do BEIR diretamente no grafo e bacias topológicas do BasinRAG."""
    from ..indexer.condensation import node_layers

    doc_ids = list(corpus.keys())
    texts = [corpus[doc_id] for doc_id in doc_ids]

    if not texts:
        return 0

    embeddings = rag.ingestor.encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    nodes = []
    for doc_id, text, emb in zip(doc_ids, texts, embeddings):
        layers = node_layers(text)
        nodes.append({
            "id": doc_id,
            "text": text,
            "embedding": emb,
            "source": doc_id,
            "chunk_index": 0,
            "l1": layers["l1"],
            "l2": layers["l2"],
            "metadata": {"doc_id": doc_id},
        })

    rag.engine.encoder_model = rag.config.encoder_model
    rag.engine.build_graph(nodes)
    rag.engine.partition_into_basins()
    rag._attach_bm25()
    rag.retriever = None

    return len(nodes)


