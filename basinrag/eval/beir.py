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
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            
        print(f"Descompactando {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
    """Adaptador para encapsular o BasinRAG em uma API similar ao BEIR para testes imparciais."""
    def __init__(self, rag_instance):
        self.rag = rag_instance
        
    def search(self, queries: Dict[str, str], corpus: Dict[str, str], search_type: str = "hybrid", top_k: int = 10) -> Dict[str, List[str]]:
        results = {}
        # Invert corpus for fast lookup by text
        text_to_id = {text: docid for docid, text in corpus.items()}
        for qid, qtext in queries.items():
            docs = self.rag.query(qtext, search_type=search_type, top_k=top_k)
            # Retorna os IDs baseados no texto ( BasinRAG descarta IDs na saída )
            retrieved_ids = []
            for d in docs:
                found_id = text_to_id.get(d.page_content, "")
                if found_id:
                    retrieved_ids.append(found_id)
            results[qid] = retrieved_ids
        return results
