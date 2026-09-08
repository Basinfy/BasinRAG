import os
import json
import urllib.request
from typing import Dict, List, Tuple

FAQUAD_DEV_URL = "https://raw.githubusercontent.com/liafacom/faquad/master/data/dev.json"

def load_faquad(cache_dir: str = ".basinrag/eval_cache") -> Dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "faquad_dev.json")
    
    if not os.path.exists(cache_path):
        print("Baixando FaQuAD dev set...")
        req = urllib.request.urlopen(FAQUAD_DEV_URL)
        data = req.read().decode("utf-8")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(data)
            
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_faquad_corpus_and_queries(faquad_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Retorna o corpus (paragrafos) e queries (perguntas e qrels).
    """
    corpus = []
    queries = []
    
    for article in faquad_data.get("data", []):
        title = article.get("title", "")
        for p_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            doc_id = f"{title}_{p_idx}"
            
            corpus.append({
                "id": doc_id,
                "title": title,
                "text": context
            })
            
            for qa in paragraph.get("qas", []):
                queries.append({
                    "qid": qa["id"],
                    "query": qa["question"],
                    "relevant_doc_ids": [doc_id],
                    "answers": [ans["text"] for ans in qa.get("answers", [])]
                })
                
    return corpus, queries

def get_mmarco_subset() -> Tuple[List[Dict], List[Dict]]:
    """
    Retorna um subconjunto sintetizado representativo do mMARCO-PT para testes locais rapidos.
    Isso evita o download de 3GB do collection.tsv.
    """
    corpus = [
        {"id": "doc1", "text": "A gravidade na Terra acelera os objetos a cerca de 9,8 m/s²."},
        {"id": "doc2", "text": "A Lua e o unico satelite natural da Terra, orbitando a uma distancia de cerca de 384.400 km."},
        {"id": "doc3", "text": "A missao Apollo 11 levou os primeiros humanos a Lua em 1969. Neil Armstrong foi o primeiro a pisar na superficie."},
        {"id": "doc4", "text": "As mare na Terra sao causadas principalmente pelas forcas gravitacionais exercidas pela Lua e pelo Sol."},
        {"id": "doc5", "text": "O Sol e a estrela central do Sistema Solar. E uma esfera quase perfeita de plasma quente."},
        {"id": "doc6", "text": "Os principais sintomas da gastrite incluem dor de estomago, nauseas, vomitos e sensacao de saciedade precoce."},
        {"id": "doc7", "text": "A gastrite e a inflamacao do revestimento do estomago, podendo ser aguda ou cronica."},
        {"id": "doc8", "text": "A agua ferve a 100 graus Celsius ao nivel do mar, mas a temperatura de ebulicao diminui em altitudes mais elevadas."},
    ]
    
    queries = [
        {"qid": "q1", "query": "qual a aceleracao da gravidade na terra?", "relevant_doc_ids": ["doc1"]},
        {"qid": "q2", "query": "quem foi o primeiro homem a pisar na lua?", "relevant_doc_ids": ["doc3"]},
        {"qid": "q3", "query": "quais sao os sintomas de gastrite?", "relevant_doc_ids": ["doc6", "doc7"]},
        {"qid": "q4", "query": "o que causa as mares?", "relevant_doc_ids": ["doc4"]},
        {"qid": "q5", "query": "a que temperatura a agua ferve?", "relevant_doc_ids": ["doc8"]},
    ]
    
    return corpus, queries
