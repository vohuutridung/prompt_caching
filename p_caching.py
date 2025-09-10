import hashlib
import os
import json
from protonx import ProtonX
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

CACHE_DIR = os.getenv('CACHE_DIR')
SIM_THRESHOLD = float(os.getenv('SIM_THRESHOLD'))
TTL_WEEK = float(os.getenv('TTL_WEEK'))

def init_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def hash_md5(string):
    # Function to hash a string into md5 code
    return hashlib.md5(string.encode()).hexdigest()

def embed(string):
    # Function to embed a string into vector embedding using protonx
    client = ProtonX(api_key=os.getenv('PROTONX_API_KEY'))
    embeddings = client.embeddings.create(string)
    return embeddings['data'][0]['embedding']

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_from_cache(question):
    file_name = hash_md5(question) + '.json'
    file_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    vector_emb = embed(question)
    best_item = None
    best_score = 0.0
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                item = json.load(f)
            compared_vector = item['embedding']
            score = cosine_sim(vector_emb, compared_vector)

            if score > best_score and score >= SIM_THRESHOLD:
                best_score = score
                best_item = item
    
    return best_item

def store_to_cache(question, result):
    filename = hash_md5(question) + '.json'
    filepath = os.path.join(CACHE_DIR, filename)

    data = {
        'question': question,
        'result': result,
        'embedding': embed(question),
        'timestamp': datetime.now().replace(microsecond=0).isoformat()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_cache():
    # clear cache on startup
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                item = json.load(f)
            timestamp = datetime.fromisoformat(item['timestamp'])

            if datetime.now() - timestamp >= timedelta(weeks=TTL_WEEK):
                os.remove(filepath)

    print('Cache cleared successfully!')



