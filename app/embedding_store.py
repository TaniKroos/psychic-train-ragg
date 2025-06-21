# embedding_store.py

import faiss
import os
import json
from typing import List, Dict, Tuple
import numpy as np

INDEX_PATH = "faiss_index/index.faiss"
META_PATH = "faiss_index/metadata.json"

DIMENSIONS = 1536  # Cohere embeddings = 1024 dims
HNSW_M = 32

class EmbeddingStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(DIMENSIONS)
        self.metadata = []
        self._load_index()

    def _load_index(self):
        os.makedirs("faiss_index", exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexHNSWFlat(DIMENSIONS, HNSW_M)
            self.index.hnsw.efConstruction = 40
            self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str], username: str):
        print("Expected index dimension:", self.index.d)
        

        ids = list(range(len(self.metadata), len(self.metadata) + len(embeddings)))
        
        vectors = np.array(embeddings).astype('float32')
        print("Incoming vector shape:", vectors.shape)
        assert vectors.shape[1] == self.index.d, f"Embedding dim {vectors.shape[1]} != index dim {self.index.d}"

        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": ids[i],
                "username": username,
                "text": chunk
            })
        self._save()

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.metadata, f)

    def search(self, query_embedding: List[float], username: str, k: int = 5) -> List[str]:
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), k * 2)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                if meta["username"] == username:
                    results.append(meta["text"])
            if len(results) >= k:
                break
        return results
