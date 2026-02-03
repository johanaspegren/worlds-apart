# modules/vector_store.py

import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.docs = {}     # id -> text
        self.vecs = {}     # id -> embedding vector

    def add(self, doc_id: str, text: str, embedding: list[float]):
        self.docs[doc_id] = text
        self.vecs[doc_id] = np.array(embedding, dtype=float)

    def query(self, query_embedding, n=3):
        if not self.vecs:
            return []

        q = np.array(query_embedding, dtype=float)

        # Compute cosine similarity
        scores = []
        for doc_id, vec in self.vecs.items():
            sim = np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec))
            scores.append((sim, doc_id))

        # sort by similarity descending
        scores.sort(reverse=True, key=lambda x: x[0])

        top = scores[:n]

        return [
            {
                "id": doc_id,
                "score": score,
                "text": self.docs[doc_id],
            }
            for score, doc_id in top
        ]
