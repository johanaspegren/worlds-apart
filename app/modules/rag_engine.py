from modules.text_processing import chunk_text
from modules.llm_handler import LLMHandler

class RAGEngine:
    def __init__(self, llm: LLMHandler, vector_store):
        self.llm = llm
        self.vector_store = vector_store

    def ingest(self, text: str):
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            emb = self.llm.embed(chunk)
            self.vector_store.add(f"chunk_{idx}", chunk, emb)

    def answer(self, question: str):
        q_emb = self.llm.embed(question)
        matches = self.vector_store.query(q_emb, n=3)

        context = "\n\n".join([m["text"] for m in matches])
        prompt = f"Answer the question using ONLY this context:\n{context}\n\nQuestion: {question}"

        return self.llm.call(prompt)
