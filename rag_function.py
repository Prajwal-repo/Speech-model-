import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGRetriever:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []

    def add_to_index(self, texts):
        """Embeds and indexes the documents in FAISS."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  
        
        self.index.add(embeddings)
        self.documents.extend(texts)  

    def retrieve_documents(self, query, top_k=3):
        """Retrieves the most relevant documents based on a query."""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, top_k)

        return [self.documents[i] for i in indices[0]]

