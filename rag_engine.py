import os
from pathlib import Path
import hashlib
import requests
import re
from typing import List

# Use ChromaDB's built-in embeddings (no sentence-transformers needed)
import chromadb
from chromadb.utils import embedding_functions

# Import LLM Factory for Bonus 1
from llm_factory import LLMFactory


class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine (with Bonuses)...")
        
        # Use ChromaDB's built-in embedding function
        self.embeddings = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embeddings
        )
        
        # Bonus 1: Use Factory Pattern for LLM
        self.llm = LLMFactory.get_provider()
        print(f"LLM Provider: {type(self.llm).__name__}")
        
        print("RAG Engine ready! (Bonuses included)")

    def is_arabic(self, text: str) -> bool:
        """Bonus 2: Detect if text contains Arabic characters"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u0870-\u089F\uFB50-\uFDFF\uFE70-\uFEFF]+')
        return bool(arabic_pattern.search(text))

    async def add_document(self, file_path: str):
        """Load and chunk a document"""
        print(f"Processing: {file_path}")
        
        # Read file content
        ext = Path(file_path).suffix.lower()
        text = ""
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        elif ext == '.docx':
            import docx
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        if not text:
            print(f"Could not read {file_path}")
            return 0
        
        # Simple chunking
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        
        # Add to ChromaDB
        ids = [hashlib.md5(f"{file_path}_{i}".encode()).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": Path(file_path).name, "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks")
        return len(chunks)

    async def retrieve(self, query: str, top_k: int = 5) -> List:
        """Retrieve relevant chunks"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        return [{"text": doc, "score": 1 - dist, "metadata": meta} 
                for doc, dist, meta in zip(documents, distances, metadatas)]

    async def generate(self, query: str, context: List[str]) -> str:
        """Generate answer using LLM Factory (Bonus 1) with Arabic support (Bonus 2)"""
        context_text = "\n\n".join([f"[Document {i+1}]: {text[:1000]}" for i, text in enumerate(context)])
        
        # Bonus 2: Choose prompt language based on query
        if self.is_arabic(query):
            prompt = f"""أنت مساعد ذكي. أجب بناءً فقط على المستندات أدناه.

المستندات:
{context_text}

السؤال: {query}

الإجابة:"""
        else:
            prompt = f"""You are a helpful assistant. Answer based ONLY on the documents below.

DOCUMENTS:
{context_text}

QUESTION: {query}

Answer concisely using only the information above. If the answer isn't in the documents, say "I don't have enough information."

ANSWER:"""

        # Bonus 1: Use the factory provider
        return self.llm.generate(prompt)

    async def query(self, question: str, top_k: int = 5) -> dict:
        """Full RAG pipeline"""
        retrieved = await self.retrieve(question, top_k)
        
        if not retrieved:
            return {
                "question": question,
                "answer": "No relevant documents found. Please upload documents first.",
                "retrieved_chunks": []
            }
        
        context_texts = [r["text"] for r in retrieved]
        answer = await self.generate(question, context_texts)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {"text": r["text"][:300] + "...", "score": r["score"], "source": r["metadata"].get("source", "unknown")}
                for r in retrieved
            ]
        }

    def health_check(self):
        """Check if LLM provider is working"""
        return self.llm.health_check() if hasattr(self.llm, 'health_check') else True