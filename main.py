from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uuid
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# Global variable for the RAG engine
rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG engine
    global rag_engine
    print("Initializing RAG Engine...")
    
    # Import here to avoid circular imports
    from rag_engine import RAGEngine
    
    rag_engine = RAGEngine()
    print("RAG Engine ready!")
    
    yield  # The application runs here
    
    # Shutdown: Clean up if needed
    print("Shutting down...")

app = FastAPI(title="RAG System API", lifespan=lifespan)

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a document (PDF, DOCX, TXT)"""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    # Save file
    file_path = DATA_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the document
    num_chunks = await rag_engine.add_document(str(file_path))
    
    return {
        "message": f"File {file.filename} uploaded and processed",
        "file_id": file_path.stem,
        "chunks_created": num_chunks
    }

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system"""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    result = await rag_engine.query(request.question, request.top_k)
    return result

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
    return {"files": files}

@app.get("/health")
async def health_check():
    """Check if Ollama is running"""
    if not rag_engine:
        return {"status": "not_initialized"}
    return {"status": "ok", "ollama": rag_engine.health_check()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)