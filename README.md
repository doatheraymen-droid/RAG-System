🤖 RAG System — Retrieval-Augmented Generation API
A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama. Upload documents and ask questions — the system retrieves relevant context and generates accurate, grounded answers.

✨ Features
Document ingestion — upload PDF, DOCX, or TXT files via REST API
Vector search — ChromaDB stores and retrieves semantically similar chunks
LLM Factory pattern — pluggable LLM providers (Ollama, OpenAI, etc.)
Arabic language support — automatically detects Arabic queries and responds in Arabic
Docker ready — single docker-compose up to run everything
REST API — FastAPI with automatic Swagger docs at /docs
🏗️ Architecture
User Query
    ↓
FastAPI (main.py)
    ↓
RAGEngine (rag_engine.py)
    ├── ChromaDB  ← vector store + embeddings
    ├── LLMFactory (llm_factory.py) ← pluggable LLM providers
    └── Arabic detector ← bilingual support (EN/AR)
📁 Project Structure
rag-system/
├── main.py           # FastAPI app — upload & query endpoints
├── rag_engine.py     # Core RAG pipeline (ingest, retrieve, generate)
├── llm_factory.py    # Factory pattern for LLM providers
├── interface.py      # Streamlit UI
├── requirements.txt  # Python dependencies
├── docker-compose.yml
└── .env              # Configuration (chunk size, LLM provider, etc.)
🚀 Getting Started
Prerequisites
Docker & Docker Compose
OR Python 3.10+
Run with Docker
git clone https://github.com/zeyad12112/rag-system
cd rag-system
docker-compose up
Run locally
pip install -r requirements.txt
python main.py
API available at: http://localhost:8000 Swagger docs at: http://localhost:8000/docs

📡 API Endpoints
Method	Endpoint	Description
POST	/upload	Upload a PDF, DOCX, or TXT file
POST	/query	Ask a question against uploaded documents
GET	/files	List all uploaded files
GET	/health	Health check
Example
# Upload a document
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 5}'
⚙️ Configuration (.env)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
LLM_PROVIDER=ollama   # or openai
🛠️ Tech Stack
Python FastAPI Docker ChromaDB Streamlit

👤 Author
Ather Aymen — github.com/doatheraymen
