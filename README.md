# Enterprise RAG Assistant

A production-grade, multi-tenant Retrieval-Augmented Generation (RAG) assistant built with **Python 3.14 / uv**, **FastAPI**, **ChromaDB**, **Sentence Transformers**, **Ollama**, **SQLite / SQLAlchemy (async)**, **JWT authentication**, and **Docker**.

---

## 🏛 Architecture

```
Client / UI / cURL
         │
         ▼
FastAPI Application Gateway (CORS, Lifespan, Swagger Docs)
         │
         ▼
JWT Authentication & Multi-Tenant Context (User Isolation)
         │
         ▼
RAG Service Orchestrator
   ├── ChromaDB (Isolated collections per user: `user_{id}_documents`)
   ├── Sentence Transformers (`all-MiniLM-L6-v2` dense local embeddings)
   ├── SQLite + aiosqlite (Users, document registry, multi-turn chat history)
   └── Ollama LLM Service (Local grounded generation + source citations)
```

---

## 🚀 Core Features

* **User Authentication**: Secure user registration and JWT bearer authentication using bcrypt and PyJWT.
* **Document Ingestion**: Supports `.pdf` (per-page text extraction), `.txt`, and `.md` file formats.
* **Smart Chunking**: Configurable recursive text chunking with custom chunk size and sliding-window overlap.
* **Vector Embeddings**: High-performance local vectorization via Sentence Transformers (`all-MiniLM-L6-v2`) offloaded to threadpools.
* **Vector Store & Multi-Tenancy**: ChromaDB vector storage with strict tenant isolation (dedicated per-user collections and metadata boundaries).
* **Semantic Retrieval & Grounded LLM**: Dense semantic retrieval feeding structured context into local Ollama models (`llama3.2:1b`, `mistral`, etc.).
* **Verified Source Citations**: Accurate citations mapping to document filenames, page numbers, chunks, and similarity scores.
* **Conversation Memory**: Persisted multi-turn chat sessions with message history and context continuity.
* **System Health Check**: Comprehensive `/health` probe verifying SQLite, ChromaDB, and Ollama statuses.
* **Interactive API Docs**: Swagger UI (`/docs`) and ReDoc (`/redoc`) out-of-the-box.
* **Docker Support**: Containerized deployment with `Dockerfile` and `docker-compose.yml`.

---

## 📁 Project Structure

```
MyAIProject/
├── .env                             # Environment configuration
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore configuration
├── .python-version                  # Pinned Python version
├── Dockerfile                       # Multi-stage container build
├── docker-compose.yml               # Service orchestration (FastAPI + Ollama)
├── pyproject.toml                   # Project dependencies and tool settings
├── README.md                        # Documentation and guides
├── main.py                          # Root entrypoint
├── app/
│   ├── config.py                    # Pydantic BaseSettings
│   ├── main.py                      # FastAPI application factory and lifespan
│   ├── core/
│   │   ├── database.py              # Async SQLAlchemy engine & session maker
│   │   ├── security.py              # Bcrypt hashing & PyJWT tokens
│   │   └── dependencies.py          # FastAPI Depends (auth, db, services)
│   ├── models/
│   │   ├── user.py                  # User entity
│   │   ├── document.py              # Document and DocumentChunk entities
│   │   └── chat.py                  # Conversation and Message entities
│   ├── schemas/
│   │   ├── auth.py                  # Auth request/response schemas
│   │   ├── document.py              # Document metadata schemas
│   │   ├── rag.py                   # Query, Citation, and Chat schemas
│   │   └── health.py                # Health status schemas
│   ├── services/
│   │   ├── auth_service.py          # Registration and token issuance
│   │   ├── document_service.py      # PDF/TXT/MD extraction and chunking
│   │   ├── embedding_service.py     # SentenceTransformer embedding service
│   │   ├── vector_store.py          # Isolated ChromaDB vector operations
│   │   ├── ollama_service.py        # Ollama HTTP client
│   │   └── rag_service.py           # RAG pipeline orchestration
│   └── api/
│       └── v1/
│           ├── router.py            # Aggregated v1 router
│           └── endpoints/
│               ├── auth.py          # Auth endpoints
│               ├── documents.py     # Document endpoints
│               ├── rag.py           # RAG and conversation endpoints
│               └── health.py        # Health probe endpoint
└── tests/                           # Async test suite (Pytest)
```

---

## 🛠 Getting Started

### 1. Prerequisites
* Python 3.12+ (or 3.14)
* [uv package manager](https://docs.astral.sh/uv/) installed
* [Ollama](https://ollama.com/) running locally (e.g. `ollama run llama3.2:1b`)

### 2. Environment Setup

```bash
# Clone repository and navigate to project root
cd MyAIProject

# Install all dependencies into virtual environment
uv sync

# Configure environment variables
cp .env.example .env
```

### 3. Run the Application

```bash
# Start the server using uv
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Swagger UI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📖 API Usage Guide & Examples

### 1. User Registration & Login

**Register a User**:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "engineer@enterprise.com", "password": "SuperSecretPassword123!", "full_name": "Lead Engineer"}'
```

**Obtain JWT Access Token**:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login-json" \
  -H "Content-Type: application/json" \
  -d '{"email": "engineer@enterprise.com", "password": "SuperSecretPassword123!"}'
```
*Save the `access_token` from the JSON response and set `TOKEN="<your-token>"`; pass `-H "Authorization: Bearer $TOKEN"` in subsequent requests.*

---

### 2. Document Ingestion

**Upload a PDF / Markdown / Text Document**:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample_policy.pdf" \
  -F "chunk_size=500" \
  -F "chunk_overlap=100"
```

**List Your Documents**:
```bash
curl -X GET "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer $TOKEN"
```

**Delete a Document and its Vectors**:
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/1" \
  -H "Authorization: Bearer $TOKEN"
```

---

### 3. RAG Query & Multi-Turn Conversations

**Ask a Question with Semantic Retrieval & Source Citations**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the password rotation requirements?",
    "top_k": 4
  }'
```

*Sample Response:*
```json
{
  "conversation_id": 1,
  "message_id": 2,
  "query": "What are the password rotation requirements?",
  "answer": "According to company security policies, passwords must be rotated every 90 days and require at least 12 characters [Source 1].",
  "citations": [
    {
      "document_id": 1,
      "filename": "sample_policy.pdf",
      "chunk_index": 2,
      "page_number": 3,
      "snippet": "Section 4.1: Passwords must be changed every 90 days and must not match the previous 5 passwords...",
      "score": 0.8921
    }
  ],
  "retrieval_count": 1
}
```

**Continue the Conversation**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": 1,
    "query": "Does this also apply to service accounts?"
  }'
```

**Retrieve Full Conversation History**:
```bash
curl -X GET "http://localhost:8000/api/v1/rag/conversations/1" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🐳 Docker Deployment

To launch both the FastAPI application and an Ollama container using Docker Compose:

```bash
# Build and run containers
docker compose up --build -d

# Check service logs
docker compose logs -f

# Download your desired model in the Ollama container
docker compose exec ollama ollama pull llama3.2:1b
```

The API will be live at `http://localhost:8000`.

---

## 🧪 Testing

Run the automated test suite covering authentication, document chunking, multi-tenancy isolation, citations, and health checks:

```bash
uv run pytest -v
```
