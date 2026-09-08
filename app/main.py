import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import check_health
from app.api.v1.router import api_router
from app.config import get_settings
from app.core.database import init_db
from app.services.vector_store import get_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown routines."""
    logger.info("Initializing Enterprise RAG Assistant...")
    settings.ensure_directories()
    await init_db()
    # Initialize ChromaDB client
    get_vector_store()
    logger.info(f"{settings.PROJECT_NAME} initialized successfully.")
    yield
    logger.info("Shutting down Enterprise RAG Assistant...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
## Enterprise RAG (Retrieval-Augmented Generation) Assistant

### Features:
* **JWT Authentication**: Register, authenticate, and secure APIs.
* **Document Processing**: Upload `.pdf`, `.txt`, and `.md` files with text extraction and configurable sliding-window chunking.
* **Vector Storage & Semantic Retrieval**: Sentence-Transformer embeddings stored in isolated per-user ChromaDB collections.
* **Ollama Integration**: Local LLM generation with grounded answers and source citations.
* **Conversation Management**: Persisted multi-turn chat sessions with message history.
* **Multi-tenancy**: Strict user-isolated document access and vector indexing.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api/v1")

# Convenience root health probe
app.add_api_route("/health", check_health, methods=["GET"], tags=["Health"])


@app.get("/", tags=["General"])
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health"
    }


def main():
    """CLI entrypoint for running via uv run main.py or direct execution."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    main()
