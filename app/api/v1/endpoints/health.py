from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
from app.schemas.health import HealthResponse
from app.services.embedding_service import get_embedding_service
from app.services.ollama_service import get_ollama_service
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def check_health(
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Comprehensive system health check inspecting SQLite, ChromaDB,
    the embedding service, and the Ollama LLM service.
    """
    components = {}

    # 1. Database check
    try:
        await db.execute(text("SELECT 1"))
        components["database"] = "healthy"
    except Exception as e:
        components["database"] = f"unhealthy ({str(e)})"

    # 2. ChromaDB check
    try:
        vs = get_vector_store()
        components["chromadb"] = "healthy" if vs.is_healthy() else "unhealthy"
    except Exception as e:
        components["chromadb"] = f"unhealthy ({str(e)})"

    # 3. Ollama service check
    try:
        ollama = get_ollama_service()
        ollama_ok = await ollama.is_healthy()
        components["ollama"] = "healthy" if ollama_ok else "unreachable"
    except Exception as e:
        components["ollama"] = f"unreachable ({str(e)})"

    # 4. Overall status determination
    is_ok = components.get("database") == "healthy" and components.get("chromadb") == "healthy"
    if is_ok:
        status_str = "ok" if components.get("ollama") == "healthy" else "degraded"
    else:
        status_str = "error"

    return HealthResponse(
        status=status_str,
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc),
        components=components
    )
