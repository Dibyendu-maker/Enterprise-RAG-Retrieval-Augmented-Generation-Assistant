from typing import Annotated
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token, oauth2_scheme
from app.models.user import User
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService, get_document_service
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.ollama_service import OllamaService, get_ollama_service
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreService, get_vector_store


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise credentials_exception

    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account."
        )

    return user


async def get_rag_service_dep(
    db: Annotated[AsyncSession, Depends(get_db)]
) -> RAGService:
    return RAGService(
        db=db,
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        ollama_service=get_ollama_service()
    )
