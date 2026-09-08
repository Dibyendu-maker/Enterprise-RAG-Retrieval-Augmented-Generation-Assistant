import logging
from typing import List, Optional
import anyio
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        return self._model

    def _sync_embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a list of text strings asynchronously."""
        if not texts:
            return []
        return await anyio.to_thread.run_sync(self._sync_embed_texts, texts)

    async def embed_query(self, query: str) -> List[float]:
        """Generate a single vector embedding for a query string."""
        results = await self.embed_texts([query])
        return results[0]

    def is_healthy(self) -> bool:
        """Check if model can be loaded or is already loaded."""
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"Embedding model check failed: {e}")
            return False


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
