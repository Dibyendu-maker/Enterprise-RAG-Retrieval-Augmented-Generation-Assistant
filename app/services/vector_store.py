import logging
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    _instance: Optional["VectorStoreService"] = None

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        self.persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
        self._client: Optional[chromadb.PersistentClient] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            logger.info(f"Initializing ChromaDB PersistentClient at {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    def _get_user_collection_name(self, user_id: int) -> str:
        """Isolated collection name for each user."""
        return f"user_{user_id}_documents"

    def get_or_create_user_collection(self, user_id: int):
        collection_name = self._get_user_collection_name(user_id)
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        user_id: int,
        document_id: int,
        filename: str,
        chunk_ids: List[str],
        chunk_texts: List[str],
        embeddings: List[List[float]],
        page_numbers: Optional[List[Optional[int]]] = None
    ) -> None:
        """Add document chunks to the user's isolated vector collection."""
        if not chunk_ids:
            return

        collection = self.get_or_create_user_collection(user_id)
        
        metadatas: List[Dict[str, Any]] = []
        for i in range(len(chunk_ids)):
            page_num = page_numbers[i] if page_numbers and i < len(page_numbers) and page_numbers[i] is not None else 1
            metadatas.append({
                "user_id": user_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": int(page_num)
            })

        collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunk_ids)} chunks for doc {document_id} into collection {collection.name}")

    def query_similar(
        self,
        user_id: int,
        query_embedding: List[float],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Query top-k most similar chunks for a user."""
        collection = self.get_or_create_user_collection(user_id)
        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 1.0
                # In cosine distance: similarity = 1 - distance
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
                meta = metadatas[i] if i < len(metadatas) else {}

                formatted_results.append({
                    "id": ids[i],
                    "content": docs[i] if i < len(docs) else "",
                    "document_id": meta.get("document_id", 0),
                    "filename": meta.get("filename", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "page_number": meta.get("page_number", 1),
                    "score": round(similarity_score, 4)
                })

        return formatted_results

    def delete_document_chunks(self, user_id: int, document_id: int) -> None:
        """Delete all chunks for a specific document from user's collection."""
        collection = self.get_or_create_user_collection(user_id)
        collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted chunks for doc {document_id} from user {user_id}'s collection")

    def is_healthy(self) -> bool:
        """Check if ChromaDB client is accessible."""
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
