import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.chat import Conversation, Message
from app.models.user import User
from app.schemas.rag import Citation, QueryRequest, QueryResponse
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.ollama_service import OllamaService, get_ollama_service
from app.services.vector_store import VectorStoreService, get_vector_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an Enterprise RAG Assistant. Your task is to provide accurate, helpful answers strictly based on the provided source context.

Instructions:
1. Base your answer strictly on the facts present in the source context below.
2. Whenever you state information from a source, cite it immediately using its citation marker (e.g., [Source 1], [Source 2]).
3. If the context does not contain enough information to answer the question, clearly state: "I do not have sufficient information in the provided documents to answer this question." Do not make up facts.
4. Keep answers professional, concise, and well-structured.
"""


class RAGService:
    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        ollama_service: Optional[OllamaService] = None
    ):
        self.db = db
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.ollama_service = ollama_service or get_ollama_service()

    async def get_or_create_conversation(
        self,
        user: User,
        conversation_id: Optional[int],
        initial_query: str
    ) -> Conversation:
        if conversation_id:
            stmt = (
                select(Conversation)
                .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
                .options(selectinload(Conversation.messages))
            )
            result = await self.db.execute(stmt)
            conv = result.scalar_one_or_none()
            if not conv:
                raise ValueError(f"Conversation with ID {conversation_id} not found for this user.")
            return conv

        # Create new conversation
        title = initial_query.strip()[:60] if initial_query else "New Conversation"
        conv = Conversation(user_id=user.id, title=title)
        self.db.add(conv)
        await self.db.commit()
        await self.db.refresh(conv)
        return conv

    def _build_context_and_citations(
        self,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Citation]]:
        context_parts: List[str] = []
        citations: List[Citation] = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_tag = f"[Source {idx}]"
            filename = chunk.get("filename", "unknown")
            page_num = chunk.get("page_number", 1)
            content = chunk.get("content", "").strip()

            snippet = content[:180] + "..." if len(content) > 180 else content
            citations.append(Citation(
                document_id=chunk.get("document_id", 0),
                filename=filename,
                chunk_index=chunk.get("chunk_index", 0),
                page_number=page_num,
                snippet=snippet,
                score=chunk.get("score", 0.0)
            ))

            header = f"{source_tag} File: {filename} (Page: {page_num}):\n"
            context_parts.append(f"{header}{content}\n")

        context_str = "\n---\n".join(context_parts)
        return context_str, citations

    async def answer_query(self, user: User, request: QueryRequest) -> QueryResponse:
        # 1. Get or create conversation record
        conversation = await self.get_or_create_conversation(user, request.conversation_id, request.query)

        # 2. Generate embedding for user query
        query_embedding = await self.embedding_service.embed_query(request.query)

        # 3. Retrieve relevant chunks from user's isolated vector store
        top_k = request.top_k or self.settings.TOP_K_RESULTS
        retrieved_chunks = self.vector_store.query_similar(
            user_id=user.id,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 4. Assemble context and citations
        context_str, citations = self._build_context_and_citations(retrieved_chunks)

        # 5. Build prompt incorporating recent conversation history
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(4)
        )
        msg_result = await self.db.execute(stmt)
        recent_messages = list(reversed(msg_result.scalars().all()))

        history_str = ""
        if recent_messages:
            history_lines = [f"{m.role.capitalize()}: {m.content}" for m in recent_messages]
            history_str = "\nConversation History:\n" + "\n".join(history_lines) + "\n"

        if not retrieved_chunks:
            answer = (
                "You haven't uploaded any documents yet, or no relevant information was found in your library. "
                "Please upload documents (.pdf, .txt, .md) to enable grounded answers."
            )
        else:
            user_prompt = f"""{history_str}
Source Context:
{context_str}

User Question:
{request.query}

Answer:"""
            try:
                answer = await self.ollama_service.generate_response(
                    prompt=user_prompt,
                    system=SYSTEM_PROMPT
                )
            except RuntimeError as e:
                logger.warning(f"Ollama generation fallback: {e}")
                # Provide retrieved snippets clearly if Ollama is unreachable
                snippets_summary = "\n".join(
                    [f"- [Source {i+1}] ({c.filename}): {c.snippet}" for i, c in enumerate(citations)]
                )
                answer = (
                    f"**Notice**: The local LLM (Ollama) is currently unreachable ({str(e)}).\n\n"
                    f"**Retrieved matching context from your documents:**\n{snippets_summary}"
                )

        # 6. Save user query message and assistant response message to SQLite
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.query
        )
        self.db.add(user_msg)

        citations_json = json.dumps([c.model_dump() for c in citations]) if citations else None
        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources_json=citations_json
        )
        self.db.add(assistant_msg)

        await self.db.commit()
        await self.db.refresh(assistant_msg)

        return QueryResponse(
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            query=request.query,
            answer=answer,
            citations=citations,
            retrieval_count=len(retrieved_chunks)
        )
