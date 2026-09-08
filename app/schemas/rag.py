import json
from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    page_number: Optional[int] = None
    snippet: str
    score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or prompt to ask the RAG assistant")
    conversation_id: Optional[int] = Field(None, description="Optional conversation ID to continue existing chat")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default number of retrieved chunks")


class QueryResponse(BaseModel):
    conversation_id: int
    message_id: int
    query: str
    answer: str
    citations: List[Citation] = []
    retrieval_count: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: List[Citation] = []
    created_at: datetime

    @classmethod
    def from_model(cls, msg: Any) -> "MessageOut":
        citations: List[Citation] = []
        if getattr(msg, "sources_json", None):
            try:
                data = json.loads(msg.sources_json)
                citations = [Citation(**c) for c in data]
            except Exception:
                citations = []
        return cls(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            citations=citations,
            created_at=msg.created_at
        )


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut] = []
