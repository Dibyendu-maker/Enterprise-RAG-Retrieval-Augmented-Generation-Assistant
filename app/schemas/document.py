from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class DocumentChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_index: int
    content: str
    char_count: int
    page_number: Optional[int] = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    created_at: datetime


class DocumentDetailOut(DocumentOut):
    chunks: List[DocumentChunkOut] = []


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentOut
