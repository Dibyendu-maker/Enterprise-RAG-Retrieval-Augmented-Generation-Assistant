import uuid
from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.document import Document, DocumentChunk
from app.models.user import User
from app.schemas.document import DocumentDetailOut, DocumentOut, DocumentUploadResponse
from app.services.document_service import get_document_service
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .txt, .md)"),
    chunk_size: Optional[int] = Form(None, description="Optional custom chunk size in characters"),
    chunk_overlap: Optional[int] = Form(None, description="Optional custom chunk overlap in characters"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document (PDF, TXT, MD), extract text, segment into chunks,
    compute embeddings, and index into the user's isolated ChromaDB collection.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename missing.")

    filename = file.filename
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Only .pdf, .txt, and .md files are supported."
        )

    # Read and validate file content
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    doc_service = get_document_service()
    doc_hash = doc_service.compute_hash(content)

    # Check for duplicate document for this user
    existing_stmt = select(Document).where(Document.user_id == current_user.id, Document.file_hash == doc_hash)
    existing_res = await db.execute(existing_stmt)
    existing_doc = existing_res.scalar_one_or_none()
    if existing_doc:
        return DocumentUploadResponse(
            message="Document already uploaded and indexed for this user.",
            document=DocumentOut.model_validate(existing_doc)
        )

    # 1. Text extraction
    try:
        extracted_pages = doc_service.extract_text(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 2. Chunking
    chunks_data = doc_service.chunk_text(
        extracted_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    if not chunks_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No extractable text chunks in document.")

    # 3. Create Document record in DB
    new_doc = Document(
        user_id=current_user.id,
        filename=filename,
        file_type=ext.lstrip("."),
        file_size=len(content),
        file_hash=doc_hash,
        chunk_count=len(chunks_data)
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # 4. Generate embeddings
    chunk_texts = [c["content"] for c in chunks_data]
    embedding_service = get_embedding_service()
    embeddings = await embedding_service.embed_texts(chunk_texts)

    # 5. Insert chunks into isolated ChromaDB collection
    chunk_ids = [str(uuid.uuid4()) for _ in chunks_data]
    page_numbers = [c.get("page_number") for c in chunks_data]

    vector_store = get_vector_store()
    vector_store.add_chunks(
        user_id=current_user.id,
        document_id=new_doc.id,
        filename=filename,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        embeddings=embeddings,
        page_numbers=page_numbers
    )

    # 6. Save Chunk metadata records in SQLite
    for i, c in enumerate(chunks_data):
        chunk_obj = DocumentChunk(
            document_id=new_doc.id,
            chunk_index=i,
            content=c["content"],
            char_count=c["char_count"],
            page_number=c.get("page_number"),
            chroma_id=chunk_ids[i]
        )
        db.add(chunk_obj)
    await db.commit()

    return DocumentUploadResponse(
        message=f"Document uploaded and indexed successfully into {len(chunks_data)} chunks.",
        document=DocumentOut.model_validate(new_doc)
    )


@router.get("", response_model=List[DocumentOut])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """List all documents owned by the current authenticated user."""
    stmt = select(Document).where(Document.user_id == current_user.id).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    docs = result.scalars().all()
    return docs


@router.get("/{document_id}", response_model=DocumentDetailOut)
async def get_document(
    document_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Retrieve detailed information and chunks for a specific document."""
    stmt = (
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
        .options(selectinload(Document.chunks))
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Delete a document and purge its vectors from ChromaDB and SQLite."""
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )

    # Purge vectors from ChromaDB
    vector_store = get_vector_store()
    vector_store.delete_document_chunks(user_id=current_user.id, document_id=document_id)

    # Delete from database (cascades to chunks)
    await db.delete(doc)
    await db.commit()

    return {"message": f"Document {document_id} and its associated vectors were deleted successfully."}
