import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pypdf

from app.config import get_settings

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.settings = get_settings()

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def extract_text(self, filename: str, content: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from file bytes.
        Returns a list of dicts: [{"text": str, "page_number": Optional[int]}]
        """
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(content)
        elif ext in [".txt", ".md"]:
            return self._extract_text_plain(content)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .txt, .md")

    def _extract_pdf(self, content: bytes) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    results.append({"text": cleaned, "page_number": idx + 1})
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")

        if not results:
            raise ValueError("No extractable text found in the PDF document.")
        return results

    def _extract_text_plain(self, content: bytes) -> List[Dict[str, Any]]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Unable to decode text file: {e}")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Document is empty.")
        return [{"text": cleaned, "page_number": 1}]

    def chunk_text(
        self,
        extracted_pages: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk extracted text into segments using configurable chunk size and overlap.
        Preserves page number metadata.
        Returns: [{"content": str, "page_number": Optional[int], "char_count": int}]
        """
        size = chunk_size or self.settings.CHUNK_SIZE
        overlap = chunk_overlap or self.settings.CHUNK_OVERLAP

        if overlap >= size:
            overlap = max(0, size // 4)

        all_chunks: List[Dict[str, Any]] = []

        for item in extracted_pages:
            text = item["text"]
            page_num = item.get("page_number", 1)
            
            # Split text using recursive boundaries
            page_chunks = self._recursive_split(text, size, overlap)
            for chunk_str in page_chunks:
                clean_chunk = chunk_str.strip()
                if clean_chunk:
                    all_chunks.append({
                        "content": clean_chunk,
                        "page_number": page_num,
                        "char_count": len(clean_chunk)
                    })

        return all_chunks

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Find the best separator
        chosen_separator = ""
        for sep in separators:
            if sep == "":
                chosen_separator = ""
                break
            if sep in text:
                chosen_separator = sep
                break

        if chosen_separator:
            parts = text.split(chosen_separator)
        else:
            # Fallback character slice
            parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
            return parts

        chunks: List[str] = []
        current_chunk = ""

        for part in parts:
            candidate = current_chunk + (chosen_separator if current_chunk else "") + part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Create overlap from end of current_chunk
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        overlap_prefix = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_prefix + chosen_separator + part
                    else:
                        current_chunk = part
                else:
                    # Single part is larger than chunk_size, split recursively with finer separator
                    next_separators = separators[separators.index(chosen_separator) + 1:] if chosen_separator in separators else []
                    sub_chunks = self._recursive_split(part, chunk_size, chunk_overlap, next_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks


_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
