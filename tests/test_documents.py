import io
import pytest
from httpx import AsyncClient
from pypdf import PdfWriter


def create_sample_pdf() -> bytes:
    """Helper to generate a minimal valid PDF with extractable text for testing."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Enterprise Security Policy) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000338 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"416\n"
        b"%%EOF"
    )


@pytest.mark.asyncio
async def test_upload_txt_document(client: AsyncClient, auth_headers: dict):
    file_content = b"Enterprise Cloud Architecture. Section 1: Security protocols and encryption at rest."
    files = {"file": ("cloud_security.txt", file_content, "text/plain")}
    data = {"chunk_size": 40, "chunk_overlap": 10}

    response = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files,
        data=data
    )
    assert response.status_code == 201
    res_data = response.json()
    assert "document" in res_data
    doc = res_data["document"]
    assert doc["filename"] == "cloud_security.txt"
    assert doc["file_type"] == "txt"
    assert doc["chunk_count"] > 1


@pytest.mark.asyncio
async def test_upload_markdown_document(client: AsyncClient, auth_headers: dict):
    md_content = b"# Enterprise Guide\n\n## Section 1\nKubernetes cluster configuration.\n\n## Section 2\nChromaDB vector persistence."
    files = {"file": ("guide.md", md_content, "text/markdown")}

    response = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files
    )
    assert response.status_code == 201
    assert response.json()["document"]["filename"] == "guide.md"


@pytest.mark.asyncio
async def test_upload_pdf_document(client: AsyncClient, auth_headers: dict):
    pdf_bytes = create_sample_pdf()
    files = {"file": ("enterprise_policy.pdf", pdf_bytes, "application/pdf")}
    response = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files
    )
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["document"]["filename"] == "enterprise_policy.pdf"
    assert res_data["document"]["file_type"] == "pdf"
    assert res_data["document"]["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_unsupported_file_extension(client: AsyncClient, auth_headers: dict):
    files = {"file": ("script.py", b"print('hello')", "text/x-python")}
    response = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_user_document_isolation(
    client: AsyncClient,
    auth_headers: dict,
    second_auth_headers: dict
):
    # User 1 (Alice) uploads a document
    alice_content = b"Confidential financial statement for Q4."
    files = {"file": ("q4_financials.txt", alice_content, "text/plain")}
    alice_upload = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files
    )
    assert alice_upload.status_code == 201
    alice_doc_id = alice_upload.json()["document"]["id"]

    # User 2 (Bob) lists documents -> should NOT see Alice's document
    bob_list = await client.get("/api/v1/documents", headers=second_auth_headers)
    assert bob_list.status_code == 200
    bob_docs = bob_list.json()
    assert not any(d["id"] == alice_doc_id for d in bob_docs)

    # User 2 attempts to get Alice's document by ID -> returns 404
    bob_get = await client.get(f"/api/v1/documents/{alice_doc_id}", headers=second_auth_headers)
    assert bob_get.status_code == 404

    # User 2 attempts to delete Alice's document -> returns 404
    bob_delete = await client.delete(f"/api/v1/documents/{alice_doc_id}", headers=second_auth_headers)
    assert bob_delete.status_code == 404


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient, auth_headers: dict):
    files = {"file": ("temp_to_delete.txt", b"Temporary notes to be purged.", "text/plain")}
    upload_res = await client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files=files
    )
    doc_id = upload_res.json()["document"]["id"]

    # Delete
    del_res = await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert del_res.status_code == 200

    # Ensure it no longer exists
    get_res = await client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert get_res.status_code == 404
