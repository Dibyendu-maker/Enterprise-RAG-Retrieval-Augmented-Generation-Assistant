from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_rag_query_empty_library(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/v1/rag/query",
        headers=auth_headers,
        json={"query": "What is our company's refund policy?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "answer" in data
    assert data["retrieval_count"] == 0
    assert "upload documents" in data["answer"].lower()


@pytest.mark.asyncio
async def test_rag_query_with_documents_and_citations(client: AsyncClient, auth_headers: dict):
    # Upload context document
    content = (
        "Enterprise Architecture Policy: All external APIs must enforce TLS 1.3 encryption. "
        "Session tokens must expire after 24 hours. Single sign-on with multi-factor authentication is mandatory."
    )
    files = {"file": ("security_policy.txt", content.encode("utf-8"), "text/plain")}
    await client.post("/api/v1/documents/upload", headers=auth_headers, files=files)

    # Mock Ollama generation to return an answer citing the retrieved source
    mock_llm_answer = "All external APIs must use TLS 1.3 encryption and session tokens must expire after 24 hours [Source 1]."
    with patch("app.services.ollama_service.OllamaService.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_llm_answer

        query_res = await client.post(
            "/api/v1/rag/query",
            headers=auth_headers,
            json={"query": "What are the API encryption requirements?"}
        )
        assert query_res.status_code == 200
        data = query_res.json()
        assert data["retrieval_count"] > 0
        assert len(data["citations"]) > 0
        citation = data["citations"][0]
        assert citation["filename"] == "security_policy.txt"
        assert "TLS 1.3" in citation["snippet"]
        assert data["answer"] == mock_llm_answer

        # Test multi-turn conversation continuation
        conv_id = data["conversation_id"]
        query_2 = await client.post(
            "/api/v1/rag/query",
            headers=auth_headers,
            json={"query": "How long can session tokens remain valid?", "conversation_id": conv_id}
        )
        assert query_2.status_code == 200
        data_2 = query_2.json()
        assert data_2["conversation_id"] == conv_id


@pytest.mark.asyncio
async def test_rag_conversations_api(client: AsyncClient, auth_headers: dict):
    # Perform a query to seed conversation
    with patch("app.services.ollama_service.OllamaService.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Test response."
        query_res = await client.post(
            "/api/v1/rag/query",
            headers=auth_headers,
            json={"query": "Hello RAG Assistant"}
        )
        conv_id = query_res.json()["conversation_id"]

    # List conversations
    list_res = await client.get("/api/v1/rag/conversations", headers=auth_headers)
    assert list_res.status_code == 200
    conv_list = list_res.json()
    assert any(c["id"] == conv_id for c in conv_list)

    # Get single conversation
    get_res = await client.get(f"/api/v1/rag/conversations/{conv_id}", headers=auth_headers)
    assert get_res.status_code == 200
    conv_detail = get_res.json()
    assert len(conv_detail["messages"]) >= 2
    assert conv_detail["messages"][0]["role"] == "user"
    assert conv_detail["messages"][1]["role"] == "assistant"

    # Delete conversation
    del_res = await client.delete(f"/api/v1/rag/conversations/{conv_id}", headers=auth_headers)
    assert del_res.status_code == 200

    # Ensure deleted
    get_res_after = await client.get(f"/api/v1/rag/conversations/{conv_id}", headers=auth_headers)
    assert get_res_after.status_code == 404
