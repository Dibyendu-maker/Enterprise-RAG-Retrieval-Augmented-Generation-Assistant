import asyncio
import os
import shutil
import tempfile
from typing import AsyncGenerator
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.main import app
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStoreService, get_vector_store

# Use isolated temporary directories for tests
TEMP_DIR = tempfile.mkdtemp(prefix="rag_test_")
TEST_DB_PATH = os.path.join(TEMP_DIR, "test.db")
TEST_DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH}"
TEST_CHROMA_DIR = os.path.join(TEMP_DIR, "chroma")

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# Mock lightweight embedding service for fast deterministic tests
class MockEmbeddingService(EmbeddingService):
    def __init__(self):
        super().__init__(model_name="mock-model")

    async def embed_texts(self, texts):
        # Return deterministic 384-dim dummy vector
        return [[0.05] * 384 for _ in texts]

    async def embed_query(self, query: str):
        return [0.05] * 384

    def is_healthy(self) -> bool:
        return True


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Ensure vector store uses test chroma directory
    vs = get_vector_store()
    vs.persist_dir = TEST_CHROMA_DIR
    vs._client = None  # Reset client to use test path

    # Override embedding service singleton with mock for fast unit tests
    import app.services.embedding_service as es_mod
    es_mod._embedding_service = MockEmbeddingService()

    yield

    # Teardown
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
async def prepare_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    # Register user 1
    register_payload = {
        "email": "alice@enterprise.com",
        "password": "SecurePassword123!",
        "full_name": "Alice Developer"
    }
    await client.post("/api/v1/auth/register", json=register_payload)

    # Login user 1
    login_payload = {
        "email": "alice@enterprise.com",
        "password": "SecurePassword123!"
    }
    res = await client.post("/api/v1/auth/login-json", json=login_payload)
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def second_auth_headers(client: AsyncClient) -> dict[str, str]:
    # Register user 2
    register_payload = {
        "email": "bob@enterprise.com",
        "password": "AnotherPassword456!",
        "full_name": "Bob Analyst"
    }
    await client.post("/api/v1/auth/register", json=register_payload)

    # Login user 2
    login_payload = {
        "email": "bob@enterprise.com",
        "password": "AnotherPassword456!"
    }
    res = await client.post("/api/v1/auth/login-json", json=login_payload)
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
