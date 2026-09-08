from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "Enterprise RAG Assistant"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/sqlite/rag_assistant.db"

    # Security & JWT
    JWT_SECRET_KEY: str = "enterprise-rag-assistant-secret-key-for-development-mode"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./data/chromadb"

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Ollama LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Document Processing & RAG Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 25
    TOP_K_RESULTS: int = 4
    UPLOAD_DIR: str = "./data/uploads"

    def ensure_directories(self) -> None:
        """Ensure necessary storage directories exist."""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        # For sqlite DB path
        if "sqlite" in self.DATABASE_URL:
            db_path_str = self.DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
            db_path = Path(db_path_str)
            if db_path.parent:
                db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
