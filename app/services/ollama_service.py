import logging
from typing import Any, Dict, List, Optional
import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self, base_url: Optional[str] = None, default_model: Optional[str] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS

    async def is_healthy(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/version")
                return res.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """Fetch list of available local Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    return [m for m in models if m]
        except Exception as e:
            logger.warning(f"Unable to fetch Ollama models: {e}")
        return []

    async def generate_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate completion using Ollama /api/generate endpoint.
        Falls back to a descriptive error if Ollama is unreachable.
        """
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                elif response.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{target_model}' not found. Please run `ollama pull {target_model}`."
                    )
                else:
                    raise RuntimeError(f"Ollama API returned status {response.status_code}: {response.text}")
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise RuntimeError(
                f"Ollama service is unreachable at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            )
        except httpx.TimeoutException:
            logger.error("Ollama request timed out.")
            raise RuntimeError("Ollama generation timed out. Try reducing context or choosing a smaller model.")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> str:
        """Generate response using Ollama /api/chat endpoint."""
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    msg = data.get("message", {})
                    return msg.get("content", "").strip()
                elif response.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{target_model}' not found. Run `ollama pull {target_model}`."
                    )
                else:
                    raise RuntimeError(f"Ollama chat returned status {response.status_code}: {response.text}")
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise RuntimeError(
                f"Ollama service is unreachable at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            )


_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service
