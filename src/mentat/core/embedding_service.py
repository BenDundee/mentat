"""OpenRouter embedding service — wraps langchain_openai.OpenAIEmbeddings."""

from pathlib import Path

import yaml
from langchain_openai import OpenAIEmbeddings

from mentat.core.logging import get_logger
from mentat.core.settings import settings

logger = get_logger(__name__)

_CONFIG_PATH = Path("configs/embedding.yml")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


class EmbeddingService:
    """Thin wrapper around OpenRouter embeddings.

    Model and dimensions are read from ``configs/embedding.yml``.
    Uses the existing ``OPENROUTER_API_KEY`` — no separate API key required.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        self._model: str = cfg["model"]
        self._dims: int = cfg["dims"]
        logger.info("Initialising EmbeddingService (model=%s)", self._model)
        self._embeddings = OpenAIEmbeddings(
            model=self._model,
            openai_api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
            openai_api_base=_OPENROUTER_BASE_URL,  # pyrefly: ignore[unexpected-keyword]
        )
        logger.info("EmbeddingService ready.")

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*.

        Args:
            text: The text to embed.

        Returns:
            List of floats (cosine-normalised).
        """
        return self._embeddings.embed_query(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts in a single API call.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        return self._embeddings.embed_documents(texts)

    @property
    def model(self) -> str:
        """Embedding model identifier."""
        return self._model

    @property
    def dims(self) -> int:
        """Number of embedding dimensions."""
        return self._dims
