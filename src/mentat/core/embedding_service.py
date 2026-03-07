"""Cohere embedding service — wraps langchain_cohere.CohereEmbeddings."""

from pathlib import Path

import yaml
from langchain_cohere import CohereEmbeddings

from mentat.core.logging import get_logger
from mentat.core.settings import settings

logger = get_logger(__name__)

_CONFIG_PATH = Path("configs/embedding.yml")


def _load_config() -> dict:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


class EmbeddingService:
    """Thin wrapper around Cohere embeddings.

    Model and dimensions are read from ``configs/embedding.yml``.
    API key is taken from application settings.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        self._model: str = cfg["model"]
        self._dims: int = cfg["dims"]
        logger.info("Initialising EmbeddingService (model=%s)", self._model)
        self._embeddings = CohereEmbeddings(  # pyrefly: ignore[missing-argument]
            model=self._model,
            cohere_api_key=settings.cohere_api_key,  # type: ignore[arg-type]
        )
        logger.info("EmbeddingService ready.")

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*.

        Args:
            text: The text to embed.

        Returns:
            List of floats (cosine-normalised by Cohere).
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
