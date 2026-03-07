"""Cohere embedding service — wraps langchain_cohere.CohereEmbeddings."""

from langchain_cohere import CohereEmbeddings

from mentat.core.logging import get_logger
from mentat.core.settings import settings

logger = get_logger(__name__)

_COHERE_MODEL = "embed-english-v3.0"
_EMBEDDING_DIMS = 1024


class EmbeddingService:
    """Thin wrapper around Cohere embeddings.

    Provides a single :meth:`embed` method that returns a 1024-dimensional
    vector for any input text.  The underlying model and API key are taken
    from application settings so callers never need to handle credentials.
    """

    def __init__(self, model: str = _COHERE_MODEL) -> None:
        logger.info("Initialising EmbeddingService (model=%s)", model)
        self._model = model
        self._embeddings = CohereEmbeddings(  # pyrefly: ignore[missing-argument]
            model=model,
            cohere_api_key=settings.cohere_api_key,  # type: ignore[arg-type]
        )
        logger.info("EmbeddingService ready.")

    def embed(self, text: str) -> list[float]:
        """Return a 1024-dimensional embedding vector for *text*.

        Args:
            text: The text to embed.

        Returns:
            List of 1024 floats (cosine-normalised by Cohere).
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
    def dims(self) -> int:
        """Number of embedding dimensions."""
        return _EMBEDDING_DIMS
