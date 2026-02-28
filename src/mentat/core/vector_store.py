"""ChromaDB-backed vector store service for RAG retrieval."""

import uuid
from datetime import datetime, timezone

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from mentat.core.logging import get_logger
from mentat.core.models import DocumentChunk

logger = get_logger(__name__)


class EmbeddingModelMismatchError(RuntimeError):
    """Raised on startup when the configured embedding model differs from the
    model used to build the existing vector store.

    The store must be rebuilt before the application can start.
    Run:  uv run python -m mentat.tools.rebuild_store
    """


class VectorStoreService:
    """Wraps ChromaDB and HuggingFace embeddings for semantic search.

    Two collections are maintained (names are configurable via rag.yml):
    - ``collection_conversations``: past conversation turns
    - ``collection_documents``: uploaded document chunks

    Both collections are queried together in :meth:`search`.

    On startup the service validates that the configured ``embedding_model``
    matches the model recorded in each collection's metadata.  If there is a
    mismatch (or the collection has data but no recorded model) an
    :class:`EmbeddingModelMismatchError` is raised immediately rather than
    silently returning nonsensical results.

    All configurable values are injected via the constructor so they stay in
    ``configs/rag.yml`` as the single source of truth.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_path: str = "data/chroma",
        collection_conversations: str = "conversations",
        collection_documents: str = "documents",
        meta_key: str = "embedding_model",
    ) -> None:
        logger.info(
            "Initializing VectorStoreService (model=%s, path=%s)",
            embedding_model,
            persist_path,
        )
        self._embedding_model_name = embedding_model
        self._meta_key = meta_key
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._conversations = self._init_collection(collection_conversations)
        self._conversations_name = collection_conversations
        self._documents = self._init_collection(collection_documents)
        self._documents_name = collection_documents
        logger.info("VectorStoreService ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection, validating the embedding model fingerprint.

        - New empty collection: metadata is written now.
        - Existing collection with matching model: proceed normally.
        - Existing collection with a *different* model, or data present but no
          model recorded: raise :class:`EmbeddingModelMismatchError`.

        Args:
            name: Collection name.

        Returns:
            The validated ChromaDB collection.

        Raises:
            EmbeddingModelMismatchError: Fingerprint mismatch or untracked data.
        """
        # Pass our metadata so brand-new collections are tagged on creation.
        # For existing collections, ChromaDB ignores the metadata argument and
        # returns the stored collection unchanged.
        collection = self._client.get_or_create_collection(
            name,
            metadata={self._meta_key: self._embedding_model_name},
        )

        stored_model: str | None = (collection.metadata or {}).get(self._meta_key)

        if stored_model is None:
            # Collection existed before fingerprinting was introduced.
            if collection.count() > 0:
                raise EmbeddingModelMismatchError(
                    f"Collection '{name}' contains data but has no embedding model "
                    f"recorded. Re-index with: "
                    f"uv run python -m mentat.tools.rebuild_store"
                )
            # Empty + no metadata → stamp it now and proceed.
            collection.modify(metadata={self._meta_key: self._embedding_model_name})
            logger.debug(
                "Stamped new collection '%s' with model '%s'",
                name,
                self._embedding_model_name,
            )

        elif stored_model != self._embedding_model_name:
            raise EmbeddingModelMismatchError(
                f"Collection '{name}' was indexed with '{stored_model}' but "
                f"configs/rag.yml specifies '{self._embedding_model_name}'. "
                f"Re-index with: uv run python -m mentat.tools.rebuild_store"
            )

        return collection

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_conversation(self, text: str, metadata: dict[str, str]) -> str:
        """Store a conversation turn and return its generated ID.

        Args:
            text: The conversation text to embed and store.
            metadata: Arbitrary string metadata (e.g. session_id, timestamp).

        Returns:
            The generated document ID.
        """
        doc_id = str(uuid.uuid4())
        embedding = self._embed(text)
        self._conversations.add(
            ids=[doc_id],
            embeddings=[embedding],  # pyrefly: ignore[bad-argument-type]
            documents=[text],
            metadatas=[metadata],  # pyrefly: ignore[bad-argument-type]
        )
        logger.debug("Stored conversation turn id=%s", doc_id)
        return doc_id

    def add_documents(
        self, texts: list[str], metadatas: list[dict[str, str]]
    ) -> list[str]:
        """Store document chunks and return their generated IDs.

        Args:
            texts: List of text chunks to embed and store.
            metadatas: Parallel list of metadata dicts for each chunk.

        Returns:
            List of generated document IDs (same order as ``texts``).
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = [self._embed(t) for t in texts]
        self._documents.add(
            ids=ids,
            embeddings=embeddings,  # pyrefly: ignore[bad-argument-type]
            documents=texts,
            metadatas=metadatas,  # pyrefly: ignore[bad-argument-type]
        )
        logger.debug("Stored %d document chunks", len(texts))
        return ids

    def search(self, query: str, n_results: int = 5) -> tuple[DocumentChunk, ...]:
        """Search both collections and return the top results.

        Args:
            query: Semantic search query string.
            n_results: Maximum number of results per collection.

        Returns:
            Tuple of :class:`DocumentChunk` objects, most relevant first.
        """
        query_embedding = self._embed(query)
        chunks: list[DocumentChunk] = []

        for collection, source in [
            (self._conversations, self._conversations_name),
            (self._documents, self._documents_name),
        ]:
            count = collection.count()
            if count == 0:
                continue
            k = min(n_results, count)
            results = collection.query(
                query_embeddings=[query_embedding],  # pyrefly: ignore
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            for doc_list, meta_list in zip(documents, metadatas):
                for doc, meta in zip(doc_list, meta_list):
                    meta_str: dict[str, str] = (
                        {str(k): str(v) for k, v in meta.items()} if meta else {}
                    )
                    chunks.append(
                        DocumentChunk(
                            content=doc,
                            source=source,
                            document_id=meta_str.get("upload_id", ""),
                            metadata=meta_str,
                        )
                    )

        return tuple(chunks)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
