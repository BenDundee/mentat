"""ChromaDB-backed vector store service for RAG retrieval."""

import uuid
from datetime import datetime, timezone

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from mentat.core.logging import get_logger
from mentat.core.models import DocumentChunk

logger = get_logger(__name__)

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHROMA_PATH = "data/chroma"
_COLLECTION_CONVERSATIONS = "conversations"
_COLLECTION_DOCUMENTS = "documents"


class VectorStoreService:
    """Wraps ChromaDB and HuggingFace embeddings for semantic search.

    Two collections are maintained:
    - ``conversations``: past conversation turns
    - ``documents``: uploaded document chunks

    Both collections are queried together in :meth:`search`.
    """

    def __init__(self, persist_path: str = _CHROMA_PATH) -> None:
        logger.info("Initializing VectorStoreService (path=%s)", persist_path)
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._conversations = self._client.get_or_create_collection(
            _COLLECTION_CONVERSATIONS
        )
        self._documents = self._client.get_or_create_collection(_COLLECTION_DOCUMENTS)
        logger.info("VectorStoreService ready.")

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

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
            (self._conversations, _COLLECTION_CONVERSATIONS),
            (self._documents, _COLLECTION_DOCUMENTS),
        ]:
            count = collection.count()
            if count == 0:
                continue
            k = min(n_results, count)
            results = collection.query(
                query_embeddings=[query_embedding],  # pyrefly: ignore[bad-argument-type]
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
