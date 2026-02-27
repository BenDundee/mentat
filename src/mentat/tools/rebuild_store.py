"""Rebuild the ChromaDB vector store after changing the embedding model.

Run with:
    uv run python -m mentat.tools.rebuild_store

This script:
  1. Reads the target embedding model from configs/rag.yml
  2. Dumps all documents + metadata from each existing collection
  3. Deletes the old collections
  4. Re-embeds everything with the new model
  5. Re-creates the collections with the new fingerprint

The original uploaded files are NOT touched.  Only the vector index is rebuilt.
"""

import sys

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from mentat.core.config import load_agent_config
from mentat.core.logging import get_logger, setup_logging
from mentat.core.vector_store import (
    _CHROMA_PATH,
    _COLLECTION_CONVERSATIONS,
    _COLLECTION_DOCUMENTS,
    _META_KEY,
)

setup_logging()
logger = get_logger(__name__)

_COLLECTIONS = [_COLLECTION_CONVERSATIONS, _COLLECTION_DOCUMENTS]


def rebuild_store(persist_path: str = _CHROMA_PATH) -> None:
    """Rebuild all ChromaDB collections using the embedding model in rag.yml.

    Args:
        persist_path: Path to the ChromaDB data directory.
    """
    rag_config = load_agent_config("rag")
    new_model: str = rag_config.extra_config.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Target embedding model: {new_model}")
    print(f"ChromaDB path:          {persist_path}")
    print()

    client = chromadb.PersistentClient(
        path=persist_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    print(f"Loading embedding model '{new_model}' (may download on first use)...")
    embedder = HuggingFaceEmbeddings(model_name=new_model)
    print("Embedding model loaded.\n")

    for collection_name in _COLLECTIONS:
        _rebuild_collection(client, embedder, collection_name, new_model)

    print("\nRebuild complete.")


def _rebuild_collection(
    client: chromadb.ClientAPI,
    embedder: HuggingFaceEmbeddings,
    collection_name: str,
    new_model: str,
) -> None:
    """Rebuild a single collection in-place.

    Args:
        client: ChromaDB client.
        embedder: HuggingFace embedder using the new model.
        collection_name: Name of the collection to rebuild.
        new_model: Name of the new embedding model (stored in metadata).
    """
    # Check whether the collection exists at all
    existing_names = [c.name for c in client.list_collections()]
    if collection_name not in existing_names:
        print(f"  [{collection_name}] does not exist — creating empty collection.")
        client.create_collection(collection_name, metadata={_META_KEY: new_model})
        return

    collection = client.get_collection(collection_name)
    old_model = (collection.metadata or {}).get(_META_KEY, "<unknown>")
    count = collection.count()

    print(f"  [{collection_name}] {count} documents, was indexed with '{old_model}'")

    if count == 0:
        print(f"  [{collection_name}] Empty — updating metadata only.")
        client.delete_collection(collection_name)
        client.create_collection(collection_name, metadata={_META_KEY: new_model})
        return

    if old_model == new_model:
        print(f"  [{collection_name}] Already uses '{new_model}' — skipping.")
        return

    # Dump all existing data
    print(f"  [{collection_name}] Dumping {count} documents...")
    result = collection.get(include=["documents", "metadatas"])
    ids: list[str] = result["ids"]
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or [{}] * len(ids)  # pyrefly: ignore[bad-assignment]

    # Re-embed
    print(f"  [{collection_name}] Re-embedding {len(documents)} documents...")
    new_embeddings = [embedder.embed_query(doc) for doc in documents]

    # Swap collection
    client.delete_collection(collection_name)
    new_collection = client.create_collection(
        collection_name, metadata={_META_KEY: new_model}
    )
    new_collection.add(
        ids=ids,
        embeddings=new_embeddings,  # pyrefly: ignore[bad-argument-type]
        documents=documents,
        metadatas=metadatas,  # pyrefly: ignore[bad-argument-type]
    )
    print(f"  [{collection_name}] Done — {len(ids)} documents re-indexed.")


if __name__ == "__main__":
    try:
        rebuild_store()
    except Exception as exc:
        logger.exception("Rebuild failed: %s", exc)
        sys.exit(1)
