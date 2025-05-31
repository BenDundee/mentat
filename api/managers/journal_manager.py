# api/repositories/journal_repository.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from api.services.vector_store_service import VectorStoreService
from langchain_text_splitters import SpacyTextSplitter

logger = logging.getLogger(__name__)


class JournalEntry:
    """Model for a journal entry."""

    def __init__(self, user_id: str, content: str, entry_id: Optional[str] = None,
                 timestamp: Optional[str] = None, tags: Optional[List[str]] = None):
        self.user_id = user_id
        self.content = content
        self.entry_id = entry_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().isoformat()
        self.tags = tags or []


class JournalRepository:
    """Repository for storing and retrieving journal entries using a vector database."""

    def __init__(self, persist_directory="./vector_db"):
        """Initialize the journal repository with a vector store service."""
        self.vector_service = VectorStoreService(
            persist_directory=persist_directory,
            collection_name="journal_entries"
        )

    def save_entry(self, entry: JournalEntry) -> str:
        """Save a journal entry with hierarchical chunking for better context preservation."""
        # Prepare metadata
        metadata = {
            "user_id": entry.user_id,
            "document_type": "journal_entry",
            "timestamp": entry.timestamp,
            "tags": ",".join(entry.tags)
        }

        # Add to vector store with hierarchical chunking
        # This is especially useful for journal entries which can be lengthy
        entry_id = self.vector_service.add_text_hierarchical(
            text=entry.content,
            metadata=metadata,
            id=entry.entry_id
        )

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a full journal entry by its ID."""
        document = self.vector_service.get_full_hierarchy(entry_id)

        if not document:
            return None

        return {
            "id": document["id"],
            "content": document["content"],
            "timestamp": document["metadata"].get("timestamp"),
            "tags": document["metadata"].get("tags", "").split(",") if document["metadata"].get("tags") else [],
            "hierarchy": document.get("hierarchy")  # Include hierarchy information
        }

    def find_similar_entries(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find journal entries semantically similar to the query, with context."""
        filter_dict = {
            "user_id": user_id,
            "document_type": "journal_entry"
        }

        # Perform semantic search with hierarchical context
        results = self.vector_service.search_by_text(
            query=query,
            filter_metadata=filter_dict,
            limit=limit,
            include_hierarchy=True  # Include hierarchical context
        )

        return [
            {
                "id": result["id"],
                "content": result["content"],
                "timestamp": result["metadata"].get("timestamp"),
                "tags": result["metadata"].get("tags", "").split(",") if result["metadata"].get("tags") else [],
                "parent_context": result.get("parent", {}).get("content") if "parent" in result else None,
                "level": result.get("level", 0),
                "similarity_score": result["metadata"].get("score", 0)
            } for result in results
        ]

    def get_entries(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve journal entries for a user."""
        docs = self.vector_service.get_by_metadata({
            "user_id": user_id,
            "document_type": "journal_entry"
        }, limit=limit)

        entries = []
        for doc in docs:
            entries.append({
                "id": doc.metadata.get("id"),
                "content": doc.page_content,
                "timestamp": doc.metadata.get("timestamp"),
                "tags": doc.metadata.get("tags", "").split(",") if doc.metadata.get("tags") else []
            })

        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries

    def delete_entry(self, entry_id: str) -> None:
        """Delete a journal entry by its ID."""
        self.vector_service.delete_by_id(entry_id)

    # Add to VectorStoreService class methods
    def create_semantic_splitter(self, chunk_size=1000):
        return SpacyTextSplitter(chunk_size=chunk_size)