
import simplejson as sj
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from api.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


class DocumentManager:
    """
    Manager for document operations including loading, storing, and retrieving
    documents from a vector database.
    """

    def __init__(self, persist_directory=Path("./vector_db"), data_directory=DATA_DIR):
        """
        Initialize the document manager.

        Args:
            persist_directory: Directory for vector database persistence
            data_directory: Directory containing processed documents to load
            collection_name: Name of the vector collection for documents
        """
        self.persist_directory = persist_directory
        self.data_directory = data_directory

        # Initialize vector store service
        self.vector_service = VectorStoreService(
            persist_directory=persist_directory,
            collection_name="documents"
        )

    def load_processed(self, use_hierarchical_chunking: bool = True) -> int:
        """
        Load all documents from the data directory or a subdirectory.

        Args:
            use_hierarchical_chunking: Whether to use hierarchical chunking

        Returns:
            Number of documents loaded
        """
        # Load each file
        documents_loaded = 0
        for file_path in self.data_directory.glob("*.json"):
            try:
                documents_loaded += \
                    self._load_and_store_document(file_path, use_hierarchical_chunking=use_hierarchical_chunking)
                logger.info(f"Loaded document: {file_path}")
            except Exception as e:
                # logger.error(f"Error loading document {file_path}: {e}")
                raise e

        logger.info(f"Loaded {documents_loaded} documents from {self.data_directory} ")
        return documents_loaded

    def _load_and_store_document(self, file_path: Path, use_hierarchical_chunking: bool = True) -> int:
        """
        Load a single document and store it in the vector database.

        Args:
            file_path: Path to the document file
            use_hierarchical_chunking: Whether to use hierarchical chunking

        Returns:
            1 if successful, 0 otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = sj.load(f)

            # Add file info to metadata
            data["metadata"].update({
                'source_path': file_path.as_posix(),
                'filename': file_path.name,
                'indexed_at': datetime.now().isoformat()
            })

            # Store the document
            self._store_document(
                content=(data.get("content") or data.get("document")),
                metadata=data["metadata"],
                document_id=str(uuid4()),
                use_hierarchical_chunking=use_hierarchical_chunking
            )

            return 1

        except Exception as e:
            #logger.error(f"Error processing document {file_path}: {e}")
            #return 0
            raise e

    def _store_document(self,
                        content: str,
                        metadata: Dict[str, Any],
                        document_id: str,
                        use_hierarchical_chunking: bool = True) -> str:
        """
        Store a document in the vector database with appropriate chunking.

        Args:
            content: Document content
            metadata: Document metadata
            document_id: Unique document ID
            use_hierarchical_chunking: Whether to use hierarchical chunking

        Returns:
            Document ID
        """
        if use_hierarchical_chunking and len(content) > 1000:
            # Use hierarchical chunking for longer documents
            self.vector_service.add_text_hierarchical(
                text=content,
                metadata=metadata,
                id_=document_id
            )
        else:
            # Use regular chunking for shorter documents
            self.vector_service.add_text_hierarchical(
                text=content,
                metadata=metadata,
                id_=document_id,
            )

        return document_id

    def add_document(self,
                     content: str,
                     metadata: Dict[str, Any],
                     document_id: Optional[str] = None,
                     use_hierarchical_chunking: bool = True) -> str:
        """
        Add a new document to the vector database.

        Args:
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID (generated if not provided)
            use_hierarchical_chunking: Whether to use hierarchical chunking

        Returns:
            Document ID
        """
        doc_id = document_id or str(uuid4())
        metadata['added_at'] = datetime.now().isoformat()

        return self._store_document(
            content=content,
            metadata=metadata,
            document_id=doc_id,
            use_hierarchical_chunking=use_hierarchical_chunking
        )

    def find_relevant_documents(self,
                                query: str,
                                filter_metadata: Optional[Dict[str, Any]] = None,
                                limit: int = 5,
                                include_hierarchy: bool = True) -> List[Dict[str, Any]]:
        """
        Find documents relevant to a query.

        Args:
            query: Search query
            filter_metadata: Optional metadata filter
            limit: Maximum number of results
            include_hierarchy: Whether to include hierarchical context

        Returns:
            List of relevant document results
        """
        # Set default filter for documents
        if filter_metadata is None:
            filter_metadata = {
                "document_type": {"$contains": "document"}  # Match any document type
            }

        # Perform semantic search
        results = self.vector_service.search_by_text(
            query=query,
            filter_metadata=filter_metadata,
            limit=limit,
            include_hierarchy=include_hierarchy
        )

        # Process results
        processed_results = []
        for result in results:
            processed_result = {
                "id": result["id"],
                "content": result["content"],
                "metadata": result["metadata"],
                "similarity_score": result["metadata"].get("score", 0)
            }

            # Add parent context if available
            if "parent" in result:
                processed_result["parent_context"] = {
                    "id": result["parent"]["id"],
                    "content": result["parent"]["content"]
                }

            # Add children context if available
            if "children" in result:
                processed_result["children"] = result["children"]

            processed_results.append(processed_result)

        return processed_results

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector database.

        Args:
            document_id: Document ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_service.delete_by_id(document_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

if __name__ == "__main__":
    dm = DocumentManager()
    load = dm.load_processed()
    print("wait!")

