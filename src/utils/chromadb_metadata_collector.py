from typing import Dict, Any, List
from pathlib import Path


class ChromaDBMetadataCollector:
    """
    Helper class to collect and format vector database metadata.

    This class provides methods to gather information about the vector database
    structure, available collections, schemas, and sample documents to help
    the QueryAgent generate more effective queries.
    """

    def __init__(self, vector_store_service):
        """
        Initialize the metadata collector.

        Args:
            vector_store_service: Vector database service instance
        """
        self.vector_store = vector_store_service

    def collect_metadata(self) -> Dict[str, Any]:
        """
        Collect comprehensive metadata about the vector database.

        Returns:
            Dictionary containing vector DB metadata
        """
        metadata = {
            "available_collections": self._get_available_collections(),
            "collection_schemas": self._get_collection_schemas(),
            "metadata_fields": self._get_metadata_fields(),
            "sample_documents": self._get_sample_documents(max_per_collection=2, max_length=300)
        }

        return metadata

    def _get_available_collections(self) -> List[str]:
        """Get list of available collections in the vector DB."""
        try:
            if hasattr(self.vector_store, 'chroma_client'):
                return [col.name for col in self.vector_store.chroma_client.list_collections()]
            return []
        except Exception:
            return []

    def _get_collection_schemas(self) -> Dict[str, Any]:
        """Get schema information for each collection."""
        schemas = {}
        collections = self._get_available_collections()
        for collection in collections:
            schemas[collection] = "Text documents with hierarchical chunking"

        return schemas

    def _get_metadata_fields(self) -> Dict[str, List[str]]:
        """Get available metadata fields that can be used for filtering."""
        fields = {}
        collections = self._get_available_collections()
        common_fields = ["document_type", "chunk_index", "is_root", "level"]

        for collection in collections:
            fields[collection] = common_fields

        return fields

    def _get_sample_documents(self, max_per_collection: int = 2, max_length: int = 300) -> Dict[str, List[str]]:
        """Get sample document excerpts from each collection."""
        samples = {}
        collections = self._get_available_collections()

        original_collection = self.vector_store.collection_name

        try:
            for collection in collections:
                # Set the collection
                self.vector_store.collection_name = collection

                # Try to get a few sample documents
                try:
                    results = self.vector_store.vector_store.get(limit=max_per_collection)
                    if results and hasattr(results, 'documents'):
                        samples[collection] = [doc[:max_length] for doc in results.documents]
                except Exception:
                    continue

        finally:
            self.vector_store.collection_name = original_collection

        return samples