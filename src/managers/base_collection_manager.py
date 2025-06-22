from abc import ABC, abstractmethod
from typing import TypeVar, Type, List, Optional, Dict, Any
from pydantic import BaseModel
from uuid import uuid4
import logging
import simplejson as sj

from src.services.chroma_db import ChromaDBService
from src.configurator import Configurator

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BaseCollectionManager(ABC):
    """Base class for collection managers that handle specific domain objects."""

    def __init__(self, config: Configurator, collection_name: str = "default"):
        """
        Initialize the manager with a ChromaDB service and collection name.

        Args:
            chroma_service: The underlying ChromaDB service
            collection_name: Name of the collection this manager handles
        """
        self.collection_name = collection_name
        self.chroma_service = ChromaDBService(config, collection_name=self.collection_name)
        self.doc_store_dir = config.document_store_dir / self.collection_name
        self.doc_store_dir.mkdir(parents=True, exist_ok=True)

        # Get or create the specific collection
        self.collection = self.chroma_service.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_service.embedding_function,
            metadata={"hnsw:space": self.chroma_service.config.data_config.distance_metric}
        )

    @abstractmethod
    def get_model_class(self) -> Type[T]:
        """Return the Pydantic model class this manager handles."""
        pass

    @abstractmethod
    def get_document_type(self) -> str:
        """Return the document type identifier for filtering."""
        pass

    def _write_to_store(self, model: T, metadata: Dict[str, Any] = None, id: str = None) -> None:
        serlialization_pkg = {
            "metadata": metadata,
            "content": model.model_dump_json(serialize_as_any=True)
        }
        with open(self.doc_store_dir / f"{id}.json", "w+") as f:
            sj.dump(serlialization_pkg, f)

    def add(self, model: T) -> str:
        """Add a model instance to the collection."""
        # Generate ID if not present
        model_id = getattr(model, f"{self.get_document_type()}_id", None)
        if not model_id:
            model_id = str(uuid4())
            setattr(model, f"{self.get_document_type()}_id", model_id)

        # Prepare metadata
        metadata = {
            "type": self.get_document_type(),
            "model_type": model.__class__.__name__,
            f"{self.get_document_type()}_id": model_id
        }

        # Add domain-specific metadata
        metadata.update(self._extract_metadata(model))

        # Cache intermediate object (chroma is bad at this), then make it searchable (chroma is good at this)
        self._write_to_store(model, metadata=metadata, id=model_id)
        self._add_searchable_content(model_id, model, metadata)

        return model_id

    def get_by_id(self, model_id: str) -> Optional[T]:
        """Retrieve a model by its ID."""

        # Full doc should live in document store
        expected_location = self.doc_store_dir / f"{model_id}.json"
        if not expected_location.exists():
            return None

        try:
            with open(expected_location, "r+") as f:
                raw = sj.load(f)
            return self.get_model_class().model_validate_json(raw["content"])
        except Exception as e:
            logger.error(f"Error deserializing {self.get_document_type()}: {e}")
            return None

    def update(self, model: T) -> bool:
        """Update an existing model."""
        model_id = getattr(model, f"{self.get_document_type()}_id", None)
        if not model_id:
            logger.error(f"Cannot update {self.get_document_type()} without ID")
            return False

        # Delete existing and re-add
        self.collection.delete(ids=[f"{self.get_document_type()}:{model_id}"])
        self.add(model)
        return True

    def delete(self, model_id: str) -> bool:
        """Delete a model by ID."""
        try:
            self.collection.delete(ids=[f"{self.get_document_type()}:{model_id}"])
            # Delete from store as well?
            return True
        except Exception as e:
            logger.error(f"Error deleting {self.get_document_type()}: {e}")
            return False

    def find_by_filter(self, filter_dict: Dict[str, Any]) -> List[T]:
        """Find models matching metadata criteria."""
        results = self.collection.get(
            where=filter_dict,
            include=["documents"]
        )

        models = []
        for doc in results["documents"]:
            try:
                model = self.get_model_class().model_validate_json(doc)
                models.append(model)
            except Exception as e:
                logger.error(f"Error deserializing {self.get_document_type()}: {e}")

        return models

    def semantic_search(self, query_text: str, n_results: int = 5) -> List[T]:
        """Find models using semantic search."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"type": f"{self.get_document_type()}_chunk"},
            include=["metadatas"]
        )

        # Extract unique model IDs
        model_ids = set()
        for metadata in results["metadatas"][0]:
            if f"{self.get_document_type()}_id" in metadata:
                model_ids.add(metadata[f"{self.get_document_type()}_id"])

        # Retrieve full models
        models = []
        for model_id in model_ids:
            model = self.get_by_id(model_id)
            if model:
                models.append(model)

        return models

    @abstractmethod
    def _extract_metadata(self, model: T) -> Dict[str, Any]:
        """Extract searchable metadata from the model."""
        pass

    @abstractmethod
    def _add_searchable_content(self, model_id: str, model: T, metadata: Dict[str, Any]) -> None:
        """Add searchable content chunks if applicable."""
        pass