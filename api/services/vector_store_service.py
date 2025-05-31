# api/services/vector_store_service.py
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
from datetime import datetime

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

logger = logging.getLogger(__name__)

class HierarchicalChunk:
    """Represents a hierarchical chunk structure with parent-child relationships."""
    def __init__(self,
                 content: str,
                 metadata: Dict[str, Any],
                 id_: str,
                 level: int = 0,
                 parent_id: Optional[str] = None):
        self.content = content
        self.metadata = metadata
        self.id_ = id_
        self.level = level
        self.parent_id = parent_id
        self.child_ids = []
    
    def add_child(self, child_id: str):
        """Add a child chunk ID to this chunk."""
        self.child_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "metadata": {
                **self.metadata,
                "level": self.level,
                "parent_id": self.parent_id,
                "child_ids": ",".join(self.child_ids) if self.child_ids else ""
            },
            "id": self.id_
        }

class VectorStoreService:
    """Service for interacting with ChromaDB vector database with hierarchical chunking support."""
    
    def __init__(self, 
                 persist_directory="./vector_db", 
                 collection_name="default",
                 chunk_sizes=(2000, 1000, 500),  # Hierarchical chunk sizes from largest to smallest
                 chunk_overlap_ratio=0.1):
        """Initialize the vector store service with ChromaDB."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap_ratio = chunk_overlap_ratio
        
        # Initialize embedding function
        self.embedding_function = OpenAIEmbeddings()
        
        # Initialize text splitters for each level
        self.text_splitters = [
            RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=int(size * chunk_overlap_ratio),
                separators=["\n\n", "\n", ". ", " ", ""]
            ) for size in chunk_sizes
        ]
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Also create a separate collection for metadata
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.metadata_collection = self.chroma_client.get_or_create_collection(f"{collection_name}_metadata")
    
    def _create_hierarchical_chunks(
            self,
            text: str,
            metadata: Dict[str, Any],
            base_id: str
    ) -> List[HierarchicalChunk]:
        """Create hierarchical chunks from text."""
        all_chunks = []
        
        # Start with the document as the top level (level 0)
        root_chunk = HierarchicalChunk(
            content=text,
            metadata={**metadata, "is_root": True},
            id_=base_id,
            level=0
        )
        all_chunks.append(root_chunk)
        
        # Process each level of chunking
        parent_chunks = [root_chunk]
        for level, splitter in enumerate(self.text_splitters, 1):
            next_level_chunks = []
            
            for parent_chunk in parent_chunks:
                # Split this parent into children
                child_texts = splitter.split_text(parent_chunk.content)
                
                # Skip further splitting if we only get one chunk
                if len(child_texts) <= 1:
                    continue
                
                # Create child chunks
                for i, child_text in enumerate(child_texts):
                    child_id = f"{parent_chunk.id_}_L{level}_C{i}"
                    child_chunk = HierarchicalChunk(
                        content=child_text,
                        metadata={
                            **metadata,
                            "chunk_index": i,
                            "total_chunks_at_level": len(child_texts),
                            "hierarchical_path": f"{parent_chunk.id_}->{child_id}"
                        },
                        id_=child_id,
                        level=level,
                        parent_id=parent_chunk.id_
                    )
                    
                    # Add bi-directional references
                    parent_chunk.add_child(child_id)
                    next_level_chunks.append(child_chunk)
                    all_chunks.append(child_chunk)
            
            # Move to the next level
            parent_chunks = next_level_chunks
            
            # If we have no more parent chunks to process, stop
            if not parent_chunks:
                break
        
        return all_chunks
    
    def add_text_hierarchical(
        self,
        text: str,
        metadata: Dict[str, Any],
        id_: Optional[str] = None
    )-> str:
        """
        Adds a text to the vector store in a hierarchical structure for better organization
        of long texts. Short texts are directly added without hierarchical chunking. The
        function also updates the metadata collection to include information about the
        document hierarchy.

        Arguments:
            text (str): The text content to be added to the vector store.
            metadata (Dict[str, Any]): Metadata associated with the text, such as additional
                properties or attributes to categorize the text.
            id_ (Optional[str]): An optional unique identifier for the text. If not provided,
                a new UUID will be generated.

        Returns:
            str: The unique identifier for the processed text entry in the vector store.
        """
        base_id = id_ or str(uuid.uuid4())
        
        # Skip hierarchical chunking for short texts
        if len(text) < self.chunk_sizes[0]:
            self.vector_store.add_texts(
                texts=[text],
                metadatas=[{**metadata, "is_root": True, "level": 0}],
                ids=[base_id]
            )
            return base_id

        # Prepare data for vector store
        texts = []
        metadatas = []
        ids = []
        for chunk in self._create_hierarchical_chunks(text, metadata, base_id):
            chunk_dict = chunk.to_dict()
            texts.append(chunk_dict["content"])
            metadatas.append(chunk_dict["metadata"])
            ids.append(chunk_dict["id"])
        
        # Add all chunks to the vector store
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # Store the document hierarchy in the metadata collection
        self.metadata_collection.upsert(
            ids=[base_id],
            metadatas=[{
                **metadata,
                "document_type": f"{metadata.get('document_type', 'document')}_hierarchical",
                "chunk_count": len(ids),
                "levels": len(self.chunk_sizes),
                "root_id": base_id
            }]
        )
        
        return base_id
    
    def search_by_text(self, 
                      query: str, 
                      filter_metadata: Optional[Dict[str, Any]] = None, 
                      limit: int = 5,
                      fetch_k: int = 20,
                      include_hierarchy: bool = True) -> List[Document]:
        """
        Search for text with hierarchical context expansion.
        
        Args:
            query: The search query
            filter_metadata: Optional metadata filter
            limit: Number of final results to return
            fetch_k: Number of initial results to fetch before processing
            include_hierarchy: Whether to include parent/child context
            
        Returns:
            List of result documents with hierarchical context
        """
        # Initial search to find the most relevant chunks
        docs = self.vector_store.similarity_search(
            query=query,
            k=fetch_k,  # Get more initial results
            filter=filter_metadata
        )
        
        if not docs or not include_hierarchy:
            return docs[:limit]
        
        # Expand results with hierarchical context
        expanded_results = []
        seen_ids = set()
        
        for doc in docs:
            if doc.metadata.get("id") in seen_ids:
                continue
                
            # Get this document with its context
            result = self._get_with_hierarchical_context(doc)
            expanded_results.append(result)
            seen_ids.add(result["id"])
            
            # Also add any parent documents to seen_ids to avoid duplication
            if "parent" in result:
                seen_ids.add(result["parent"]["id"])
        
        # Return the top results after expansion
        return expanded_results[:limit]
    
    def _get_with_hierarchical_context(self, doc: Document) -> Dict[str, Any]:
        """Get a document with its hierarchical context (parent and children)."""
        result = {
            "id": doc.metadata.get("id"),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "level": doc.metadata.get("level", 0)
        }
        
        # Get parent if this is not a root document
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_docs = self.vector_store.get(
                ids=[parent_id]
            )
            if parent_docs:
                result["parent"] = {
                    "id": parent_docs[0].metadata.get("id"),
                    "content": parent_docs[0].page_content,
                    "level": parent_docs[0].metadata.get("level", 0)
                }
        
        # Get children if any
        child_ids_str = doc.metadata.get("child_ids", "")
        if child_ids_str:
            child_ids = child_ids_str.split(",")
            child_docs = self.vector_store.get(ids=child_ids)
            if child_docs:
                result["children"] = [
                    {
                        "id": child.metadata.get("id"),
                        "content": child.page_content,
                        "level": child.metadata.get("level", 0)
                    } for child in child_docs
                ]
        
        return result
    
    def get_full_hierarchy(self, root_id: str) -> Dict[str, Any]:
        """
        Retrieve a complete document hierarchy.
        
        Args:
            root_id: The ID of the root document
            
        Returns:
            Complete hierarchical structure with all levels
        """
        # First, get the root document
        root_docs = self.vector_store.get(
            ids=[root_id]
        )
        
        if not root_docs:
            return None
        
        root_doc = root_docs[0]
        
        # Initialize the result with the root
        result = {
            "id": root_id,
            "content": root_doc.page_content,
            "metadata": root_doc.metadata,
            "level": 0
        }
        
        # Get all chunks associated with this root document
        all_docs = self.vector_store.get(
            where={"hierarchical_path": {"$contains": root_id}}
        )
        
        # Build the hierarchy
        levels = {}
        for doc in all_docs:
            level = doc.metadata.get("level", 0)
            if level not in levels:
                levels[level] = []
            
            levels[level].append({
                "id": doc.metadata.get("id"),
                "content": doc.page_content,
                "parent_id": doc.metadata.get("parent_id"),
                "children": []
            })
        
        # Sort levels from top to bottom
        sorted_levels = sorted(levels.items())
        
        # Add the hierarchy information
        result["hierarchy"] = {
            "levels": sorted_levels,
            "total_levels": len(sorted_levels)
        }
        
        return result
    
    def get_with_context_window(self, chunk_id: str, window_size: int = 1) -> Dict[str, Any]:
        """
        Get a chunk with surrounding context window.
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            window_size: Number of siblings to include on each side
            
        Returns:
            Chunk with its context window
        """
        # Get the chunk
        chunk_docs = self.vector_store.get(
            ids=[chunk_id]
        )
        
        if not chunk_docs:
            return None
        
        chunk_doc = chunk_docs[0]
        chunk_metadata = chunk_doc.metadata
        
        # Get parent to find siblings
        parent_id = chunk_metadata.get("parent_id")
        if not parent_id:
            # This might be a root document
            return {
                "id": chunk_id,
                "content": chunk_doc.page_content,
                "metadata": chunk_metadata
            }
        
        # Get the parent
        parent_docs = self.vector_store.get(
            ids=[parent_id]
        )
        
        if not parent_docs:
            return {
                "id": chunk_id,
                "content": chunk_doc.page_content,
                "metadata": chunk_metadata
            }
        
        # Get sibling chunks
        child_ids_str = parent_docs[0].metadata.get("child_ids", "")
        if not child_ids_str:
            return {
                "id": chunk_id,
                "content": chunk_doc.page_content,
                "metadata": chunk_metadata,
                "parent": {
                    "id": parent_id,
                    "content": parent_docs[0].page_content
                }
            }
        
        # Get all siblings
        sibling_ids = child_ids_str.split(",")
        
        # Find the position of this chunk among siblings
        try:
            chunk_index = sibling_ids.index(chunk_id)
        except ValueError:
            chunk_index = 0
        
        # Calculate the window range
        start_idx = max(0, chunk_index - window_size)
        end_idx = min(len(sibling_ids), chunk_index + window_size + 1)
        
        window_ids = sibling_ids[start_idx:end_idx]
        
        # Get the window chunks
        window_docs = self.vector_store.get(
            ids=window_ids
        )
        
        # Sort window by chunk index
        window_chunks = sorted(
            window_docs, 
            key=lambda x: x.metadata.get("chunk_index", 0)
        )
        
        # Assemble the result
        result = {
            "id": chunk_id,
            "content": chunk_doc.page_content,
            "metadata": chunk_metadata,
            "parent": {
                "id": parent_id,
                "content": parent_docs[0].page_content
            },
            "context_window": [
                {
                    "id": doc.metadata.get("id"),
                    "content": doc.page_content,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "is_current": doc.metadata.get("id") == chunk_id
                } for doc in window_chunks
            ]
        }
        
        return result

