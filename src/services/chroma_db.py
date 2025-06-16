import chromadb
import itertools as it
from itertools import zip_longest as itzip
import logging
import os
import shutil
import simplejson as sj
import tiktoken as tt
from typing import Dict, List, Optional, Tuple, Any, Type, TypeVar
import uuid
import json

from src.configurator import Configurator
from src.types import ConversationState, Chunk, QueryResult, SearchResultItem
from src.types.chat import ConversationState, SimpleMessageContentIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory, Message

logger = logging.getLogger(__name__)


# https://github.com/openai/openai-python/issues/519
# https://platform.openai.com/docs/api-reference/embeddings/create
# The following must be true (API limitation @ OpenAI)
#  > The input parameter may not take a list longer than 2048 elements (chunks of text).
#  > The total number of tokens across all list elements of the input parameter cannot exceed 1,000,000.
#  > Rate cannot exceed 1M tokens/minute
#  > Each individual array element (chunk of text) cannot be more than 8191 tokens.
#  > no element in the list should be BLANK/EMPTY/NULL content in the input parameter (list of paragraph)
#
# Error is:
# `Error during initialization: APIStatusError.__init__() missing 2 required keyword-only arguments: 'response' and 'body'`
_BATCH_SIZE = 200  # May run into errors if chunk size is large

T = TypeVar('T', bound=ConversationState)

class ChromaDBService:
    """Service for interacting with ChromaDB using OpenAI embeddings."""

    def __init__(self, config: Configurator, collection_name: str = "default", recreate_collection: bool = False):
        """
        Initializes an object to handle embedding and collection management for OpenAI and ChromaDB.

        :param config: Configuration object that contains all necessary configurations such as API keys
            and embedding model information.
        :type config: Configurator
        :param recreate_collection: Flag to indicate whether to recreate the collection if it already exists.
        :type recreate_collection: bool, optional
        """
        self.config = config

        # Initialize embedding function with OpenAI
        self.embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.config.api_config.openai_key
            , model_name=self.config.data_config.embedding_model
        )

        # If recreating, delete the entire persist directory
        if recreate_collection and os.path.exists(self.config.chroma_db_dir):
            shutil.rmtree(self.config.chroma_db_dir)
            os.makedirs(self.config.chroma_db_dir)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.config.chroma_db_dir.as_posix())

        # Get or create a collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.config.data_config.distance_metric},  # Explicitly set distance metric
        )

        # Tokenizer
        self.encode = lambda x: tt.get_encoding(self.config.data_config.embedding_model_tokenizer).encode(x)
        self.decode = lambda x: tt.get_encoding(self.config.data_config.embedding_model_tokenizer).decode(x)
        self.get_tokens = lambda x: len(self.encode(x))

    def initialize_db(self) -> List[str]:
        data = []
        for fn in self.config.data_files:
            with open(fn, 'r+') as f:
                data.append(sj.load(f))

        chunks, metadatas = [], []
        for d in data:
            try:
                _ch, _md = self.get_chunks_and_metadatas(d)
            except Exception as e:
                raise Exception(f"error reading file with metadata {d['metadata']}, details follow ... {e}")
            chunks.extend(_ch)
            metadatas.extend(_md)

        return self.add_documents(documents=chunks, metadatas=metadatas)

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, str]],
        ids: Optional[List[str]] = None,
        chunk_documents: bool = True
    ) -> List[str]:
        """Add documents to the collection.

        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document. If not provided, UUIDs will be generated.

        Returns:
            List[str]: The IDs of the added documents
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        # https://github.com/openai/openai-python/issues/519
        assert not any(d is None for d in documents)  # no null dox!

        # Chunk if necessary
        _documents, _metadatas, _ids = [], [], []
        if chunk_documents:
            for (_d, _m, _i) in zip(documents, metadatas, ids):
                try:
                    _d_chunk, _md_chunk = self.get_chunks_and_metadatas({"document": _d, "metadata": metadatas[documents.index(_d)]})
                    _documents.extend(_d_chunk)
                    _metadatas.extend(_md_chunk)
                    _ids.extend([_i]*len(_d_chunk))
                except Exception as e:
                    logger.error(f"error reading file with metadata {metadatas[documents.index(_d)]}, details follow ... {e}")
                    continue
        else:
            _documents, _metadatas, _ids = documents, metadatas, ids

        _ds = it.batched(_documents, _BATCH_SIZE)
        _ms = it.batched(_metadatas, _BATCH_SIZE)
        _is = it.batched(_ids, _BATCH_SIZE)
        for (_d, _m, _i) in zip(_ds, _ms, _is):
            self.collection.add(documents=list(_d), metadatas=list(_m), ids=list(_i))
        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, str]] = None,
        ids: Optional[List[str]] = None,
    ) -> QueryResult:
        """Query the collection for similar documents.

        Args:
            query_text: Text to find similar documents for
            n_results: Number of results to return
            where: Optional filter criteria
            ids: Optional list of IDs to filter by

        Returns:
            QueryResult containing documents, metadata, distances and IDs
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=max(1, min(n_results, self.get_count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return QueryResult(
            results = [
                Chunk(text=doc, metadata=md, distance=dist,id=id) for id, doc, md, dist 
                in zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
            ])

    def get_chunks_and_metadatas(self, file_json: dict) -> Tuple[List[str], List[Dict[str, str]]]:
        """Split the document into chunks with overlap. Factor this out?"""
        document = self.encode(file_json["document"])
        chunk_size = self.config.data_config.db_chunk_size
        overlap = self.config.data_config.db_chunk_overlap

        def _yield_chunks():
            for x in itzip(*[document[i::chunk_size - overlap] for i in range(chunk_size)]):
                yield tuple(i for i in x if i is not None) if x[-1] is None else x

        chunks = [self.decode(x) for x in _yield_chunks()]
        return chunks, [file_json["metadata"]]*len(chunks)


if __name__ == "__main__":
    config = Configurator()

    chroma_db_service = ChromaDBService(config, recreate_collection=True)
    added_ids = chroma_db_service.add_documents(
        documents=["Hello, world!", "This is a test document."],
        metadatas=[{"source": "test"}, {"source": "test"}],
    )
    print("Added documents with IDs:", added_ids)

    results = chroma_db_service.query(query_text="Hello, world!")
    print("Query results:", results)

    chroma_db_service.delete_by_ids([added_ids[0]])
    print("Deleted document with ID:", added_ids[0])

    updated_results = chroma_db_service.query(query_text="Hello, world!")
    print("Updated results after deletion:", updated_results)