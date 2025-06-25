# src/services/rag_service.py
import chromadb
import logging
import simplejson as sj
from typing import Dict, Optional

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext

from src.configurator import Configurator
from src.types.chat import ConversationState


logger = logging.getLogger(__name__)


class RAGService:
    """Encapsulates all RAG/retrieval functionality using LlamaIndex."""

    def __init__(self, config: Configurator):
        self.config = config

        # Update LlamaIndex settings
        Settings.llm = OpenRouter(api_key=self.config.api_config.openrouter_key, model="google/gemini-2.0-flash-001")
        Settings.embed_model = \
            OpenAIEmbedding(model=self.config.data_config.embedding_model, api_key=self.config.api_config.openai_key)

        # Docstore Setup
        self.docstore_file = self.config.document_store_loc
        self.docstore_file.touch(exist_ok=True)
        self.docstore = SimpleDocumentStore()

        # Chroma Setup -- To add a new collection, add it to the list below and update self._setup_router()
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_db_dir.as_posix())
        self.collections = ["conversations", "documents", "persona"]
        self.indices = {
            c: VectorStoreIndex.from_vector_store(
                ChromaVectorStore(
                    chroma_collection=self.chroma_client.get_or_create_collection(c)
                )
            )  for c in self.collections
        }
        self.storage_contexts = {
            c: ChromaVectorStore(
                chroma_collection=self.chroma_client.get_or_create_collection(c)
            ) for c in self.collections
        }
        self.semantic_splitter = \
            SemanticSplitterNodeParser(embed_model=Settings.embed_model,breakpoint_percentile_threshold=95)
        self.query_engine = self._setup_router()

        # Load initial documents if needed
        self._load_initial_documents()

    def _load_initial_documents(self):
        """Load existing data files into LlamaIndex."""
        documents = []
        for file_path in self.config.data_files:
            try:
                with open(file_path, 'r') as f:
                    data = sj.load(f)
                documents.append(Document(text=data.get("content", ""), metadata=data.get("metadata", {})))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if documents:
            for doc in documents:
                self.indices["documents"].insert(doc)

    def _setup_router(self) -> RouterQueryEngine:
        """Set up router query engine."""
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.indices["conversations"].as_query_engine(similarity_top_k=5),
                name="conversations",
                description="Use this for queries about past conversations, user interactions, or chat history."
            ),
            QueryEngineTool.from_defaults(
                query_engine=self.indices["documents"].as_query_engine(similarity_top_k=5),
                name="documents",
                description="Use this for queries about general knowledge, documents, or factual information."
            ),
            QueryEngineTool.from_defaults(
                query_engine=self.indices["persona"].as_query_engine(similarity_top_k=3),
                name="persona",
                description="Use this for queries about user persona, preferences, or personal information."
            )
        ]

        return RouterQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            verbose=True
        )

    def add_full_document(self, content: str, metadata: Dict, collection_name: str = "documents") -> list:
        """Add document with chunking using LlamaIndex's integrated approach."""
        doc = Document(text=content, metadata=metadata)

        # Use semantic chunking for more intelligent splits
        nodes = self.semantic_splitter.get_nodes_from_documents([doc])
        
        # Add all chunks to the index
        doc_ids = []
        if collection_name not in self.indices:
            self.indices[collection_name] = VectorStoreIndex.from_documents(
                [], storage_context=self.storage_contexts[collection_name]
            )

        for node in nodes:
            self.indices[collection_name].insert(node)
            doc_ids.append(node.node_id)
            
        return doc_ids

    def get_full_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve using LlamaIndex's docstore."""
        return self.docstore.get_document(doc_id)

    def persist(self):
        """Persist all storage."""
        for storage_context in self.storage_contexts.values():
            storage_context.persist()

    def query(self, query_text: str) -> str:
        """Execute a query across all collections."""
        try:
            response = self.query_engine.query(query_text)
            return str(response)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return "I apologize, but I encountered an error processing your request."

    def add_conversation(self, conversation: ConversationState):
        """Add a conversation to the conversations index."""
        text_content = self._conversation_to_text(conversation_data)
        doc = Document(
            text=text_content,
            metadata={
                "type": "conversation",
                "user_id": conversation_data.get("user_id", "unknown"),
                "conversation_id": conversation_data.get("conversation_id", ""),
                **conversation_data.get("context", {})
            }
        )
        self.indices["conversations"].insert(doc)

    def add_persona_data(self, persona_data: Dict):
        """Add persona information to the persona index."""
        doc = Document(
            text=persona_data.get("content", ""),
            metadata={
                "type": "persona",
                "user_id": persona_data.get("user_id", "unknown"),
                **persona_data.get("metadata", {})
            }
        )
        self.indices["persona"].insert(doc)

    def _conversation_to_text(self, conversation_data: Dict) -> str:
        """Convert conversation data to searchable text."""
        parts = []

        if "user_message" in conversation_data:
            parts.append(f"User: {conversation_data['user_message'].get('content', '')}")

        if "history" in conversation_data:
            for msg in conversation_data["history"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if hasattr(content, 'get'):
                    content = content.get('content', str(content))
                parts.append(f"{role.title()}: {content}")

        return "\n".join(parts)

if __name__ == "__main__":
    # Initialize configuration and RAG service
    config = Configurator()
    rag_service = RAGService(config)

    # Add a document
    doc_content = "This is a sample document about AI technology."
    doc_metadata = {"type": "tech", "subject": "AI"}
    doc_id = rag_service.add_full_document(doc_content, doc_metadata)

    # Add conversation data
    conversation = {
        "user_message": {"content": "Tell me about AI"},
        "history": [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI is a fascinating field"}
        ],
        "user_id": "user123",
        "context": {"topic": "technology"}
    }
    rag_service.add_conversation(conversation)

    # Add persona data
    persona = {
        "content": "User shows interest in technology and AI",
        "user_id": "user123",
        "metadata": {"interests": ["technology", "AI"]}
    }
    rag_service.add_persona_data(persona)

    # Query across all collections
    query_result = rag_service.query("What do we know about AI?")
    print(f"Query result: {query_result}")

    # Retrieve the full document
    retrieved_doc = rag_service.get_full_document(doc_id)
    if retrieved_doc:
        print(f"Retrieved document: {retrieved_doc.text}")

    # Persist all changes
    rag_service.persist()