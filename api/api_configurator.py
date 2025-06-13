import os
from pathlib import Path

from api.interfaces import LLMCredentials
from api.managers import (
    LLMManager, PromptManager, ConversationManager, DocumentManager,
    PersonaManager, QueryManager
)


BASE_DIR = Path(__file__).parent.parent

class APIConfigurator:

    def __init__(self):

        self.prompts_dir = BASE_DIR / "prompts"
        self.queries_dir = BASE_DIR / "queries"
        self.config_dir = BASE_DIR / "configs"
        self.data_dir = BASE_DIR / "data" / "processed"
        self.app_data_dir = BASE_DIR / "data" / "app_data"
        self.vector_db_dir = BASE_DIR / "data" / "app_data" / ".vector_db"

        # Individual configs: define in .env at top level
        self.llm_client_config = LLMCredentials(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # Manage the managers...
        # Refactor these so that there's only a single instance of the vector DB
        self.llm_manager = LLMManager(self.llm_client_config)
        self.prompt_manager = PromptManager()
        self.query_manager = QueryManager(self.queries_dir)
        self.conversation_manager = ConversationManager(self.vector_db_dir)
        self.document_manager = DocumentManager(self.vector_db_dir, self.data_dir)
        self.persona_manager = PersonaManager(self.app_data_dir)

        self.initialize()

    def initialize(self):

        # Docs
        # TODO: Move this to DocumentManager.__init__
        self.document_manager.load_processed()
