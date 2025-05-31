from pathlib import Path
import yaml

from api.interfaces import LLMCredentials
from api.managers import LLMManager, PromptManager, ConversationManager, DocumentManager


BASE_DIR = Path(__file__).parent.parent


class APIConfigurator:

    def __init__(self):

        self.prompts_dir = BASE_DIR / "prompts"
        self.queries_dir = BASE_DIR / "queries"
        self.config_dir = BASE_DIR / "configs"
        self.data_dir = BASE_DIR / "data" / "processed"
        self.vector_db_dir = BASE_DIR / "data" / "app_data" / ".vector_db"

        # Individual configs:
        with open(self.config_dir / "api.yaml", "r") as f:
            self.llm_client_config = LLMCredentials(**yaml.safe_load(f))

        # I AMT THE MANAGER MANAGER!
        self.llm_manager = LLMManager(self.llm_client_config)
        self.prompt_manager = PromptManager()
        self.conversation_manager = ConversationManager(self.vector_db_dir)
        self.document_manager = DocumentManager(self.vector_db_dir, self.data_dir)

        self.initialize()

    def initialize(self):
        self.document_manager.load_processed()