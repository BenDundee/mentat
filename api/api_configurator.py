from pathlib import Path
import yaml

from api.interfaces import LLMCredentials
from api.managers import LLMManager, PromptManager


BASE_DIR = Path(__file__).parent.parent


class APIConfigurator:

    def __init__(self):

        self.prompts_dir = BASE_DIR / "prompts"
        self.queries_dir = BASE_DIR / "queries"
        self.data_dir = BASE_DIR / "data"
        self.config_dir = BASE_DIR / "configs"
        self.db_loc = self.data_dir / "dbs"

        # Individual configs:
        with open(self.config_dir / "api.yaml", "r") as f:
            self.llm_client_config = LLMCredentials(**yaml.safe_load(f))

        # Other
        self.llm_provider = LLMManager(self.llm_client_config)
        self.prompt_manager = PromptManager()