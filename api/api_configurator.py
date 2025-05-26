from dataclasses import dataclass
from pathlib import Path
import yaml

from api.interfaces import LLMCredentials
from api.services.prompt_manager import PromptManager
from api.services import LLMProvider


BASE_DIR = Path(__file__).parent.parent


class APIConfigurator:

    def __init__(self):

        self.prompts_dir = BASE_DIR / "prompts"
        self.queries_dir = BASE_DIR / "queries"
        self.data_dir = BASE_DIR / "data"
        self.config_dir = BASE_DIR / "configs"

        # Individual configs:
        with open(self.config_dir / "api.yaml", "r") as f:
            self.llm_client_config = LLMCredentials(**yaml.safe_load(f))

        # Other
        self.prompt_manager = PromptManager()
        self.llm_provider = LLMProvider(self.llm_client_config)