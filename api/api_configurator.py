from dataclasses import dataclass
from pathlib import Path
import yaml

from api.util.prompt_manager import PromptManager
from api.services import LLMProvider


BASE_DIR = Path(__file__).parent.parent

@dataclass
class LLMClientConfig:
    openai_api_key: str
    openrouter_api_key: str


class APIConfigurator:

    def __init__(self):

        self.prompts_dir = BASE_DIR / "prompts"
        self.queries_dir = BASE_DIR / "queries"
        self.data_dir = BASE_DIR / "data"
        self.config_dir = BASE_DIR / "configs"

        self.prompt_manager = PromptManager()
        self.llm_provider = LLMProvider()

        # Individual configs:
        with open(self.config_dir / "api.yaml", "r") as f:
            self.llm_client_config = LLMClientConfig(**yaml.safe_load(f))