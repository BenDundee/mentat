from dataclasses import dataclass
from instructor import from_openai, Instructor, Mode
import logging
from openai import OpenAI
from pathlib import Path
from typing import Dict
from yaml import safe_load, dump

from src.agents.types import Persona

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BASE_DIR = Path(__file__).parent.parent


@dataclass
class ConnectionConfig:
    host: str = "localhost"
    port: int = 8080
    endpoint: str = None
    debug: bool = False


@dataclass
class DeploymentConfig:
    api_cfg: ConnectionConfig = None
    app_cfg: ConnectionConfig = None
    api: str = None
    app: str = None


@dataclass
class APIConfig:
    openai_key: str = None
    openrouter_key: str = None
    openrouter_endpoint: str = None


@dataclass
class DataConfig:
    db_chunk_size: int = None
    db_chunk_overlap: int = None
    db_batch_size: int = None
    db_directory: str = None
    db_name: str = None
    embedding_model: str = None
    embedding_model_tokenizer: str = None
    distance_metric: str = None
    db_max_queries: int = None
    db_results_per_query: int = None
    search_results_per_query: int = None
    search_engine_num_queries: int = None


@dataclass
class UserConfig:
    user_name: str = None
    address_as: str = None


def assert_present(filename: Path):
    try:
        assert filename.exists()
    except AssertionError:
        raise Exception (f"Cound not find {filename} in config directory")


class Configurator:

    def __init__(self):
        self.base_dir = BASE_DIR

        # Configs
        self.config_dir = BASE_DIR / "config"
        self.config_files = list(self.config_dir.glob("*.yml"))

        # User-provided data
        self.data_dir = BASE_DIR / "data" / "processed"
        self.data_files = list(self.data_dir.glob("*.json"))

        # App data
        self.app_data = BASE_DIR / "data" / "app_data"
        self.app_data_files = list(self.app_data.glob("*.yml"))

        # ALL THE CONFIGS
        self.deployment_config = self.configure_deployment()
        self.api_config: APIConfig = APIConfig()
        self.agent_config_map: Dict[str, str] = {}
        self.data_config: DataConfig = DataConfig()
        self.user_config: UserConfig = UserConfig()
        self.persona: Persona = Persona()
        self.configured = False
        self.configure()

    def configure(self, override_configuration=False):
        if override_configuration or not self.configured:
            self.api_config = self.__load_api_yml()
            self.agent_config_map = self.__load_agents_yml()
            self.data_config = self.__load_data_yml()
            self.user_config = self.__load_user_yml()
            self.persona = self.__load_persona()
            self.configured = True

    def __load_api_yml(self):
        api_config = self.config_dir / "api.yml"
        assert_present(api_config)
        with open(api_config, "r") as f:
            return APIConfig(**safe_load(f))

    def __load_data_yml(self):
        data_config = self.config_dir / "data.yml"
        assert_present(data_config)
        with open(data_config, "r") as f:
            data_config = DataConfig(**safe_load(f))

        # Check for absolute/relative path for DB directory
        # TODO: Move to DataConfig class
        if not Path(data_config.db_directory).is_absolute():
            data_config.db_directory = str(self.base_dir / data_config.db_directory)

        return data_config

    def __load_agents_yml(self):
        agent_config = self.config_dir / "agents.yml"
        assert_present(agent_config)
        with open(agent_config, "r") as f:
            return {a["agent_name"]: a["prompt"] for a in safe_load(f)["agents"]}

    def get_openrouter_client(self) -> Instructor:
        return from_openai(
            OpenAI(
                base_url=self.api_config.openrouter_endpoint,
                api_key=self.api_config.openrouter_key
            )
            , mode=Mode.JSON
        )

    def configure_deployment(self) -> DeploymentConfig:
        app_cfg = ConnectionConfig(host="localhost", port=8080)
        api_cfg = ConnectionConfig(host="localhost", port=8081)
        if (self.config_dir / "deployment.yml").exists():
            logger.info("Loading server config from file")
            with open(self.config_dir / "deployment.yml", "r") as f:
                config = safe_load(f)
                app_cfg = ConnectionConfig(**config["app"])
                api_cfg = ConnectionConfig(**config["api"])

        return DeploymentConfig(
            app_cfg=app_cfg,
            api_cfg=api_cfg,
            api=f"http://{api_cfg.host}:{api_cfg.port}/{api_cfg.endpoint}?debug={api_cfg.debug}",
            app=f"http://{app_cfg.host}:{app_cfg.port}/{app_cfg.endpoint}?debug={app_cfg.debug}"
        )

    def __load_user_yml(self):
        user_config = self.config_dir / "user.yml"
        assert_present(user_config)
        with open(user_config, "r") as f:
            return UserConfig(**safe_load(f))

    def __load_persona(self):
        persona_config = self.app_data / "persona.yml"
        if not persona_config.exists():
            return Persona.get_empty_persona()
        else:
            with open(persona_config, "r") as f:
                return Persona(**safe_load(f))

    def update_persona(self, persona: Persona):
        self.persona = persona
        with open(self.app_data / "persona.yml", "w") as f:
            dump(self.persona.get_summary(), f, sort_keys=False, indent=2)








