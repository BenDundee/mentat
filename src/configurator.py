from dataclasses import dataclass
from instructor import from_openai, Instructor, Mode
import logging
from openai import OpenAI
from pathlib import Path
from typing import Dict
from yaml import safe_load

from src.interfaces import APIConfig, DataConfig, DeploymentConfig, ConnectionConfig, Persona

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BASE_DIR = Path(__file__).parent.parent


def assert_present(filename: Path):
    try:
        assert filename.exists()
    except AssertionError:
        raise Exception (f"Cound not find {filename} in config directory")


class Configurator:

    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = BASE_DIR / "config"
        self.config_files = list(Path(self.config_dir).glob("*.yml"))
        self.data_dir = BASE_DIR / "data" / "processed"
        self.data_files = list(Path(self.data_dir).rglob("**/*.json"))
        self.app_data = BASE_DIR / "data" / "app_data"
        self.chroma_db_dir = self.app_data / ".chroma_db"
        self.document_store_loc = self.app_data / ".docstore"

        # ALL THE CONFIGS
        self.deployment_config = self.configure_deployment()
        self.api_config: APIConfig = APIConfig()
        self.agent_config_map: Dict[str, str] = {}
        self.data_config: DataConfig = DataConfig()
        self.configured = False
        self.configure()

    def configure(self, override_configuration=False):
        if override_configuration or not self.configured:
            self.api_config = self.__load_api_yml()
            self.agent_config_map = self.__load_agents_yml()
            self.data_config = self.__load_data_yml()
            self.persona = self.__load_persona_yml()
            self.configured = True

    def __load_api_yml(self):
        api_config = self.config_dir / "api.yml"
        assert_present(api_config)
        with open(api_config, "r") as f:
            return APIConfig(**safe_load(f))

    def __load_data_yml(self) -> DataConfig:
        data_config = self.config_dir / "data.yml"
        assert_present(data_config)
        with open(data_config, "r") as f:
            return DataConfig(**safe_load(f))

    def __load_agents_yml(self) -> Dict[str, str] :
        agent_config = self.config_dir / "agents.yml"
        assert_present(agent_config)
        with open(agent_config, "r") as f:
            return {a["agent_name"]: a["prompt"] for a in safe_load(f)["agents"]}

    def __load_persona_yml(self) -> Persona :
        persona_config = self.config_dir / "persona.yml"
        if not persona_config.exists():
            return Persona.get_empty_persona()
        with open(persona_config, "r") as f:
            return Persona(**safe_load(f))

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
