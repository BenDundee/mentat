import logging
from typing import Dict, List

from src.agents import AgentHandler
from src.managers import ManagerHandler
from src.configurator import Configurator
from src.types import ConversationState

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: Configurator):
        self.config = config

        logger.info("Handlers...")
        self.agent_handler = AgentHandler(self.config)
        self.manager_handler = ManagerHandler(self.config)

        logger.info("Initializing conversation...")
        self.conversation_state =ConversationState()

    def get_response(self, messages: List[Dict]) -> str:
        logger.info(f"Received input, updating memory...")
        self.agent_handler.update_memory(messages)

        return "Yo"


if __name__ == "__main__":

    config = Configurator()
    controller = Controller(config)
    controller.get_response({"role": "user", "content": "This is a test"})
    logger.info("Wait!")
