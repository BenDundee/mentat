
# TODO: Do we need access to the raw classes? Maybe?
from .conversation_collection_manager import ConversationCollectionManager
from .prompt_manager import PromptManager


#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
from src.configurator import Configurator


class ManagerHandler:
    def __init__(self, config: Configurator):

        self.config = config

        self.prompt_manager = PromptManager()
        self.conversation_manager = ConversationCollectionManager(config)