import logging
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from api.api_configurator import APIConfigurator
from api.agency._agent import _Agent
from api.interfaces import SimpleResponderResponse, ConversationState

logger = logging.getLogger(__name__)


class CoachingSession(_Agent):
    """Schema for the coaching session response."""
    def __init__(self, config: APIConfigurator):
        """ Initialize the CoachingSession with the LLM provider and prompt manager."""
        self.llm_provider = config.llm_provider
        self.llm_params = config.prompt_manager.get_llm_settings("coaching_session")
        self.prompt_template = config.prompt_manager.get_prompt("coaching_session").template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
