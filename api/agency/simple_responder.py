import logging
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from api.api_configurator import APIConfigurator
from api.agency._agent import _Agent
from api.interfaces import SimpleResponderResponse, ConversationState

logger = logging.getLogger(__name__)


class SimpleResponder(_Agent):
    """Schema for the intent detection response."""

    def __init__(self, config: APIConfigurator):
        """ Initialize the IntentDetector with the LLM provider and prompt manager.

        :param config: The APIConfigurator object containing the LLM provider and prompt manager.
        """
        self.llm_provider = config.llm_provider
        self.llm_params = config.prompt_manager.get_llm_settings("simple_responder")
        self.prompt_template = config.prompt_manager.get_prompt("simple_responder").template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

    def run(self, state: ConversationState) -> ConversationState:
        """
        Generates a simple response to the user's message based on a provided conversation
        state. Leverages a language model to produce the response, and can process basic
        inputs like user messages and chat history. Handles exceptions during the response
        generation process, providing a fallback error message if needed.

        Args:
            state (ConversationState): The current conversation state, including the
                "user_message" and optionally "chat_history".

        Returns:
            ConversationState: The updated state containing the generated response.
        """
        try:
            response = (
                self.prompt
                | self.llm_provider.llm(self.llm_params)
            ).invoke({
                "input": state.user_message,
                "chat_history": state.history  # Empty for now can be extended to support history
            })
            state.response = response
            state.context["simple_responder_response"] = response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.errors.append(f"Error generating response: {e}")
            state.context["simple_responder_response"] = SimpleResponderResponse(
                response="I apologize, but I'm having trouble generating a response right now."
            )

        return state
