import logging
from pydantic import BaseModel, Field
from typing import Optional, Any

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from api.api_configurator import APIConfigurator
from api.agency import _Agent

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

    def run(self, user_message: str) -> Optional[str]:
        """
        Generate a simple response to the user's message.
        
        Args:
            user_message: The message from the user
            
        Returns:
            The generated response as a string
        """
        logger.info(f"Generating simple response for: '{user_message[:50]}...'")
        try:
            # Get the LLM from the provider
            llm = self.llm_provider.llm(self.llm_params)
            
            # Create a simple response chain
            response_chain = (
                self.prompt 
                | llm
            )
            
            # Execute the response chain
            response = response_chain.invoke({
                "input": user_message,
                "chat_history": []  # Empty for now, can be extended to support history
            })
            
            # Extract the text from the response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated response: '{response_text[:50]}...'")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now."