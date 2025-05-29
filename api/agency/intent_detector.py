import logging
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from api.api_configurator import APIConfigurator
from api.interfaces import Intent, IntentDetectionResponse
from api.agency._agent import _Agent

logger = logging.getLogger(__name__)

class IntentDetector(_Agent):
    """
    A class that uses LangChain's Choice pattern to detect user intent.
    """
    
    def __init__(self, config: APIConfigurator):
        """ Initialize the IntentDetector with the LLM provider and prompt manager.
        
        :param config: The APIConfigurator object containing the LLM provider and prompt manager.
        """
        self.llm_provider = config.llm_provider
        self.llm_params = config.prompt_manager.get_llm_settings("intent_detector")
        self.prompt_template = config.prompt_manager.get_prompt("intent_detector").template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        self.classification_chain = (
            RunnablePassthrough.assign(intent_descriptions=lambda _: Intent.llm_rep())
            | self.prompt
            | self.llm_provider.llm(self.llm_params)
            | JsonOutputParser()
            | (lambda x: IntentDetectionResponse.parser(x))
        )
    
    def run(self, user_message: str) -> IntentDetectionResponse:
        """
        Detect the intent of a user message.
        
        Args:
            user_message: The message from the user
            
        Returns:
            The detected intent as a string
        """
        logger.info(f"Detecting intent for: '{user_message[:50]}...'")
        try:
            # Execute the classification chain
            result = self.classification_chain.invoke({"input": user_message, "chat_history": []})
            logger.info(f"Detected intent: {result.intent} with confidence {result.confidence}")
            logger.debug(f"Intent reasoning: {result.reasoning}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return IntentDetectionResponse(
                intent=Intent.SIMPLE,
                confidence=0,
                reasoning="I'm sorry, I couldn't understand your message. Please try again."
            )
