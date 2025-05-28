import logging
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from api.api_configurator import APIConfigurator
from api.interfaces import Intent
from api.agency import _Agent

logger = logging.getLogger(__name__)

class IntentDetectionResponse(BaseModel):
    """Schema for the intent detection response."""
    intent: Intent = Field(description="The detected intent of the user's message")
    confidence: float = Field(description="Confidence score for the detected intent (0-1)")
    reasoning: str = Field(description="Explanation of why this intent was chosen")

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
        
        # Set up the classification chain using Choice pattern
        self._setup_classification_chain()
        
    def _setup_classification_chain(self):
        """
        Set up the classification chain using the Choice pattern.
        """
        # Get the LLM from the provider
        llm = self.llm_provider.llm(self.llm_params)
        
        # Create the classification chain
        self.classification_chain = (
            RunnablePassthrough.assign(
                intent_descriptions=lambda _: Intent.llm_rep()
            )
            | self.prompt
            | llm.bind(
                functions=[{
                    "name": "classify_intent",
                    "description": "Classify the intent of the user message",
                    "parameters": IntentDetectionResponse.model_json_schema()
                }],
                function_call={"name": "classify_intent"}
            )
            | PydanticOutputFunctionsParser(pydantic_schema=IntentDetectionResponse)
        )
    
    def run(self, user_message: str) -> Optional[str]:
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
            result = self.classification_chain.invoke({"input": user_message})
            logger.info(f"Detected intent: {result.intent} with confidence {result.confidence}")
            logger.debug(f"Intent reasoning: {result.reasoning}")
            return result.intent
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Fallback to simple response if there's an error
            return Intent.SIMPLE.value
