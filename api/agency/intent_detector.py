import logging
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.output_parsers import ChoiceOutputParser

from api.api_configurator import APIConfigurator
# Import the existing Intent enum
from api.interfaces import Intent

logger = logging.getLogger(__name__)

class IntentDetectionResponse(BaseModel):
    """Schema for the intent detection response."""
    intent: Intent = Field(description="The detected intent of the user's message")
    confidence: float = Field(description="Confidence score for the detected intent (0-1)")
    reasoning: str = Field(description="Explanation of why this intent was chosen")

class IntentDetector:
    """
    A class that uses LangChain's Choice pattern to detect user intent.
    """
    
    def __init__(self, config: APIConfigurator):
        """
        Initialize the intent detector.
        
        Args:
            llm_provider: The LLM provider to use for intent detection
        """
        self.llm_provider = config.llm_provider
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for an executive coaching application.
Your job is to analyze user messages and determine their primary intent.

Available intents are:
{intent_descriptions}

Based on the user's message, identify the most appropriate intent category.
Consider both explicit statements and implicit needs.
Default to "simple_message" only if no other intent clearly applies.
Be decisive - choose the best match even if multiple intents seem possible."""),
            ("human", "{input}")
        ])
        
        # Set up the classification chain using Choice pattern
        self._setup_classification_chain()
        
    def _setup_classification_chain(self):
        """
        Set up the classification chain using the Choice pattern.
        """
        # Get the LLM from the provider
        llm = self.llm_provider.llm()
        
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
    
    def detect(self, user_message: str) -> Optional[str]:
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
            
            # Return the string value of the intent
            return result.intent
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Fallback to simple response if there's an error
            return Intent.SIMPLE.value
    
    def add_intent(self, intent_name: str):
        """
        Add a new intent description to the intent detector.
        
        Args:
            intent_name: The name of the intent to add
        """
        # Note: This assumes that the intent has already been added to the Intent enum
        # We're just adding the description here
        if intent_name not in self.intent_descriptions:
            self.intent_descriptions[intent_name] = f"Intent related to {intent_name}"
            logger.info(f"Added new intent description: {intent_name}")
    
    def remove_intent(self, intent_name: str):
        """
        Remove an intent description from the intent detector.
        
        Args:
            intent_name: The name of the intent to remove
        """
        if intent_name in self.intent_descriptions:
            del self.intent_descriptions[intent_name]
            logger.info(f"Removed intent description: {intent_name}")