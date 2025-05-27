import logging
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.output_parsers import JsonOutputParser

from api.interfaces import Intent, IntentDetectionResponse

logger = logging.getLogger(__name__)


class IntentDetector:
    """
    A class that uses LangChain's modern agent patterns to detect user intent.
    """
    
    def __init__(self, llm_provider):
        """
        Initialize the intent detector.
        
        Args:
            llm_provider: The LLM provider to use for intent detection
        """
        self.llm_provider = llm_provider
        self.intents = [i for i in Intent]
        
        # Initialize the parser
        self.parser = JsonOutputParser(pydantic_object=IntentDetectionResponse)
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for an executive coaching application.
Your job is to analyze user messages and determine their primary intent.

Available intents are:
{intents}

Based on the user's message, identify the most appropriate intent category.
Respond with a JSON object that includes the intent name, your confidence level (0-1),
and a brief explanation of your reasoning.

Considerations:
- Focus on the primary purpose of the message, not secondary elements
- Consider both explicit statements and implicit needs
- Default to "simple_response" only if no other intent clearly applies
- Be decisive - choose the best match even if multiple intents seem possible"""),
            ("human", "{input}")
        ])
        
        # Create the detect intent tool
        self.detect_intent_tool = Tool(
            name="detect_intent",
            description="Detects the intent of a user message",
            func=self._detect_intent_impl
        )
        
        # Set up the agent using modern patterns
        self._setup_agent()
        
    def _detect_intent_impl(self, user_message: str) -> Dict[str, Any]:
        """
        Implementation of the intent detection logic.
        
        Args:
            user_message: The message from the user
            
        Returns:
            A dictionary with the detected intent information
        """
        # Get the LLM from the provider
        llm = self.llm_provider.llm()
        
        # Create a chain for intent detection using modern pipe syntax
        intent_chain = self.prompt | llm | self.parser
        
        # Execute the chain
        return intent_chain.invoke({
            "intents": "\n".join([f"- {intent}" for intent in self.intents]),
            "input": user_message
        })
    
    def _setup_agent(self):
        """
        Set up the agent using modern LangChain patterns.
        """
        from langchain.agents import create_react_agent, AgentExecutor
        
        # Get the LLM from the provider
        llm = self.llm_provider.llm()
        
        # Define tools the agent can use
        tools = [self.detect_intent_tool]
        
        # Create the agent with create_react_agent (modern pattern)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intent detection agent. Use the tools to analyze user messages."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
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
            # For simplicity in this initial implementation, we'll use the direct implementation
            # rather than the full agent. As your system grows, you can switch to using the agent
            # for more complex scenarios.
            result = self._detect_intent_impl(user_message)
            
            logger.info(f"Detected intent: {result.intent} with confidence {result.confidence}")
            logger.debug(f"Intent reasoning: {result.reasoning}")
            
            return result.intent
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Fallback to simple response if there's an error
            return "simple_response"
    
    def add_intent(self, intent_name: str):
        """
        Add a new intent to the list of recognized intents.
        
        Args:
            intent_name: The name of the intent to add
        """
        if intent_name not in self.intents:
            self.intents.append(intentname)
            logger.info(f"Added new intent: {intent_name}")
    
    def remove_intent(self, intent_name: str):
        """
        Remove an intent from the list of recognized intents.
        
        Args:
            intent_name: The name of the intent to remove
        """
        if intent_name in self.intents:
            self.intents.remove(intent_name)
            logger.info(f"Removed intent: {intent_name}")