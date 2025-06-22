import logging
from typing import Dict, List, Optional

from atomic_agents.lib.components.agent_memory import Message

from src.agents import AgentHandler
from src.configurator import Configurator
from src.managers import PromptManager
from src.services import RAGService, ConversationService
from src.types import SimpleMessageContentIOSchema
from src.types.chat import ConversationState

logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator):
        self.config = config
        self.rag_service = RAGService(config)
        self.conversation = ConversationService(self.rag_service)
        self.agent_handler = AgentHandler(self.config)
        self.prompt_manager = PromptManager()

        logger.info("Setting initial states...")
        self.agent_handler.initialize_agents(self.prompt_manager)
        
        logger.info("Controller initialized successfully")
    
    def get_response(self, input: Message, history: List[Message], conversation_id: Optional[str]) -> str:
        """Process user messages and generate responses.

        input (Message): The latest user message.
        history (List[Message]): The conversation history up to this point.
        return (str): The generated response.
        """
        logger.info("Processing user request...")
        try:
            self._update_conversation_state(input, history, conversation_id)
            self._check_persona()
            
            # Generate response using RAG
            # response = self.rag_service.query(last_message)

            response = self.conversation.advance_conversation()
            logger.info("Response generated successfully")
            return "In the time of chimpanzees I was a monkey"
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _update_conversation_state(self, input: Message, history: List[Message], conversation_id: Optional[str]):
        # First check that the current conversation has the correct ID, if not start a new conversation
        if not self.conversation.state.conversation_id or self.conversation.state.conversation_id != conversation_id:
            logger.info("Starting new conversation...")
            self.conversation.state.conversation_id = conversation_id
            self.conversation.state.history = history
            self.conversation.state.user_message = input.content

        else:  # If conversation ID matches, update history
            # TODO: Figure out if this is needed and if so, what to do
            if set(history) ^ set(self.conversation.state.history):  # symmetric diff
                logger.error("Conversation history does not match, saving conversation and starting new one...")
            logger.info("Updating conversation history...")
            self.conversation.state.history = history

    def _check_persona(self):
        # Check if persona needs initialization
        if self.conversation.state.persona.is_empty():
            pass
            # self._update_persona(last_message)

    def _store_conversation(self, messages: List[Dict]):
        """Store conversation data for future retrieval."""
        conversation_data = {
            "user_message": {"content": messages[-1]["content"]},
            "history": messages,
            "user_id": "default_user",  # Make this dynamic as needed
            "context": {}
        }
        self.rag_service.add_conversation(conversation_data)
    
    def _update_persona(self, user_input: str):
        """Update user persona based on interactions."""
        logger.info("Updating persona...")
        
        # Query for persona-relevant information
        persona_query = f"What can we learn about the user from: {user_input}"
        persona_response = self.rag_service.query(persona_query)
        
        # Store persona data
        persona_data = {
            "content": persona_response,
            "user_id": "default_user",
            "metadata": {"source": "conversation_analysis"}
        }
        self.rag_service.add_persona_data(persona_data)
        
        # Update conversation state
        self.conversation_state.persona.update(persona_response)