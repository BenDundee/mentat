import logging
from typing import Dict, List, Optional

from atomic_agents.lib.components.agent_memory import Message

from src.agents import AgentHandler
from src.configurator import Configurator
from src.managers import PromptManager
from src.services.rag_service import RAGService
from src.types.chat import ConversationState

logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator):
        self.config = config
        self.rag_service = RAGService(config)
        self.agent_handler = AgentHandler(self.config)
        self.prompt_manager = PromptManager()

        logger.info("Setting initial states...")
        self.agent_handler.initialize_agents(self.prompt_manager)
        self.conversation_state = ConversationState()
        
        logger.info("Controller initialized successfully")
    
    def get_response(self, input: Message, history: List[Message], conversation_id: Optional[str]) -> str:
        """Process user messages and generate responses.

        input (Message): The latest user message.
        history (List[Message]): The conversation history up to this point.
        return (str): The generated response.
        """
        logger.info("Processing user request...")
        
        try:
            # First check that the current conversation has the correct ID, if not start a new conversation
            # -- if no conversaiton ID exists generate one

            # If conversation ID matches, update history
            
            # Check if persona needs initialization
            if self.conversation_state.persona.is_empty():
                pass
                # self._update_persona(last_message)
            
            # Generate response using RAG
            # response = self.rag_service.query(last_message)
            
            logger.info("Response generated successfully")
            return "In the time of chimpanzees I was a monkey"
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "I apologize, but I encountered an error processing your request."
    
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