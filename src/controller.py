import logging
from typing import Dict, List, Optional

from atomic_agents.lib.components.agent_memory import Message

from src.agents import AgentHandler
from src.configurator import Configurator
from src.interfaces import SimpleMessageContentIOSchema
from src.managers import PromptManager
from src.managers.persona_manager import PersonaManager
from src.managers.query_manager import QueryManager
from src.services import RAGService, ConversationService


logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator):
        self.config = config
        self.rag_service = RAGService(config)
        self.conversation = ConversationService(self.rag_service)
        self.agent_handler = AgentHandler(self.config)
        self.prompt_manager = PromptManager()
        self.persona_manager = PersonaManager(config)
        self.query_manager = QueryManager()

        logger.info("Setting initial states...")
        self.agent_handler.initialize_agents(self.prompt_manager)
        
        logger.info("Controller initialized successfully")
    
    def get_response(self, input: Message, history: Optional[List[Message]], conversation_id: Optional[str]) -> Message:
        """Process user messages and generate responses.

        input (Message): The latest user message.
        history (List[Message]): The conversation history up to this point.
        return (str): The generated response.
        """
        logger.info("Processing user request...")
        try:
            self.conversation.initiate_turn(input, history, conversation_id)
            if self.conversation.state.persona.is_empty():
                self._update_persona()

            self.conversation.advance_conversation(response="In the time of chimpanzees I was a monkey")
            logger.info("Response generated successfully")
            return self.conversation.state.response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return get_message(
                role="assistant",
                message="I apologize, but I encountered an error processing your request.",
                turn_id=input.turn_id
            )

    def _store_conversation(self, messages: List[Dict]):
        """Store conversation data for future retrieval."""
        conversation_data = {
            "user_message": {"content": messages[-1]["content"]},
            "history": messages,
            "user_id": "default_user",  # Make this dynamic as needed
            "context": {}
        }
        #self.rag_service.add_conversation(conversation_data)
    
    def _update_persona(self):
        """Update user persona based on interactions."""
        logger.info("Updating persona...")
        self.agent_handler.persona_agent.get_context_provider("persona_context").clear()

        logger.debug("Generating queries...")
        persona_query = self.query_manager.get_query("persona_update")
        self.agent_handler.query_agent.get_context_provider("query_context").query_prompt = persona_query
        queries = self.agent_handler.query_agent.run().queries
        query = ', '.join(queries)
        self.agent_handler.query_agent.get_context_provider("query_context").clear()

        logger.debug("Running RAG...")
        query_result = self.rag_service.query_all_collections_combined(query)
        self.agent_handler.persona_agent.get_context_provider("persona_context").query_result = query_result
        persona = self.agent_handler.persona_agent.run()

        logger.debug("Updating internals...")
        self.conversation.state.persona = persona
        self.rag_service.add_persona_data(persona)


if __name__ == "__main__":

    from src.configurator import Configurator
    from src.utils import get_message

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    controller = Controller(Configurator())
    msg = get_message(role="user", message="Hello")
    output = controller.get_response(msg, [], "test")
    print("wait")
