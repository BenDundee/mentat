import logging
from typing import Dict, List, Optional

from atomic_agents.lib.components.agent_memory import Message

from src.agent_flows import AgentFlows
from src.configurator import Configurator
from src.services import RAGService, ConversationService


logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator):
        self.config = config
        self.rag_service = RAGService(config)
        self.conversation = ConversationService(self.rag_service)
        self.agent_flows = AgentFlows(self.config, self.rag_service)
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
                logger.info("Updating persona...")
                self.conversation.state.persona = self.agent_flows.update_persona()

            # Determine intent
            logger.info("Detecting intent of incoming message...")
            self.conversation.state.detected_intent = self.agent_flows.determine_intent(self.conversation.state)

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
        # TODO: Move this method to `RAGService`
        conversation_data = {
            "user_message": {"content": messages[-1]["content"]},
            "history": messages,
            "user_id": "default_user",  # Make this dynamic as needed
            "context": {}
        }
        #self.rag_service.add_conversation(conversation_data)
    



if __name__ == "__main__":

    from src.configurator import Configurator
    from src.utils import get_message

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    controller = Controller(Configurator())
    msg = get_message(role="user", message="Hello")
    output = controller.get_response(msg, [], "test")
    print("wait")
