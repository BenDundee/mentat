import logging
from typing import Dict, List, Optional, Tuple

from atomic_agents.lib.components.agent_memory import Message

from src.agent_flows import AgentFlows
from src.configurator import Configurator
from src.interfaces import Intent, CoachingSessionState
from src.services import RAGService, ConversationService


logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator):
        self.config = config
        # TODO: Consider whether rag service and convo svc need to be here or not. Can they be moved to `AgentFlows`?
        self.rag_service = RAGService(config)
        self.conversation = ConversationService(self.rag_service)
        self.agent_flows = AgentFlows(self.config, self.rag_service, self.conversation)
        logger.info("Controller initialized successfully")
    
    def get_response(self, user_message: str, conversation_id: Optional[str]) -> Tuple[str, str]:
        """Process user messages and generate responses.

        user_message (str): The latest user message.
        history (List[Message]): The conversation history up to this point.
        return (str): The generated response.
        """
        logger.info("Processing user request...")
        try:
            if conversation_id != self.conversation.state.conversation_id:
                # TODO: Write existing conversation to disk, initiate new conversation
                pass
            self.conversation.initiate_turn(user_message, conversation_id)

            # Update persona -- don't outsource this decision to Orchestration Agent
            if self.conversation.state.persona.is_empty():
                logger.info("Updating persona...")
                self.agent_flows.update_persona()

            # Orchestrate conversation
            logger.info("Orchestrating conversation...")
            self.agent_flows.orchestrate()
            if self.conversation.current_turn.confidence < 75: # TODO: Config???
                pass

            # Simple response: Invokes agent directly
            if self.conversation.current_intent() == Intent.SIMPLE:
                pass

            # Coaching session: Initiation
            elif self.conversation.state.detected_intent == Intent.COACHING_SESSION_REQUEST:
                # Coaching Session Management Agent -> CoachingAgent
                self.conversation.state.coaching_session = CoachingSessionState.get_new_session("12345")

            # Coaching session: Continuation
            elif self.conversation.state.detected_intent == Intent.COACHING_SESSION_RESPONSE:
                # Coaching Session Management Agent -> CoachingAgent
                pass

            elif self.conversation.state.detected_intent == Intent.FEEDBACK:
                # Critical Feedback Agent -> Coaching Agent
                pass

            else:
                self.conversation.state.response = \
                    get_message("assistant", "In the time of chimpanzees I was a monkey.", input.turn_id)

            logger.info("Response generated successfully")
            return self.conversation.advance_conversation()
            
        except Exception as e:
            # TODO: better error reporting, use `ConversationState.TurnState.errors`
            logger.error(f"Error processing request: {e}")
            return "assistant", "I apologize, but I encountered an error processing your request."

    def _store_conversation(self, messages: List[Dict]):
        """Store conversation data for future retrieval."""
        # TODO: Move this method to `RAGService`
        # TODO: Also fix this code
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
