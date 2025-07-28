import logging
from typing import Optional, Tuple

from src.agent_flows import AgentFlows
from src.configurator import Configurator
from src.interfaces import Intent, CoachingSessionState, Persona
from src.services import RAGService, ConversationService
from src.utils import get_message


logger = logging.getLogger(__name__)

class Controller:
    """Main application controller - orchestrates high-level business logic."""
    
    def __init__(self, config: Configurator, debug_mode=False):
        self.config = config
        # TODO: Consider whether rag service and convo svc need to be here or not. Can they be moved to `AgentFlows`?
        self.rag_service = RAGService(config, initialize=(not debug_mode))
        self.conversation = ConversationService(self.config, self.rag_service)
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
            self.conversation.initiate_turn(user_message, conversation_id)

            if self.conversation.state.persona.is_empty():
                logger.info("Updating persona...")
                self.agent_flows.update_persona()

            logger.info("Orchestrating conversation...")
            self.agent_flows.orchestrate()
            if self.conversation.current_turn.intent_confidence < 75: # TODO: Config???
                pass

            if (len(self.conversation.current_turn.action_directives) == 0 and
                self.conversation.current_intent() == Intent.SIMPLE
            ):
                logger.info("Generating simple response...")
                self.agent_flows.generate_simple_response()

            else:
                logger.info("Executing actions...")
                for directive in self.conversation.current_turn.action_directives:
                    pass

            # self.conversation.state.response = \
            #    get_message("assistant", "In the time of chimpanzees I was a monkey.", "test")

            logger.info("Response generated successfully")
            return self.conversation.advance_conversation()
            
        except Exception as e:
            # TODO: better error reporting, use `ConversationState.TurnState.errors`
            logger.error(f"Error processing request: {e}")
            raise e
            #return "assistant", "I apologize, but I encountered an error processing your request."

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    controller = Controller(Configurator(), True)
    output = controller.get_response(
        user_message="""
            Hello, my name is Ben. I am excited to be working with you. I have provided you with several documents that 
            will help you understand my background and my career trajectory. I’d like your help in taking my career to 
            the next level. Please let me know what questions I can answer for you.
        """
        , conversation_id="test"
    )
    print("wait")
