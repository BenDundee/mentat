import logging
from typing import Dict, List, Callable, TypeVar, Optional
from langchain.schema import ChatMessage
from langchain_core.messages import SystemMessage

from api.api_configurator import APIConfigurator
from api.interfaces import Intent, ConversationState
from api.agency import IntentDetector, SimpleResponder, CoachingSession, SemanticQueryAgent


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=ConversationState)

class Controller:
    def __init__(self, config: APIConfigurator):
        self.config = config
        self.intent_detector = IntentDetector(config)

        self._intent_to_workflow_map: Dict[Intent, Callable[[T], T]] = {}
        self._register_known_workflows()

        # Create Persona if not
        if self.config.persona_manager.persona.is_empty():
            raw_query = self.config.query_manager.get_query("create_persona")
            queries = SemanticQueryAgent(config).run({"input": raw_query})
            results = self.config

    def _register_known_workflows(self):
        """
        Registers the workflows the orchestrator knows about.
        This is where you'll add more workflows in the future.
        """
        self._intent_to_workflow_map[Intent.SIMPLE] = SimpleResponder(self.config).run
        self._intent_to_workflow_map[Intent.COACHING_SESSION_REQUEST] = CoachingSession(self.config).run
        self._intent_to_workflow_map[Intent.COACHING_SESSION_RESPONSE] = CoachingSession(self.config).run

    def process_message(
        self,
        user_message: str,
        history: List[ChatMessage],
        user_id: str = "guest",
        session_id: Optional[str] = None,
    ) -> ConversationState:
        """
        Main entry point to handle a user's message.
        """
        logger.info(f"Orchestrator processing message from {user_id}: '{user_message[:50]}...'")
        state = ConversationState(
            user_message=user_message,
            user_id=user_id,
            history=history,
            detected_intent=None,
            response=None,
            conversation_id=session_id,
        )
        state = self.intent_detector.run(state)
        workflow_handler = self._intent_to_workflow_map.get(state.detected_intent)
        if workflow_handler:
            logger.info(f"Found handler for intent '{state.detected_intent}'. Executing workflow...")
            try:
                state = workflow_handler(state)
            except Exception as e:
                logger.error(f"Error during execution of workflow for intent '{state.detected_intent}': {e}", exc_info=True)
                state.errors.append(f"Error during execution of workflow for intent '{state.detected_intent}': {e}")
                state.response = SystemMessage(content="I'm sorry, but I'm having trouble processing your request right now.")
        else:
            logger.warning(f"No workflow handler found for intent: '{state.detected_intent}'")
            state.errors.append(f"No workflow handler found for intent: '{state.detected_intent}'")
            state.response = SystemMessage(content="I'm sorry, but I'm having trouble processing your request right now.")

        logger.info(f"Orchestrator finished...")
        return state
