import logging
from typing import Dict, Any, Callable

from api.services import LLMProvider
from api.api_configurator import APIConfigurator
from api.agency.intent_detector import IntentDetector

logger = logging.getLogger(__name__)

# --- Workflow Abstraction (Minimal: A Callable) ---
# A workflow is a function that takes the current state (including user message)
# and dependencies (like an LLM provider), and returns an updated state with an output.
WorkflowHandler = Callable[[Dict[str, Any], APIConfigurator], Dict[str, Any]]


class WorkflowOrchestrator:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.intent_detector = IntentDetector(llm_provider)
        self._intent_to_workflow_map: Dict[str, WorkflowHandler] = {}
        self._register_known_workflows()

    def _register_known_workflows(self):
        """
        Registers the workflows the orchestrator knows about.
        This is where you'll add more workflows in the future.
        """
        self.add_workflow("simple_response", handle_simple_response)
        # You can add more workflows as they're implemented
        # self.add_workflow("goal_setting", handle_goal_setting)
        # self.add_workflow("feedback", handle_feedback)
        # self.add_workflow("action_planning", handle_action_planning)
        # self.add_workflow("reflection", handle_reflection)

    def add_workflow(self, intent_name: str, handler: WorkflowHandler):
        """
        Allows adding or updating workflow handlers. Key for extensibility.
        """
        self._intent_to_workflow_map[intent_name] = handler
        # Ensure the intent detector knows about this intent
        self.intent_detector.add_intent(intent_name)
        logger.info(f"Workflow for intent '{intent_name}' registered.")

    def process_message(self, user_message: str, user_id: str = "guest") -> Dict[str, Any]:
        """
        Main entry point to handle a user's message.
        """
        logger.info(f"Orchestrator processing message from {user_id}: '{user_message[:50]}...'")

        # 1. Determine Intent using the intent detector
        intent = self.intent_detector.detect(user_message)

        # 2. Prepare Initial State
        current_state = {
            "user_message": user_message,
            "user_id": user_id,
            "detected_intent": intent,
            "history": []  # Placeholder for future conversation history
        }

        if not intent:
            logger.warning("No intent could be determined.")
            current_state["output"] = "I'm not sure how to help with that."
            return current_state

        # 3. Select and Execute Workflow
        workflow_handler = self._intent_to_workflow_map.get(intent)

        if workflow_handler:
            logger.info(f"Found handler for intent '{intent}'. Executing workflow...")
            try:
                # Pass the LLM provider to the handler
                final_state = workflow_handler(current_state, self.llm_provider)
            except Exception as e:
                logger.error(f"Error during execution of workflow for intent '{intent}': {e}", exc_info=True)
                final_state = {**current_state, "output": f"An error occurred while handling your '{intent}' request."}
        else:
            logger.warning(f"No workflow handler found for intent: '{intent}'")
            final_state = {**current_state, "output": f"I don't know how to handle a '{intent}' request yet."}

        logger.info(f"Orchestrator finished. Final output: '{final_state.get('output', '')[:50]}...'")
        return final_state