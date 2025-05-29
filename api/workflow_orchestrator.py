import logging
from typing import Dict, Any, Callable

from api.api_configurator import APIConfigurator
from api.interfaces import Intent
from api.agency.intent_detector import IntentDetector
from api.agency.simple_responder import SimpleResponder
from api.agency._agent import _Agent

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    def __init__(self, config: APIConfigurator):
        self.config = config
        self.intent_detector = IntentDetector(config)
        self._intent_to_workflow_map: Dict[Intent, _Agent] = {}

        self._register_known_workflows()

    def _register_known_workflows(self):
        """
        Registers the workflows the orchestrator knows about.
        This is where you'll add more workflows in the future.
        """
        self._intent_to_workflow_map[Intent.SIMPLE] = SimpleResponder(self.config)
        # You can add more workflows as they're implemented
        # self.add_workflow("goal_setting", handle_goal_setting)
        # self.add_workflow("feedback", handle_feedback)
        # self.add_workflow("action_planning", handle_action_planning)
        # self.add_workflow("reflection", handle_reflection)

    def process_message(self, user_message: str, user_id: str = "guest") -> Dict[str, Any]:
        """
        Main entry point to handle a user's message.
        """
        logger.info(f"Orchestrator processing message from {user_id}: '{user_message[:50]}...'")

        # 1. Determine Intent using the intent detector
        intent = self.intent_detector.run(user_message)

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
                final_state = workflow_handler(current_state, self.config)
            except Exception as e:
                logger.error(f"Error during execution of workflow for intent '{intent}': {e}", exc_info=True)
                final_state = {**current_state, "output": f"An error occurred while handling your '{intent}' request."}
        else:
            logger.warning(f"No workflow handler found for intent: '{intent}'")
            final_state = {**current_state, "output": f"I don't know how to handle a '{intent}' request yet."}

        logger.info(f"Orchestrator finished. Final output: '{final_state.get('output', '')[:50]}...'")
        return final_state