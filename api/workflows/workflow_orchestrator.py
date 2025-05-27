import logging
from typing import Dict, Any, Callable, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from api.services import LLMProvider
from api.api_configurator import APIConfigurator

logger = logging.getLogger(__name__)

# --- Workflow Abstraction (Minimal: A Callable) ---
# A workflow is a function that takes the current state (including user message)
# and dependencies (like an LLM provider), and returns an updated state with an output.
WorkflowHandler = Callable[[Dict[str, Any], LLMProvider], Dict[str, Any]]


class WorkflowOrchestrator:
    def __init__(self, config: APIConfigurator = APIConfigurator()):
        self.config = config
        self.llm_provider = self.config.llm_provider
        self._intent_to_workflow_map: Dict[str, WorkflowHandler] = {}
        self._register_known_workflows()

    def _register_known_workflows(self):
        """
        Registers the workflows the orchestrator knows about.
        This is where you'll add more workflows in the future.
        """
        self.add_workflow("simple_response", handle_simple_response)
        # Example for future:
        # self.add_workflow("goal_setting", handle_goal_setting_workflow_function)

    def add_workflow(self, intent_name: str, handler: WorkflowHandler):
        """
        Allows adding or updating workflow handlers. Key for extensibility.
        """
        self._intent_to_workflow_map[intent_name] = handler
        logger.info(f"Workflow for intent '{intent_name}' registered.")

    def process_message(self, user_message: str, user_id: str = "guest") -> Dict[str, Any]:
        """
        Main entry point to handle a user's message.
        """
        logger.info(f"Orchestrator processing message from {user_id}: '{user_message[:50]}...'")

        # 1. Determine Intent
        intent = detect_user_intent(user_message)

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

# --- MVP Workflow Implementation: Simple Response ---
def handle_simple_response(state: Dict[str, Any], config: APIConfigurator) -> Dict[str, Any]:
    """
    Enhanced workflow: Uses a proper LangChain prompt template to structure the message
    before sending it to the LLM.
    """
    user_message = state.get("user_message", "No message provided")
    logger.info(f"Executing 'handle_simple_response' for message: '{user_message[:50]}...'")
    
    # Create a chat prompt template with a system message and a human message
    prompt_template = config.prompt_manager.get_prompt("simple_response").template

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an executive coach named Mentat. Your role is to provide guidance, ask thoughtful questions, "
            "and help users develop their professional and personal skills. "
            "Be supportive, insightful, and focused on growth."
        ),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Format the prompt with the user's message
    prompt = prompt_template.format_messages(input=user_message)
    
    # Get the LLM
    llm = config.llm_provider.llm()
    
    try:
        # Send the formatted prompt to the LLM
        response_data = llm.invoke(prompt)
        output_content = response_data.content
    except Exception as e:
        logger.error(f"Error during LLM call in handle_simple_response: {e}")
        output_content = "There was an issue generating a response."

    return {**state, "output": output_content}

# --- Intent Detection (Simplified for MVP) ---
def detect_user_intent(user_message: str) -> Optional[str]:
    """
    Determines the user's intent. For MVP, this is very basic.
    It will always return "simple_response" to route to our MVP workflow.
    """
    # In the future, this could involve:
    # - Keyword matching
    # - Regex patterns
    # - A machine learning classifier
    # - An LLM call to classify the intent
    logger.info(f"Detecting intent for: '{user_message[:50]}...'. MVP always chooses 'simple_response'.")
    return "simple_response"  # For MVP, all roads lead to simple_response