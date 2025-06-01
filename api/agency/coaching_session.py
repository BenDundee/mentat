import logging
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import uuid

from api.api_configurator import APIConfigurator
from api.agency._agent import _Agent
from api.interfaces import CoachingSessionState, ConversationState, CoachingStage

logger = logging.getLogger(__name__)


class CoachingSession(_Agent):
    """Agent for handling structured coaching sessions."""

    def __init__(self, config: APIConfigurator):
        self.llm_provider = config.llm_manager
        self.llm_params = config.prompt_manager.get_llm_settings("coaching_session")
        self.prompt_templates = {
            CoachingStage.INITIATION: config.prompt_manager.get_prompt("coaching_initiation").template,
            CoachingStage.GOAL_SETTING: config.prompt_manager.get_prompt("coaching_goal_setting").template,
            CoachingStage.EXPLORATION: config.prompt_manager.get_prompt("coaching_exploration").template,
            CoachingStage.CONCLUSION: config.prompt_manager.get_prompt("coaching_conclusion").template,
            CoachingStage.ASSIGNMENT: config.prompt_manager.get_prompt("coaching_assignment").template,
        }

    def run(self, state: ConversationState) -> CoachingSessionState:
        # Convert to CoachingSessionState if not already
        if not isinstance(state, CoachingSessionState):
            coaching_state = CoachingSessionState(**state.model_dump())
            # Generate a new session ID if needed
            if not coaching_state.session_id:
                coaching_state.session_id = str(uuid.uuid4())
        else:
            coaching_state = state

        # Select the appropriate prompt based on the current stage
        current_prompt = self.prompt_templates[coaching_state.stage]

        # Prepare the prompt with stage-specific context
        prompt = ChatPromptTemplate.from_messages([
            ("system", current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        # Create chain with the appropriate prompt
        chain = prompt | self.llm_provider.get_chat_model(self.llm_params)

        # Run the chain
        response = chain.invoke({
            "input": coaching_state.user_message,
            "chat_history": coaching_state.history,
            "goal": coaching_state.goal,
            "stage": coaching_state.stage.value,
            "session_id": coaching_state.session_id
        })

        # Update the state with the response
        coaching_state.response = response

        # Update the stage based on the conversation flow
        # This could be done with a separate "stage detector" agent or
        # through structured output from the LLM
        coaching_state = self._update_stage(coaching_state)

        return coaching_state

    def _update_stage(self, state: CoachingSessionState) -> CoachingSessionState:
        """Determine if the coaching session should advance to the next stage."""
        # This could use a classifier or structured output from the LLM
        # For now, a simple placeholder implementation
        if state.stage == CoachingStage.INITIATION and state.goal:
            state.stage = CoachingStage.GOAL_SETTING
        elif state.stage == CoachingStage.GOAL_SETTING and len(state.history) >= 4:
            state.stage = CoachingStage.EXPLORATION
        elif state.stage == CoachingStage.EXPLORATION and len(state.history) >= 10:
            state.stage = CoachingStage.CONCLUSION
        elif state.stage == CoachingStage.CONCLUSION and state.assignments:
            state.stage = CoachingStage.ASSIGNMENT
            state.is_active = False  # End the session when assignments are given

        return state