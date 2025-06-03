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
        self.prompts = {
            CoachingStage.INITIATION : config.prompt_manager.get_prompt("coaching_initiation"),
            CoachingStage.GOAL_SETTING : config.prompt_manager.get_prompt("coaching_goal_setting"),
            CoachingStage.EXPLORATION : config.prompt_manager.get_prompt("coaching_exploration"),
            CoachingStage.CONCLUSION : config.prompt_manager.get_prompt("coaching_conclusion"),
            CoachingStage.ASSIGNMENT : config.prompt_manager.get_prompt("coaching_assignment"),
        }

    def run(self, state: ConversationState) -> CoachingSessionState:
        if not isinstance(state, CoachingSessionState):
            coaching_state = CoachingSessionState(**state.model_dump())
            if not coaching_state.session_id:
                coaching_state.session_id = str(uuid.uuid4())
        else:
            coaching_state = state

        # Select the appropriate prompt based on the current stage
        current_prompt = self.prompts[coaching_state.stage].template
        prompt = ChatPromptTemplate.from_messages([
            ("system", current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        coaching_state.response = (
            prompt
            | self.llm_provider.get_chat_model(self.llm_params)
        ).invoke({
            "input": coaching_state.user_message,
            "chat_history": coaching_state.history,
            "goal": coaching_state.goal,
            "stage": coaching_state.stage.value,
            "session_id": coaching_state.session_id
        })
        coaching_state = self._update_stage(coaching_state)
        return coaching_state

    def _update_stage(self, state: CoachingSessionState) -> CoachingSessionState:
        """Determine if the coaching session should advance to the next stage."""
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
