from __future__ import annotations  # https://peps.python.org/pep-0563/

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory, Message
import logging
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from src.tools import SearchTool, SearchToolConfig

from src.interfaces import (
    ConversationState, Persona, SimpleMessageContentIOSchema, CoachingStage,
    OrchestrationAgentOutputSchema, OrchestrationAgentInputSchema,
    AgentPrompt, QueryAgentInputSchema, QueryAgentOutputSchema, CoachResponse,
    Intent, AgentAction, CoachingAgentInputSchema
)

from src.agents.intent_context import IntentContextProvider
from src.agents.persona_context import PersonaContextProvider
from src.agents.query_context import QueryContextProvider

# https://peps.python.org/pep-0563/  lame
if TYPE_CHECKING:
    import src.configurator as cfg


logger = logging.getLogger(__name__)


class AgentHandler(object):

    def __init__(self, config: cfg.Configurator):
        self.config = config
        self.prompt_dir = config.base_dir / "prompts"

    def initialize_agents(self, prompt_manager: PromptManager, prompt_configs: Optional[Dict[str, str]] = None):
        logger.info("Initializing agents...")
        self.query_agent = self.__configure_agent(
            prompt=prompt_manager.get_agent_prompt("query"),
            input_schema=QueryAgentInputSchema,
            output_schema=QueryAgentOutputSchema
        )
        self.persona_agent = self.__configure_agent(
            prompt=prompt_manager.get_agent_prompt("persona"),
            input_schema=None,
            output_schema=Persona
        )
        self.orchestration_agent = self.__configure_agent(
            prompt=prompt_manager.get_agent_prompt(
                "orchestration",
                intent_descriptions=Intent.llm_rep(),
                actions_and_parameter_requirements=AgentAction.llm_rep(),
                query_descriptions=prompt_configs.get("query_descriptions", "")),
            input_schema=OrchestrationAgentInputSchema,
            output_schema=OrchestrationAgentOutputSchema
        )
        self.coach_agent = self.__configure_agent(
            prompt=prompt_manager.get_agent_prompt(
                "coach",
                coaching_framework=CoachingStage.llm_rep()
            ),
            input_schema=CoachingAgentInputSchema,
            output_schema=CoachResponse
        )

        self.agent_map = {
            "query": self.query_agent,
            "persona": self.persona_agent,
            "orchestration": self.orchestration_agent,
            "coach": self.coach_agent,
            # "critic": self.critic_agent, -- this agent ensures the coach constructs a good response
            # "summarization": self.summarization_agent -- this agent updates the CoachingSession that lives in ConversationState
        }

        logger.info("Initializing tools...")
        self.search_tool = SearchTool(
            SearchToolConfig(base_url="google.com", max_results=self.config.data_config.search_results_per_query)
        )

        logger.info("Registering context providers...")
        persona_context = PersonaContextProvider(title="persona_context")
        self.persona_agent.register_context_provider("persona_context", persona_context)

        query_context = QueryContextProvider(title="query_context")
        self.query_agent.register_context_provider("query_context", query_context)

    def __configure_agent(
            self,
            prompt: AgentPrompt,
            input_schema: Optional[Type[BaseIOSchema]],
            output_schema: Optional[Type[BaseIOSchema]]
    ) -> BaseAgent:
        # TODO: Get `client` from AgentPrompt.llm_params
        return BaseAgent(
            BaseAgentConfig(
                client=self.config.get_openrouter_client(),
                model=prompt.llm_parameters.model,
                model_api_parameters=prompt.llm_parameters.model_api_parameters,
                system_prompt_generator=SystemPromptGenerator(**prompt.system_prompt.__dict__),
                input_schema=input_schema,
                output_schema=output_schema
            )
        )


if __name__ == "__main__":
    from src.configurator import Configurator
    from src.managers import PromptManager

    agent_handler = AgentHandler(Configurator())
    agent_handler.initialize_agents(PromptManager())
    print("wait")