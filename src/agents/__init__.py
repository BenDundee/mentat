from __future__ import annotations  # https://peps.python.org/pep-0563/

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory, Message
import logging
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from src.tools import SearchTool, SearchToolConfig

from src.interfaces import (
    ConversationState, Persona, SimpleMessageContentIOSchema, Intent, AgentPrompt, PersonaContextProvider,
    QueryAgentInputSchema, QueryAgentOutputSchema, QueryAgentContextProvider
)

# https://peps.python.org/pep-0563/  lame
if TYPE_CHECKING:
    import src.configurator as cfg


logger = logging.getLogger(__name__)


class AgentHandler(object):

    def __init__(self, config: cfg.Configurator):
        self.config = config
        self.prompt_dir = config.base_dir / "prompts"

    def initialize_agents(self, prompt_manager: PromptManager):
        logger.info("Initializing agents...")
        self.intent_detection_agent = self.__configure_agent(
            prompt=prompt_manager.get_agent_prompt("intent-detection", intent_descriptions=Intent.llm_rep()),
            input_schema=ConversationState,
            output_schema=ConversationState
        )
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

        self.agent_map = {
            "intent_detection": self.intent_detection_agent,
            "query": self.query_agent,
            "persona": self.persona_agent,
        }

        logger.info("Initializing tools...")
        self.search_tool = SearchTool(
            SearchToolConfig(base_url="google.com", max_results=self.config.data_config.search_results_per_query)
        )

        logger.info("Initializing chat memory...")
        self.full_memory = AgentMemory()

        logger.info("Registering context providers...")
        persona_context = PersonaContextProvider(title="persona_context")
        self.persona_agent.register_context_provider("persona_context", persona_context)
        query_context = QueryAgentContextProvider(title="query_context")
        self.query_agent.register_context_provider("query_context", query_context)

    def update_memory(self, msgs: List[Dict]):
        # Get history and update. I think there's a better way to do this?
        self.full_memory.history = []  # Reset every time. Is there a better way to do it?
        for m in msgs:
            self.full_memory.initialize_turn()
            self.full_memory.history.append(
                Message(
                    role=m["role"],
                    content=SimpleMessageContentIOSchema(content=m["content"]),
                    turn_id=self.full_memory.current_turn_id
                )
            )

        # Update memory
        logger.info("Updating relevant agent memories...")
        # self.persona_agent.memory = self.full_memory

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

    def reconfigure_agent(self, agent_name: str):
        agent = self.agent_map[agent_name]
        memory = agent.memory
        contexts = agent.system_prompt_generator.context_providers

        # Reconfigure, reset
        self.__configure_agent(agent_name, agent.input_schema, agent.output_schema)
        agent.memory = memory
        agent.system_prompt_generator.context_providers = contexts


if __name__ == "__main__":
    from src.configurator import Configurator
    from src.managers import PromptManager

    agent_handler = AgentHandler(Configurator())
    agent_handler.initialize_agents(PromptManager())
    print("wait")