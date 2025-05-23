from __future__ import annotations  # https://peps.python.org/pep-0563/

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory, Message
import logging
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from src.agents.types import (
    SimpleMessageContentIOSchema, PreprocessingAgentOutputSchema, SearchAgentInputSchema, SearchAgentOutputSchema,
    QueryAgentInputSchema, QueryAgentOutputSchema, ContextManagerAgentInputSchema, ContextManagerAgentOutputSchema,
    PersonaAgentInputSchema, Persona, ResponseGeneratorAgentOutputSchema, FeedbackAgentOutputSchema
)
from src.tools import SearchTool, SearchToolConfig
from src.utils import PromptHandler

from src.agents.agent_factory import AgentFactory

# https://peps.python.org/pep-0563/  lame
if TYPE_CHECKING:
    import src.configurator as cfg


logger = logging.getLogger(__name__)


class AgentHandler(object):

    def __init__(self, config: cfg.Configurator):
        self.config = config
        self.prompt_dir = f"{config.base_dir}/prompts"

        logger.info("Initializing agents...")
        self.preprocessing_agent = self.__configure_agent(
            agent_name="preprocessing",
            input_schema=None,
            output_schema=PreprocessingAgentOutputSchema)
        self.query_agent = self.__configure_agent(
            agent_name="query",
            input_schema=QueryAgentInputSchema,
            output_schema=QueryAgentOutputSchema)
        self.search_agent = self.__configure_agent(
            agent_name="search",
            input_schema=SearchAgentInputSchema,
            output_schema=SearchAgentOutputSchema)
        # self.goal_tracking_agent = self.__configure_agent("goal_tracking", None, None)
        # self.reflective_journaling_agent = self.__configure_agent("reflective_journaling", None, None)
        self.persona_agent = self.__configure_agent(
            agent_name="persona",
            input_schema=PersonaAgentInputSchema,
            output_schema=Persona)
        self.context_manager_agent = self.__configure_agent(
            agent_name="context_manager",
            input_schema=ContextManagerAgentInputSchema,
            output_schema=ContextManagerAgentOutputSchema)
        self.coaching_agent = self.__configure_agent(
            agent_name="coaching",
            input_schema=ContextManagerAgentOutputSchema,
            output_schema=ResponseGeneratorAgentOutputSchema)
        self.feedback_agent = self.__configure_agent(
            agent_name="feedback",
            input_schema=ResponseGeneratorAgentOutputSchema,
            output_schema=FeedbackAgentOutputSchema)

        self.agent_map = {
            "preprocessing": self.preprocessing_agent,
            "query": self.query_agent,
            "search": self.search_agent,
            #"goal_tracking": self.goal_tracking_agent,
            #"reflective_journaling": self.reflective_journaling_agent,
            "persona": self.persona_agent,
            "context_manager": self.context_manager_agent,
            "coaching": self.coaching_agent,
            "feedback": self.feedback_agent
        }

        logger.info("Initializing tools...")
        self.search_tool = SearchTool(
                SearchToolConfig(base_url="google.com", max_results=self.config.data_config.search_results_per_query)
            )

        logger.info("Initializing chat memory...")
        self.full_memory = AgentMemory()

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
        self.context_manager_agent.memory = self.full_memory
        self.persona_agent.memory = self.full_memory
        self.coaching_agent.memory = self.full_memory  # Think about this one...
        self.feedback_agent.memory = self.full_memory  # Think about this one...

    def __configure_agent(
            self,
            agent_name: str,
            input_schema: Optional[Type[BaseIOSchema]],
            output_schema: Optional[Type[BaseIOSchema]]
    ) -> BaseAgent:
        prompt_loc = self.config.agent_config_map[agent_name]
        prompt = PromptHandler(prompt_loc).read()
        return BaseAgent(
            BaseAgentConfig(
                client=self.config.get_openrouter_client(),
                model=prompt["model"],
                system_prompt_generator=SystemPromptGenerator(**prompt["system_prompt"]),
                input_schema=input_schema,
                output_schema=output_schema,
                model_api_parameters=prompt.get("api_parameters", {})
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
    agent_handler = AgentHandler(Configurator())
    print("wait")