from abc import ABC, abstractmethod
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, SystemPromptGenerator
from typing import Optional, Type, Dict
from instructor import Instructor

from src.agents import PreprocessingAgentOutputSchema
from src.configurator import Configurator
from src.utils import PromptHandler



class AgentFactory:
    def __init__(self, config: Configurator):
        self._agent_registry = {}
        self.config = config

    def register(self, agent_name):
        def decorator(agent_cls):
            self._agent_registry[agent_name] = agent_cls
            return agent_cls
        return decorator

    def create(self, agent_name, *args, **kwargs):
        agent_cls = self._agent_registry.get(agent_name)
        client = kwargs.pop("client", self.config.get_openrouter_client())
        prompt_loc = self.config.agent_config_map[agent_name]
        prompt = PromptHandler(prompt_loc).read()
        if not agent_cls:
            raise ValueError(f"Agent '{agent_name}' not found")
        return agent_cls(client=client, prompt=prompt, *args, **kwargs)


class AgentWrapper(ABC):
    def __init__(self, client: Instructor, prompt: str):
        self.client = client
        self.prompt = prompt
        self.agent: Optional[BaseAgent] = None

    @abstractmethod
    def run(self, input: BaseIOSchema) -> BaseIOSchema:
        pass


@AgentFactory.register("preprocessing")
class PreprocessingAgent:
    def __init__(self, client: Instructor, prompt: Dict):
        super().__init__(client, prompt)
        self.agent = BaseAgent(
            BaseAgentConfig(
                client=client,
                model=prompt["model"],
                system_prompt_generator=SystemPromptGenerator(**prompt["system_prompt"]),
                input_schema=None,
                output_schema=PreprocessingAgentOutputSchema,
                model_api_parameters=prompt.get("api_parameters", {})
            )
        )

    def run(self, input: BaseIOSchema):
        return self.agent.run(input)