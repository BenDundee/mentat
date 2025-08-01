import yaml as yml
from pathlib import Path
import logging
from typing import Dict, Optional

from src.interfaces import AgentPrompt


logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent.parent


class PromptManager:
    """Holds all prompts"""
    def __init__(self):
        self.prompt_lib = BASE_DIR / "prompts"
        self.agent_prompts = list(self.prompt_lib.glob("agents/*.yaml"))
        self.query_prompts = list(self.prompt_lib.glob("query/*.yaml"))

        self._agent_prompts: Dict[str, AgentPrompt] = {}
        self._load_prompts()

    def _load_prompts(self):
        self._load_agent_prompts()
        self._load_query_prompts()

    def _load_agent_prompts(self):
        for prompt_file in self.agent_prompts:
            try:
                prompt_name = prompt_file.stem
                with open(prompt_file, "r") as f:
                    agent = AgentPrompt(**yml.safe_load(f))
                    self._agent_prompts[prompt_name] = agent
            except Exception as e:
                logger.warning(f"Encountered exception while loading prompt in {prompt_file}: {e}")

    def _load_query_prompts(self):
        # TODO: Remove query manager
        pass

    def get_agent_prompt(self, prompt_name: str, **input_vars: Optional[Dict[str, str]]) -> AgentPrompt:

        prompt = self._agent_prompts.get(prompt_name)
        if not prompt:
            logger.warning(f"Prompt {prompt_name} not found")
            return AgentPrompt()

        # check that all input variables are present
        if prompt.input_variables:
            for var in prompt.input_variables:
                if var not in input_vars:
                    logger.warning(f"Prompt {prompt_name} is missing input variable {var}")

        prompt.format(**input_vars)
        return prompt

    def check_system_prompt(self, prompt_name: str, **input_vars: Optional[Dict[str, str]]) -> str:
        from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
        ap = self.get_agent_prompt(prompt_name, **input_vars)
        sp = SystemPromptGenerator(
            background=ap.system_prompt.background,
            steps=ap.system_prompt.steps,
            output_instructions=ap.system_prompt.output_instructions
        )
        return sp.generate_prompt()



if __name__ == "__main__":
    from src.interfaces import Intent, AgentPrompt, AgentAction
    from src.managers.query_manager import QueryManager

    pm = PromptManager()
    qm = QueryManager()
    prompt = pm.check_system_prompt(
        prompt_name="orchestration",
        intent_descriptions=Intent.llm_rep(),
        actions_and_parameter_requirements=AgentAction.llm_rep(),
        query_descriptions=qm.generate_query_summary()
    )
    print(prompt)
    print("")