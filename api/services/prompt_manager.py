
import logging
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from api.interfaces import PromptContainer, LLMParameters, ModelKWArgs
from api.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages loading and access to prompt templates from YAML files.
    Supports different interfaces of prompts and caching for performance.
    """

    base_dir = Path(__file__).parent.parent.parent
    prompts_dir = base_dir / "prompts"

    def __init__(self):
        """Initialize the PromptManager with the directory containing prompt YAML files."""
        self.prompts = {}
        self.load_all_prompts()
        logger.debug(f"Loaded prompts: {self.prompts.keys()}")

    def load_all_prompts(self) -> None:
        """Load all prompt templates from the prompts directory and its subdirectories."""
        logger.info(f"Loading prompts from {self.prompts_dir}")
        _ = [self.load_prompt_from_file(f) for f in self.prompts_dir.glob("**/*.yaml")]
        logger.info(f"Loaded {len(self.prompts)} prompts")

    def load_prompt_from_file(self, file_path: Path) -> None:
        """Load prompt templates from a single YAML file."""
        try:
            with open(file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            for prompt_name, prompt_config in prompt_data.items():
                self.prompts[prompt_name] = _create_prompt_template(prompt_name, prompt_config)
                logger.debug(
                    f"Loaded prompt '{prompt_name}' from {file_path} with LLM parameters: {self.prompts[prompt_name].llm_parameters}"
                )
        except Exception as e:
            logger.warning(f"Error loading prompts, prompts not loaded from {file_path}: {e}")

    def get_prompt(self, prompt_name: str) -> Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]:
        """Get a prompt template by name."""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        return self.prompts[prompt_name].prompt_template

    def get_react_prompt(self, prompt_name: str) -> PromptTemplate:
        """Shim for fallback Reach agent behavior"""
        react_prompt_content = (
            self.prompts[prompt_name].prompt_template.template +
                "\n\nTools available: {toolbox}\n\nTool Names: {tool_names}\n\n{agent_scratchpad}"
        )
        react_prompt_input_variables = \
            list(set(self.prompts[prompt_name].prompt_template.input_variables + ["toolbox", "tool_names", "agent_scratchpad"]))
        return PromptTemplate(
            template=react_prompt_content,
            input_variables=react_prompt_input_variables
        )

    def get_llm_settings(self, prompt_name: str) -> LLMParameters:
        """Get LLM settings associated with a prompt."""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        return self.prompts[prompt_name].llm_parameters

    def reload_prompts(self) -> None:
        """Reload all prompts from disk."""
        self.prompts = {}
        self.load_all_prompts()


# --------------------------------------Helpers
def _create_prompt_template(prompt_name: str, prompt_config: Dict[str, Any]) -> PromptContainer:
    """Create the appropriate prompt template based on configuration."""
    template_format = prompt_config.get("template_format", "string")
    template = None
    if template_format == "string":
        template = PromptTemplate(
            template=prompt_config["template"],
            input_variables=prompt_config.get("input_variables", []),
        )

    elif template_format == "chat":
        messages = []
        for msg in prompt_config["messages"]:
            if msg["role"] == "system":
                messages.append(SystemMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "human":
                messages.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessagePromptTemplate.from_template(msg["content"]))

        template = ChatPromptTemplate(
            messages=messages,
            input_variables=prompt_config.get("input_variables", [])
        )

    elif template_format == "few_shot":
        example_prompt = PromptTemplate(
            template=prompt_config["example_template"],
            input_variables=prompt_config.get("example_variables", [])
        )

        template = FewShotPromptTemplate(
            examples=prompt_config["examples"],
            example_prompt=example_prompt,
            prefix=prompt_config.get("prefix", ""),
            suffix=prompt_config.get("suffix", ""),
            input_variables=prompt_config.get("input_variables", [])
        )

    else:
        raise ValueError(f"Unsupported template format: {template_format}")

    # Next get LLM parameters
    raw_llm_cfg = prompt_config.get("llm_parameters", {})
    llm_parameters = LLMProvider.get_default_llm_parameters()
    if raw_llm_cfg:
        if "model" not in raw_llm_cfg or "model_provider" not in raw_llm_cfg:
            logger.warning(
                "Missing model or model provider in LLM parameters, using default settings, using default LLM settings")
        else:
            llm_parameters = LLMParameters(**raw_llm_cfg)

    return PromptContainer(
        prompt_name=prompt_name,
        prompt_template=template,
        llm_parameters=llm_parameters
    )


if __name__ == "__main__":
    pm = PromptManager()
    print(pm.get_prompt("executive_coach_system"))
    print("wait")
