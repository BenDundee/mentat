import os
import yaml
from typing import Dict, Any, List, Optional, Union
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

from pathlib import Path


class PromptManager:
    """
    Manages loading and access to prompt templates from YAML files.
    Supports different types of prompts and caching for performance.
    """

    base_dir = Path(__file__).parent.parent.parent
    prompts_dir = base_dir / "prompts"

    def __init__(self):
        """Initialize the PromptManager with the directory containing prompt YAML files."""
        self.prompts = {}
        self.load_all_prompts()

    def load_all_prompts(self) -> None:
        """Load all prompt templates from the prompts directory and its subdirectories."""
        _ = [self.load_prompt_from_file(f) for f in self.prompts_dir.glob("**/*.yaml")]

    def load_prompt_from_file(self, file_path: Path) -> None:
        """Load prompt templates from a single YAML file."""
        try:
            with open(file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            for prompt_name, prompt_config in prompt_data.items():
                self.prompts[prompt_name] = self._create_prompt_template(prompt_config)
        except Exception as e:
            print(f"Error loading prompts from {file_path}: {e}")

    def _create_prompt_template(self, prompt_config: Dict[str, Any]) -> Union[
        PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]:
        """Create the appropriate prompt template based on configuration."""
        template_format = prompt_config.get("template_format", "string")

        if template_format == "string":
            return PromptTemplate(
                template=prompt_config["template"],
                input_variables=prompt_config.get("input_variables", [])
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

            return ChatPromptTemplate(
                messages=messages,
                input_variables=prompt_config.get("input_variables", [])
            )

        elif template_format == "few_shot":
            example_prompt = PromptTemplate(
                template=prompt_config["example_template"],
                input_variables=prompt_config.get("example_variables", [])
            )

            return FewShotPromptTemplate(
                examples=prompt_config["examples"],
                example_prompt=example_prompt,
                prefix=prompt_config.get("prefix", ""),
                suffix=prompt_config.get("suffix", ""),
                input_variables=prompt_config.get("input_variables", [])
            )

        else:
            raise ValueError(f"Unsupported template format: {template_format}")

    def get_prompt(self, prompt_name: str) -> Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]:
        """Get a prompt template by name."""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        return self.prompts[prompt_name]

    def get_llm_settings(self, prompt_name: str) -> Dict[str, Any]:
        """Get LLM settings associated with a prompt."""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        # Parse the YAML file again to get the llm_settings
        # This is a bit inefficient but ensures we get the latest settings
        for root, _, files in os.walk(self.prompts_dir):
            for file in files:
                if file.endswith(".yaml"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        prompt_data = yaml.safe_load(f)
                        if prompt_name in prompt_data:
                            return prompt_data[prompt_name].get("llm_settings", {})

        return {}

    def reload_prompts(self) -> None:
        """Reload all prompts from disk."""
        self.prompts = {}
        self.load_all_prompts()