from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import OpenAI, HuggingFaceHub
from langchain_core.language_models import BaseLanguageModel
from typing import Dict, Any, Optional, List
import os


class LLMProvider:
    """
    TODO: Do I want to get rid of this and control this with the system prompts?

    Factory class for providing different LLM instances based on configuration.
    Supports different models for different tasks and caching of instances.
    """

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the LLM provider with configuration.

        Args:
            config: Dictionary mapping task interfaces to LLM configurations
                Example: {
                    "default": {"model": "ChatOpenAI", "model_name": "gpt-4", "temperature": 0.7},
                    "coding": {"model": "ChatOpenAI", "model_name": "gpt-4", "temperature": 0.2},
                    "creative": {"model": "ChatAnthropic", "model_name": "claude-2", "temperature": 0.9}
                }
        """
        # Default configuration if none provided
        self.config = config or {
            "default": {"model": "ChatOpenAI", "model_name": "gpt-3.5-turbo", "temperature": 0.7}
        }

        # Cache for LLM instances
        self._llm_cache: Dict[str, BaseLanguageModel] = {}

    def get_llm(self, task_type: str = "default") -> BaseLanguageModel:
        """
        Get an LLM instance for the specified task type.

        Args:
            task_type: Type of task (e.g., "default", "coding", "creative")

        Returns:
            LLM instance configured for the specified task
        """
        # Return cached instance if available
        if task_type in self._llm_cache:
            return self._llm_cache[task_type]

        # Get config for task type, fall back to default if not specified
        llm_config = self.config.get(task_type, self.config["default"])

        # Create new LLM instance based on config
        llm = self._create_llm_from_config(llm_config)

        # Cache the instance
        self._llm_cache[task_type] = llm

        return llm

    def _create_llm_from_config(self, config: Dict[str, Any]) -> BaseLanguageModel:
        """Create an LLM instance based on configuration."""
        model_type = config.pop("model")

        # Create instance based on model type
        if model_type == "ChatOpenAI":
            return ChatOpenAI(**config)
        elif model_type == "ChatAnthropic":
            return ChatAnthropic(**config)
        elif model_type == "OpenAI":
            return OpenAI(**config)
        elif model_type == "HuggingFaceHub":
            return HuggingFaceHub(**config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def reload_config(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Reload the configuration and clear the cache.
        Useful for runtime configuration changes.
        """
        self.config = config
        self._llm_cache.clear()