"""LLM provider registry and factory."""

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from mentat.core.config import AgentConfig
from mentat.core.settings import settings


@dataclass(frozen=True)
class LLMProviderInfo:
    """Metadata for a supported LLM provider."""

    name: str
    base_url: str
    env_key: str


PROVIDER_REGISTRY: dict[str, LLMProviderInfo] = {
    "openrouter": LLMProviderInfo(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
    ),
}


def build_llm(config: AgentConfig) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client from an AgentConfig.

    Args:
        config: Parsed agent configuration.

    Returns:
        A configured ChatOpenAI instance.

    Raises:
        KeyError: If the provider is not in PROVIDER_REGISTRY.
        EnvironmentError: If the required API key env var is not set.
    """
    provider = PROVIDER_REGISTRY.get(config.provider)
    if provider is None:
        raise KeyError(
            f"Unknown provider '{config.provider}'. "
            f"Available: {list(PROVIDER_REGISTRY)}"
        )

    api_key = settings.openrouter_api_key

    params: dict[str, Any] = {
        "model": config.model,
        "api_key": api_key,
        "base_url": provider.base_url,
        **config.llm_params,
    }

    return ChatOpenAI(**params)
