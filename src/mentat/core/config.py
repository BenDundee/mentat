"""Agent configuration loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent."""

    provider: str
    model: str
    system_prompt: str
    llm_params: dict[str, Any] = field(default_factory=dict)
    prompt_defaults: dict[str, Any] = field(default_factory=dict)
    extra_config: dict[str, Any] = field(default_factory=dict)


_CONFIG_DIR = Path("configs")


def load_agent_config(agent_name: str) -> AgentConfig:
    """Load agent configuration from configs/<agent_name>.yml.

    Args:
        agent_name: The agent identifier (e.g. "orchestration").

    Returns:
        Parsed AgentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required keys are missing from the config.
    """
    config_path = _CONFIG_DIR / f"{agent_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Agent config not found: {config_path}. "
            f"Create configs/{agent_name}.yml first."
        )

    with config_path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    return AgentConfig(
        provider=data["provider"],
        model=data["model"],
        system_prompt=data["system_prompt"],
        llm_params=data.get("llm_params", {}),
        prompt_defaults=data.get("prompt_defaults", {}),
        extra_config=data.get("extra_config", {}),
    )
