"""Tests for core/config.py."""

import pytest

from mentat.core.config import AgentConfig, load_agent_config


def test_load_orchestration_config():
    """The orchestration config should load without errors."""
    config = load_agent_config("orchestration")
    assert isinstance(config, AgentConfig)
    assert config.provider == "openrouter"
    assert config.model != ""
    assert config.system_prompt != ""


def test_load_config_missing_file():
    """Loading a non-existent config should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_agent_config("does_not_exist")


def test_agent_config_is_frozen():
    """AgentConfig must be immutable (frozen dataclass)."""
    config = load_agent_config("orchestration")
    with pytest.raises((AttributeError, TypeError)):
        config.model = "something-else"  # type: ignore[misc]
