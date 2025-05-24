import json
import os
from typing import Dict, Any, Optional


class Configurator:
    """Manages configuration for the ExecutiveCoachAgent and its components."""

    DEFAULT_CONFIG_PATH = "config/default_config.json"

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        path = config_path or cls.DEFAULT_CONFIG_PATH

        if not os.path.exists(path):
            return cls.get_default_config()

        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return cls.get_default_config()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "llm": {
                "default": {
                    "model": "ChatOpenAI",
                    "model_name": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "creative": {
                    "model": "ChatOpenAI",
                    "model_name": "gpt-4",
                    "temperature": 0.9
                },
                "analytical": {
                    "model": "ChatOpenAI",
                    "model_name": "gpt-4",
                    "temperature": 0.2
                },
                "coding": {
                    "model": "ChatOpenAI",
                    "model_name": "gpt-4",
                    "temperature": 0.1
                }
            },
            "tools": {
                "goal_tracker": {
                    "llm_type": "analytical"
                },
                "journal_manager": {
                    "llm_type": "creative"
                },
                "workflow_orchestrator": {
                    "llm_type": "default"
                }
            },
            "agents": {
                "llm_type": "default",
                "max_iterations": 5,
                "verbose": True
            }
        }

    @classmethod
    def save_config(cls, config: Dict[str, Any], config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        path = config_path or cls.DEFAULT_CONFIG_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
