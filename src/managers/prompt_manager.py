import yaml as yml
import simplejson as sj
from pathlib import Path
import logging
from typing import Literal, Dict, Any

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent.parent


class PromptManager:
    """Holds all prompts"""
    def __init__(self):
        self.prompt_lib = BASE_DIR / "prompts"
        self.prompt_list = list(self.prompt_lib.rglob("**/*.prompt"))
        self._prompts: Dict[str, Any] = {}
        self.load_prompts()

    def load_prompts(self):
        for prompt_file in self.prompt_list:
            try:
                with open(prompt_file, "r") as f:
                    prompt_name = prompt_file.stem
                    self._prompts[prompt_name] = yml.safe_load(f)
            except Exception as e:
                logger.warning(f"Encountered exception while loading prompt in {prompt_file}: {e}")

    def get_prompt(self, prompt_name: str) -> str:
        prompt = self._prompts.get(prompt_name)
        if not prompt:
            logger.warning(f"Prompt {prompt_name} not found")
            return ""
        return prompt


if __name__ == "__main__":
    pm = PromptManager()
    print(pm.get_prompt("chat_history"))
    print("wait")