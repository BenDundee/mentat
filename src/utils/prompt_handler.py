import yaml as yml
import simplejson as sj
from pathlib import Path
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class PromptHandler:
    """Class to load prompts from a YAML file. Putting it here to make it accessible to the workbench"""
    def __init__(self, prompt_file):
        self.__base_dir = Path(__file__).parent.parent.parent
        self.prompt_file = self.__base_dir / "prompts" / prompt_file
        self.as_dict = None

    def read(self):
        if self.as_dict is not None: return self.as_dict
        with open(self.prompt_file, "r") as f:
            self.as_dict = yml.safe_load(f)
        return self.as_dict

    def write(self, prompt_dict, fmt: Literal["yml", "json"]="yml"):
        if fmt == "json":
            with open(self.prompt_file, "w") as f:
                sj.dump(prompt_dict, f, indent=4)
        elif fmt == "yml":
            with open(self.prompt_file, "w") as f:
                yml.dump(prompt_dict, f)
        else:
            raise ValueError("Format must be either 'yml' or 'json'")