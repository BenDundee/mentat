from typing import Optional, Dict, Any
from abc import ABC
from pydantic import BaseModel

class _Agent(ABC):
    def run(self, state: Dict[str, Any]) -> BaseModel:
        pass
