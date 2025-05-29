from typing import Optional
from abc import ABC
from pydantic import BaseModel

class _Agent(ABC):
    def run(self, user_message: str) -> BaseModel:
        pass
