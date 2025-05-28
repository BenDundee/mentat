from .intent_detector import IntentDetector


from typing import Optional
from abc import ABC
class _Agent(ABC):
    def run(self, user_message: str) -> Optional[str]:
        pass
