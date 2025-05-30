from typing import Dict, Any
from abc import ABC

from api.interfaces import ConversationState


class _Agent(ABC):
    def run(self, state: Dict[str, Any]) -> ConversationState:
        pass
