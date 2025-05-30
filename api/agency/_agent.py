from typing import Optional, Dict, Any, TYPE_CHECKING
from abc import ABC


#if TYPE_CHECKING:
from api.interfaces import ConversationState


class _Agent(ABC):
    def run(self, state: Dict[str, Any]) -> ConversationState:
        pass
