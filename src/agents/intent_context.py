from typing import List, Optional
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
from src.interfaces import TurnState


class IntentContextProvider(SystemPromptContextProviderBase):

    def __init__(self, title="intent_context"):
        super().__init__(title)
        self.turn_history: Optional[List[TurnState]] = None

    def clear(self):
        self.turn_history = None

    def get_info(self) -> str:
        if self.turn_history:
            as_str = "\n-----------------\n".join(t.model_dump_json(indent=4) for t in self.turn_history)
            return f"This is a short summary of the conversation so far:\n{as_str}"
        else:
            return ("There is no information about the previous messages. This may be because this is the first message "
                    "in a conversation.")