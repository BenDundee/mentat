from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
from typing import Optional



class PersonaContextProvider(SystemPromptContextProviderBase):

    def __init__(self, title="persona_context"):
        super().__init__(title)
        self.query_result: Optional[str] = None

    def clear(self):
        self.query_result = None

    def get_info(self) -> str:
        return self.query_result