from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class IntentContextProvider(SystemPromptContextProviderBase):

    def __init__(self, title="intent_context"):
        super().__init__(title)
        self.previous_intent = None

    def clear(self):
        self.previous_intent = None

    def get_info(self) -> str:
        if self.previous_intent:
            return f"The last identified intent was: {self.previous_intent}"
        else:
            return ("There is no information about the intent of the last message. This may be because this is the "
                    "first message in a conversation.")