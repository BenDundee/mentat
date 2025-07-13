from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class QueryContextProvider(SystemPromptContextProviderBase):
    """Context provider for the query agent."""

    def __init__(self, title="query_context"):
        super().__init__(title)
        self.query_prompt = ""

    def clear(self):
        self.query_prompt = ""

    def get_info(self) -> str:
        return f"Use the following directive to construct the queries:\n\n{self.query_prompt}"