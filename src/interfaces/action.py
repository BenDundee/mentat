from typing import Dict, Optional
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema

from enum import Enum


class AgentAction(Enum):
    QUERY_DOCS = "query_docs"
    SEARCH_WEB = "search_web"
    GENERATE_RESPONSE = "generate_response"
    INITIATE_COACHING_SESSION = "initiate_coaching_session"

    @staticmethod
    def description():
        return {
            AgentAction.QUERY_DOCS.value:
                "Use when additional context from stored personal documents, feedback, past conversations, or journal "
                "entries would help generate a better response. This action requires generating a relevant semantic "
                "query.",
            AgentAction.SEARCH_WEB.value:
                "Use when the user's query requires recent, external, or general knowledge not present in internal "
                "documents.",
            AgentAction.INITIATE_COACHING_SESSION.value:
                "Use proactively when the user explicitly requests or clearly indicates the need for a structured "
                "coaching conversation. Examples include requests for career advice, discussing feedback, or seeking "
                "help to solve a professional challenge.",
            AgentAction.GENERATE_RESPONSE.value:
                "Use when the user's input can be answered directly and doesn't require additional context or "
                "retrieval. Note that th",
        }

    @staticmethod
    def action_parameters(num_web_searches: int = 5) -> Dict[str, list[str]]:
         return {
            AgentAction.QUERY_DOCS.value: [ # Will this work???
                "i.  Select one or more relevant queries from the set listed below",
                "ii. For each query, provide a brief description of the needed information"
            ],
            AgentAction.SEARCH_WEB.value: [
                f"i.  Construct {num_web_searches} web searches that are relevant to the task"
            ],
            AgentAction.INITIATE_COACHING_SESSION.value: [
                "i.  No additional parameters are required for this action. "
            ],
             AgentAction.GENERATE_RESPONSE.value: [
                 "i.  No additional parameters are required for this action."
             ]
        }

    # Problem: The orchestrator has to choose a set of actions, and each action has a different set of parameters that
    # are needed. For example, once the orchestrator chooses a query against the db, it must also come up with some text
    # that can serce as a filter. If the agent chooses a search against the web, it must also come up with a few
    # queries to execute. How can we convey the structure of the information, as each action implies a different set of
    # additional information required.
    @staticmethod
    def llm_rep() -> str:
        output = ["The following actions are available. For each action, note the required information:"]
        for _enum, desc in AgentAction.description().items():
            parameters = AgentAction.action_parameters()[_enum]
            output.append(f"  • `{_enum}`: {desc}")
            output.append(f"     ○ Additional output requirements:")
            output.extend(f"         {p}" for p in parameters)

        return "\n".join(output)


class ActionDirective(BaseIOSchema):
    """Additional directive for the agent that defines how it should take specific actions. """
    action: AgentAction = Field(..., description="The action to take")
    parameters: Optional[dict[str, str]] = Field(default_factory=dict, description="Additional parameters for the action")
    reasoning: Optional[str] = Field(..., description="Explanation of why this action was chosen")


if __name__ == "__main__":
    llm_rep = AgentAction.llm_rep()
    print(llm_rep)
