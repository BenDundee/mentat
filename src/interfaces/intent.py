from enum import Enum

from pydantic import Field
from typing import Dict
from atomic_agents.lib.base.base_io_schema import BaseIOSchema


class Intent(Enum):
    SIMPLE = "simple_message"
    COACHING_SESSION_REQUEST = "coaching_session_request"
    COACHING_SESSION_RESPONSE = "coaching_session_response"
    FEEDBACK = "feedback"

    @staticmethod
    def description():
        return {
            Intent.SIMPLE.value: "General conversation, questions, or statements.",
            Intent.COACHING_SESSION_REQUEST.value:
                "A request for a coaching session, it may be either explicitly requested or implied by the conversaion. "
                "For example, a request for feedback about a specific work situation could transition into an ad hoc "
                "coaching session. ***Be proactive about finding opportunities for a coaching session.***",
            Intent.COACHING_SESSION_RESPONSE.value:
                "A response during a coaching session. Note that there must be an active coaching session for this "
                "intent to be valid.",
            Intent.FEEDBACK.value:
                "A request for feedback or help from the user. This may include things like a request for thoughts "
                "about whether a particular job posting aligns with long term goals, help drafting a resume/cover "
                "letter, or advice about a specific work situation.",
        }

    @staticmethod
    def get_intent(intent: str) -> "Intent":
        for i in Intent:
            if i.value == intent:
                return i
        raise Exception(f"Invalid intent: {intent}")

    @staticmethod
    def llm_rep():
        return "\n".join(
            f"  • `{intent}`: {desc}" for intent, desc in Intent.description().items()
        )
