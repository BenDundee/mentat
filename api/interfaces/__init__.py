from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain.schema import ChatMessage


class ChatRequest(BaseModel):
    """Input schema for the chat endpoint."""
    message: str = Field(..., description="The message to send to the chatbot")
    history: Optional[List[ChatMessage]] = None
    user_id: Optional[str] = "default_user"

@dataclass
class LLMCredentials:
    """Credentials for accessing LLMs."""
    openai_api_key: str
    openrouter_api_key: str

@dataclass
class ModelKWArgs:
    """Parameters for accessing LLMs."""
    top_p: Optional[float] = None

@dataclass
class LLMParameters:
    model_provider: str
    model: str

    #---- API params
    temperature: float = 0.0
    max_tokens: float = 2000
    model_kwargs: ModelKWArgs = None

    def get_client_kwargs(self):
        model_kwargs = self.model_kwargs or ModelKWArgs()
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_kwargs": {
                "top_p": model_kwargs.top_p
            }
        }

