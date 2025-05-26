from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pydantic import Field, SecretStr

from api.interfaces import LLMParameters, LLMCredentials


class LLMProvider:
    def __init__(self, api_keys: LLMCredentials):
        self.api_keys = api_keys

    def llm(self, llm_parameters: Optional[LLMParameters] = None) -> BaseLanguageModel:

        if llm_parameters is None:
            return ChatOpenRouter(api_key=self.api_keys.openrouter_api_key, **self.get_default_llm_parameters().__dict__)

        client = None
        key_to_use = None
        if llm_parameters.model_provider == "openrouter":
            client = ChatOpenRouter
            key_to_use = self.api_keys.openrouter_api_key
        elif llm_parameters.model_provider == "openai_chat":
            client = ChatOpenAI
            key_to_use = self.api_keys.openai_api_key
        elif llm_parameters.model_provider == "openai":
            client = OpenAI
            key_to_use = self.api_keys.openai_api_key
        else:
            raise ValueError(f"Unsupported model type: {llm_parameters.model_provider}")
        return client(api_key=key_to_use, **llm_parameters.__dict__)

    @staticmethod
    def get_default_llm_parameters() -> LLMParameters:
        return LLMParameters(
            model_provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.7
        )


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(alias="api_key")

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )