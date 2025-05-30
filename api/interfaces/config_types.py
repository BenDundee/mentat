from typing import Optional, Union
from dataclasses import dataclass
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate


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

    def to_dict(self):
        if self.model_kwargs is None:
            model_kwargs = ModelKWArgs()
        elif isinstance(self.model_kwargs, dict):
            model_kwargs = ModelKWArgs(**self.model_kwargs)
        else:
            model_kwargs = self.model_kwargs
        
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_kwargs": {
                "top_p": model_kwargs.top_p
            }
        }

    @staticmethod
    def from_dict(d: dict) -> "LLMParameters":
        return LLMParameters(
            model_provider=d["model_provider"],
            model=d["model"],
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
            model_kwargs=ModelKWArgs(**d.get("model_kwargs"))
        )

@dataclass
class PromptContainer:
    prompt_name: str
    prompt_template: Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]
    llm_parameters: LLMParameters
