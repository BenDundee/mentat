from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class ModelAPIParameters:
    temperature: float
    top_k: int
    top_p: float
    max_tokens: int


@dataclass
class LLMParams:
    model_provider: str
    model: str
    model_api_parameters: ModelAPIParameters


@dataclass
class SystemPrompt:
    background: List[str]
    steps: List[str]
    output_instructions: List[str]


@dataclass
class AgentPrompt:
    name: Optional[str] = None
    description: Optional[str] = None
    llm_parameters: Optional[LLMParams] = None
    system_prompt: Optional[SystemPrompt] = None
    input_variables: Optional[List[str]] = None

    def __post_init__(self):
        '''Convert dictionary system_prompt to SystemPrompt object if needed.'''
        if isinstance(self.system_prompt, dict):
            self.system_prompt = SystemPrompt(**self.system_prompt)
        
        if isinstance(self.llm_parameters, dict):
            # Handle nested ModelAPIParameters conversion
            if isinstance(self.llm_parameters.get('model_api_parameters'), dict):
                self.llm_parameters['model_api_parameters'] = ModelAPIParameters(
                    **self.llm_parameters['model_api_parameters']
                )
            self.llm_parameters = LLMParams(**self.llm_parameters)

    def format(self, **kwargs: Dict[str, str]):
        if self.system_prompt is None:
            return
            
        old = self.system_prompt
        self.system_prompt = SystemPrompt(
            background=[d.format(**kwargs) for d in old.background],
            steps=[s.format(**kwargs) for s in old.steps],
            output_instructions=[o.format(**kwargs) for o in old.output_instructions]
        )