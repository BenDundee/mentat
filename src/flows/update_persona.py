from src.agents import AgentHandler
from src.configurator import Configurator
from src.services import ChromaDBService
from src.utils import PromptHandler
from src.agents.types import QueryAgentInputSchema, PersonaAgentInputSchema


def update_persona(agent_handler: AgentHandler, chroma_db: ChromaDBService, config: Configurator):

    prompt = PromptHandler("persona-update.prompt").read()["prompt"]
    persona_query = agent_handler.query_agent.run(QueryAgentInputSchema(user_input=prompt))
    persona_query_result = \
        chroma_db.query(persona_query.query, where={"owner": self.config.user_config.user_name})
    config.update_persona(
        agent_handler.persona_agent.run(PersonaAgentInputSchema(query_result=persona_query_result)))
