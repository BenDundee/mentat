import logging

from src.agents import AgentHandler
from src.configurator import Configurator
from src.interfaces import Persona
from src.managers.prompt_manager import PromptManager
from src.managers.persona_manager import PersonaManager
from src.managers.query_manager import QueryManager
from src.services import RAGService


logger = logging.getLogger(__name__)


class AgentFlows:

    def __init__(self, cfg: Configurator, rag_service: RAGService):
        self.config = cfg
        self.rag_service = rag_service
        self.agent_handler = AgentHandler(self.config)
        self.prompt_manager = PromptManager()
        self.persona_manager = PersonaManager(self.config)
        self.query_manager = QueryManager()

        logger.info("Setting initial states...")
        self.agent_handler.initialize_agents(self.prompt_manager)

    def update_persona(self) -> Persona:
        """Update user persona based on interactions."""
        logger.debug("Updating persona...")
        self.agent_handler.persona_agent.get_context_provider("persona_context").clear()

        logger.debug("Generating queries...")
        persona_query = self.query_manager.get_query("persona_update")
        self.agent_handler.query_agent.get_context_provider("query_context").query_prompt = persona_query
        queries = self.agent_handler.query_agent.run().queries
        query = ', '.join(queries)
        self.agent_handler.query_agent.get_context_provider("query_context").clear()

        logger.debug("Running RAG...")
        query_result = self.rag_service.query_all_collections_combined(query)
        self.agent_handler.persona_agent.get_context_provider("persona_context").query_result = query_result
        persona = self.agent_handler.persona_agent.run()
        self.rag_service.add_persona_data(persona)

        return persona