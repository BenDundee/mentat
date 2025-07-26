import logging
from typing import Tuple

from src.agents import AgentHandler
from src.configurator import Configurator
from src.interfaces import OrchestrationAgentInputSchema
from src.managers.prompt_manager import PromptManager
from src.managers.persona_manager import PersonaManager
from src.managers.query_manager import QueryManager
from src.services import RAGService, ConversationService

logger = logging.getLogger(__name__)


class AgentFlows:

    def __init__(self, cfg: Configurator, rag_service: RAGService, conversation_svc: ConversationService = None):
        self.config = cfg
        self.rag_service = rag_service
        self.conversation_svc = conversation_svc
        self.agent_handler = AgentHandler(self.config)
        self.prompt_manager = PromptManager()
        self.persona_manager = PersonaManager(self.config)
        self.query_manager = QueryManager()

        logger.info("Setting initial states...")
        prompt_configs = {
            "query_desctiptions": self.query_manager.generate_query_summary(),
        }
        self.agent_handler.initialize_agents(self.prompt_manager, prompt_configs)

    def update_persona(self):
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
        self.conversation_svc.state.persona = persona

    def orchestrate(self):
        instructions = self.agent_handler.orchestration_agent.run(
            OrchestrationAgentInputSchema(user_input=self.conversation_svc.current_turn.user_message)
        )
        self.conversation_svc.orchestrate_turn(instructions)

    def generate_simple_response(self):
        # response = self.agent_handler.coach_agent.run(conversation_state)
        pass


if __name__ == "__main__":
    from src.utils import get_message

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    config = Configurator()
    rag_service = RAGService(config, initialize=False)
    convo_svc = ConversationService(rag_service)
    am = AgentFlows(config, rag_service, convo_svc)

    convo_svc.initiate_turn(user_msg="Hello")
    am.orchestrate()

    print("wait")