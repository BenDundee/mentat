import logging
from typing import Dict, List

from src.agents import AgentHandler
from src.agents.types import (
    PreprocessingAgentInputSchema, SearchAgentInputSchema, SearchToolInputSchema, QueryAgentInputSchema, 
    ContextManagerAgentInputSchema, PersonaAgentInputSchema, Goal, ResponseDraftWithFeedback
)
from src.configurator import Configurator
from src.services.chroma_db import ChromaDBService
from src.utils import PromptHandler

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: Configurator):
        self.config = config

        # Initialize class -- need to abstract these?
        logger.info("Initializing Chroma DB...")
        self.chroma_db = ChromaDBService(config=self.config,recreate_collection=True)
        self.chroma_db.initialize_db()

        logger.info("Initializing agents...")
        self.agent_handler = AgentHandler(self.config)

    def get_response(self, messages: List[Dict]) -> str:
        logger.info(f"Received input, updating memory...")
        self.agent_handler.update_memory(messages)

        # First: if persona is empty it needs to be generated
        if self.config.persona.is_empty():
            logger.info("Updating persona...")
            prompt = PromptHandler("persona-update.prompt").read()["prompt"]
            persona_query = self.agent_handler.query_agent.run(QueryAgentInputSchema(user_input=prompt))
            persona_query_result = \
                self.chroma_db.query(persona_query.query, where={"owner": self.config.user_config.user_name})
            self.config.update_persona(
                self.agent_handler.persona_agent.run(PersonaAgentInputSchema(query_result=persona_query_result)))
            logger.info(f"Updated persona: {self.config.persona.get_summary()}")

        # On to pre-processing!
        logger.info("Pre-processing...")
        last_message = messages[-1]["content"]
        preprocessing_agent_output = \
            self.agent_handler.preprocessing_agent.run(PreprocessingAgentInputSchema(user_input=last_message))

        # Enrich context, given input
        if preprocessing_agent_output.detected_intent in ("coaching_request", "document_upload"):  # HACK for now
            logging.info("Detected coaching request...")

            # Query agent
            logger.info("Querying document store...")
            query_agent_output = self.agent_handler.query_agent.run(QueryAgentInputSchema(user_input=last_message))
            query_result = self.chroma_db.query(
                query_agent_output.query, n_results=self.config.data_config.db_results_per_query)

            # Search agent
            logger.info("Searching internet...")
            search_agent_output = (
                self.agent_handler.search_agent.run(
                    SearchAgentInputSchema(user_input=last_message,
                                           num_queries=self.config.data_config.search_engine_num_queries))
            )
            search_results = \
                self.agent_handler.search_tool.run(SearchToolInputSchema(queries=search_agent_output.queries))
            
            # TODO: Abstract this out to a function?
            # add search results to chroma db, get relevant results, then delete search results
            logger.info("Getting relevant search results...")
            search_results_ids = self.chroma_db.add_search_results(search_results.results)
            relevant_search_results = self.chroma_db.query(  # not sure what to do here -- need to figure out
                query_agent_output.query, 
                n_results=10,
                ids=search_results_ids
            )
            self.chroma_db.delete_by_ids(search_results_ids)

            # Goal agent -- ignore for now, in future consider personal goals in response

            # Journal agent -- recommend specific topics to reflect on, in depth

            # Context Manager -- collect all information from other agents, recommend specific actions, next steps
            logger.info("Collecting all information from other agents...")
            context = ContextManagerAgentInputSchema(
                search_results=relevant_search_results,
                query_result=query_result,
                user_input=last_message
            )
            context_manager_output = self.agent_handler.context_manager_agent.run(context)
            max_retries = 5  # TODO: Make this a config
            for i in range(max_retries):
                # Response Generation
                logger.info("Generating response...")
                response_generator_output = self.agent_handler.coaching_agent.run(context_manager_output)

                # Feedback -- ensure that the response is appropriate to the user's needs and goals, etc.
                logger.info("Getting feedback on response...")
                feedback_output = self.agent_handler.feedback_agent.run(response_generator_output)

                if feedback_output.rewrite_response:
                    logger.info("Feedback agent determined that the response needs to be rewritten.")
                    context_manager_output.add_previous_response(ResponseDraftWithFeedback(
                        response_draft=response_generator_output.response,
                        feedback=feedback_output.feedback,
                        reasoning=feedback_output.reasoning
                    ))
                else:
                    break
            
            # Either max retries or we break
            logger.info("Feedback agent determined that the response is acceptable.")
            response = {"role": "assistant", "content": response_generator_output.response}
            # self.agent_handler.update_memory(response)  # TODO: Figure this out
            return response["content"]

        elif preprocessing_agent_output.detected_intent == "journal_entry":
            logging.info("Detected journal entry")
            return "Hello World!"

        elif preprocessing_agent_output.detected_intent == "document_upload":
            logging.info("Detected document upload")
            return "Hello World!"

        else:
            raise Exception("Unknown intent from preprocessing agent")


if __name__ == "__main__":

    config = Configurator()
    controller = Controller(config)
    controller.get_response({"role": "user", "content": "This is a test"})
    logger.info("Wait!")
