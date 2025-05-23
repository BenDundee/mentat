import logging
from typing import Dict, List

from src.agents import AgentHandler, AgentFactory
from src.agents.types import (
    PreprocessingAgentInputSchema, SearchAgentInputSchema, SearchToolInputSchema, QueryAgentInputSchema, 
    ContextManagerAgentInputSchema, PersonaAgentInputSchema, Goal, ResponseDraftWithFeedback
)
from src.configurator import Configurator
from src.services.chroma_db import ChromaDBService
from src.flows import update_persona, do_research

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
        self.agent_factory = AgentFactory(self.config)

    def get_response(self, messages: List[Dict]) -> str:
        logger.info(f"Received input, updating memory...")
        self.agent_handler.update_memory(messages)

        # First: if persona is empty it needs to be generated
        if self.config.persona.is_empty():
            logger.info("Updating persona...")
            update_persona(agent_handler=self.agent_handler, chroma_db=self.chroma_db, config=self.config)
            logger.info(f"Updated persona: {self.config.persona.get_summary()}")

        # On to pre-processing!
        logger.info("Pre-processing...")
        last_message = messages[-1]["content"]
        preprocessing_agent_output = \
            self.agent_handler.preprocessing_agent.run(PreprocessingAgentInputSchema(user_input=last_message))

        # Enrich context, given input
        if preprocessing_agent_output.detected_intent in ("coaching_request", "document_upload"):  # HACK for now
            logging.info("Detected coaching request...")

            query_result, relevant_search_results = None, None
            if preprocessing_agent_output.requires_research:
                logger.info("Research is required")
                query_result, search_result = do_research(
                    agent_handler=self.agent_handler,
                    chroma_db=self.chroma_db,
                    config=self.config,
                    user_input=last_message
                )

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
