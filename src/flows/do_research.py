
import logging
from typing import Tuple

from src.agents import AgentHandler
from src.configurator import Configurator
from src.services import ChromaDBService
from src.agents.types import (
    QueryAgentInputSchema, SearchAgentInputSchema, SearchToolInputSchema, SearchToolOutputSchema, QueryResult
)

logger = logging.getLogger(__name__)

def do_research(
        agent_handler: AgentHandler,
        chroma_db: ChromaDBService,
        config: Configurator,
        user_input: str
) -> tuple[QueryResult, QueryResult]:
    # Query agent
    logger.info("Querying document store...")
    query_agent_output = agent_handler.query_agent.run(QueryAgentInputSchema(user_input=user_input))
    query_result = chroma_db.query(query_agent_output.query, n_results=config.data_config.db_results_per_query)

    # Search agent
    logger.info("Searching internet...")
    search_agent_output = (
        agent_handler.search_agent.run(
            SearchAgentInputSchema(user_input=user_input,
                                   num_queries=config.data_config.search_engine_num_queries))
    )
    search_results = agent_handler.search_tool.run(SearchToolInputSchema(queries=search_agent_output.queries))
    relevant_search_results = chroma_db.refine_search_results(
        search_results=search_results.results
        , query=query_agent_output.query
        , n_results=10
    )

    return query_result, relevant_search_results