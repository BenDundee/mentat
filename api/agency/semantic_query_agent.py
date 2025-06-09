from api.agency._agent import _Agent
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate

from api.api_configurator import APIConfigurator
from api.util import VectorDBMetadataCollector
from api.services import VectorStoreService
from api.interfaces import ConversationState


class SemanticQueryAgent(_Agent):
    """
    Agent responsible for formulating semantic search queries using an LLM.

    This agent leverages an LLM to transform user questions or context into
    effective semantic search queries that can be executed against vector databases.
    """

    def __init__(self, config: APIConfigurator, n_queries: int = 3):
        """
        Initialize the QueryAgent with LLM and prompt configuration.

        Args:
            config: Configuration containing LLM and prompt settings
        """
        self.llm_provider = config.llm_manager
        self.llm_params = config.prompt_manager.get_llm_settings("query_agent")
        self.prompt = config.prompt_manager.get_prompt("query_agent")
        self.template = self.prompt.template.partial(n_queries=n_queries)

        self.vector_db_location = config.vector_db_dir

    def run(self, state: ConversationState) -> ConversationState:
        """
        Generate semantic search queries based on input context.

        Args:
            state: Dictionary containing input information and vector DB metadata

        Returns:
            Dictionary containing the generated semantic queries
        """

        # Extract vector DB metadata if available
        vector_db_metadata = get_vectordb_metadata()
        self.template = self.template.partial(vector_db_info=_get_md_prompt(vector_db_metadata))

        # Build the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.template),
            ("user", _format_user_prompt(context=input_context))
        ])

        # Generate the semantic query using the LLM
        response = (
            prompt
            | self.llm_provider.llm(self.llm_params)
        ).invoke(input_context)

        # Extract and parse the response
        try:
            # Parse the response content as needed

            # Need to be more opinionated here. Need to return a list of objects, each one has:
            #  - `collection_name`: the collection to be queried
            #  - `query`: the query string
            #  - `filters`: a dictionary of metadata filters to be applied to the query
            #  - `top_k`: the number of results to return
            #  - `reasoning`: the reasoning behind the query
            #
            # Then we need a helper function (probably in vectore_store_Service.py) that consumes this object and
            # executes the queries against the vector database.







            query_result = {
                "generated_query": response.content,
                "original_input": input_text,
                "success": True
            }

            return query_result

        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "original_input": input_text,
                "success": False
            }

    def formulate_query(self,
                        input_text: str,
                        db_metadata: Dict[str, Any] = None,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        state = {"input": input_text, "context": context or {}, **db_metadata}

        return self.run(state)



# -------------------------- HELPERS
def _get_md_prompt() -> str:
    """ Get vector DB metadata """
    vector_db = VectorStoreService()
    vector_db_info = VectorDBMetadataCollector(vector_db).collect_metadata()

    prompt = "Here's information about our vector database that may help you formulate a better query:\n\n"

    # Add information about available collections
    collections = vector_db_info.get("available_collections", [])
    if collections:
        prompt += f"  - Available collections: {', '.join(collections)}\n\n"

    # Add schema information if available
    schemas = vector_db_info.get("collection_schemas", {})
    if schemas:
        prompt += "  - Collection schemas:\n"
        for collection, schema in schemas.items():
            prompt += f"- {collection}: {schema}\n"
        prompt += "\n"

    # Add metadata fields information
    metadata_fields = vector_db_info.get("metadata_fields", {})
    if metadata_fields:
        prompt += "  - Available metadata fields for filtering:\n"
        for collection, fields in metadata_fields.items():
            prompt += f"- {collection}: {', '.join(fields)}\n"
        prompt += "\n"

    # Add sample documents if available (but keep it concise)
    sample_docs = vector_db_info.get("sample_documents", {})
    if sample_docs:
        prompt += "  - Sample document excerpts (to understand content format):\n"
        for collection, samples in sample_docs.items():
            if isinstance(samples, list) and samples:
                prompt += f"- {collection} example: {samples[0][:200]}...\n"
        prompt += "\n"

    return prompt



if __name__ == "__main__":

    from api.interfaces import LLMCredentials
    from api.managers import LLMManager, PromptManager
    import os

    class FakeConfig:
        def __init__(self, llm_config: LLMCredentials):
            self.prompt_manager = PromptManager()
            self.llm_manager = LLMManager(llm_config)

    llm_client_config = LLMCredentials(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
    )

    cfg = FakeConfig(llm_client_config)
    query_agent = SemanticQueryAgent(cfg)

    state = {
        "input": "What is the best way to learn Python?",
        "context": {
            "available_collections": ["python_questions", "python_answers"],
            "collection_schemas": {
                "python_questions": "Questions about Python programming language",
                "python_answers": "Answers to Python programming questions"
            }
        }
    }
    result = query_agent.run(state)
    print(result)
