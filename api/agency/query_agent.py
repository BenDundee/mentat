from api.agency._agent import _Agent
from typing import Dict, Any, List, Optional, Union
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class QueryAgent(_Agent):
    """
    Agent responsible for formulating semantic search queries using an LLM.

    This agent leverages an LLM to transform user questions or context into
    effective semantic search queries that can be executed against vector databases.
    """

    def __init__(self, config):
        """
        Initialize the QueryAgent with LLM and prompt configuration.

        Args:
            config: Configuration containing LLM and prompt settings
        """
        self.llm_provider = config.llm_manager
        self.llm_params = config.prompt_manager.get_llm_settings("query_agent")
        self.prompt = config.prompt_manager.get_prompt("query_agent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate semantic search queries based on input context.

        Args:
            state: Dictionary containing input information and vector DB metadata

        Returns:
            Dictionary containing the generated semantic queries
        """
        # Extract input from state
        input_text = state.get("input", "")
        context = state.get("context", {})

        # Extract vector DB metadata if available
        vector_db_info = self._prepare_vector_db_info(state)

        # Build the input context with vector DB information
        input_context = {
            "input": input_text,
            "vector_db_info": vector_db_info,
            **context
        }

        # Build the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt.template),
            ("user", self._format_user_prompt(input_context))
        ])

        # Generate the semantic query using the LLM
        response = (
                prompt
                | self.llm_provider.get_chat_model(self.llm_params)
        ).invoke(input_context)

        # Extract and parse the response
        try:
            # Parse the response content as needed
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

    def _prepare_vector_db_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format vector database information to help the LLM.

        Args:
            state: State dictionary that may contain vector DB metadata

        Returns:
            Formatted vector DB information for the LLM
        """
        # Extract vector DB metadata if available
        collections = state.get("available_collections", [])
        schemas = state.get("collection_schemas", {})
        sample_documents = state.get("sample_documents", {})
        metadata_fields = state.get("metadata_fields", {})

        vector_db_info = {
            "available_collections": collections,
            "collection_schemas": schemas,
            "metadata_fields": metadata_fields
        }

        # Include sample documents if available but keep it concise
        if sample_documents:
            vector_db_info["sample_documents"] = sample_documents

        return vector_db_info

    def _format_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the user prompt with vector DB information.

        Args:
            context: Dictionary containing input and vector DB information

        Returns:
            Formatted user prompt string
        """
        input_text = context.get("input", "")
        vector_db_info = context.get("vector_db_info", {})
        purpose = context.get("purpose", "general")

        # Base prompt with the user's input
        prompt = f"I need to create a semantic search query for the following: {input_text}\n\n"

        # Add vector DB information if available
        if vector_db_info:
            prompt += "Here's information about our vector database that may help you formulate a better query:\n\n"

            # Add information about available collections
            collections = vector_db_info.get("available_collections", [])
            if collections:
                prompt += f"Available collections: {', '.join(collections)}\n\n"

            # Add schema information if available
            schemas = vector_db_info.get("collection_schemas", {})
            if schemas:
                prompt += "Collection schemas:\n"
                for collection, schema in schemas.items():
                    prompt += f"- {collection}: {schema}\n"
                prompt += "\n"

            # Add metadata fields information
            metadata_fields = vector_db_info.get("metadata_fields", {})
            if metadata_fields:
                prompt += "Available metadata fields for filtering:\n"
                for collection, fields in metadata_fields.items():
                    prompt += f"- {collection}: {', '.join(fields)}\n"
                prompt += "\n"

            # Add sample documents if available (but keep it concise)
            sample_docs = vector_db_info.get("sample_documents", {})
            if sample_docs:
                prompt += "Sample document excerpts (to understand content format):\n"
                for collection, samples in sample_docs.items():
                    if isinstance(samples, list) and samples:
                        prompt += f"- {collection} example: {samples[0][:200]}...\n"
                prompt += "\n"

        # Final instructions
        prompt += """
Based on this information, please generate a semantic search query that will effectively retrieve relevant information.
Your response should be in JSON format and include:
1. The generated query
2. A brief explanation of your reasoning
3. Suggested metadata filters (if applicable)
"""

        return prompt

    def formulate_query(self,
                        input_text: str,
                        db_metadata: Dict[str, Any] = None,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        state = {
                    "input": input_text,
                    "context": context or {},
                    **db_metadata or {}
        }
        return self.run(state)


if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock, patch


    class TestQueryAgent(unittest.TestCase):
        def setUp(self):
            self.config = MagicMock()
            self.config.llm_manager = MagicMock()
            self.config.prompt_manager = MagicMock()
            self.config.prompt_manager.get_llm_settings.return_value = {}
            self.config.prompt_manager.get_prompt.return_value = MagicMock(template="test template")
            self.agent = QueryAgent(self.config)

        def test_successful_query_generation(self):
            mock_response = MagicMock()
            mock_response.content = "Generated query text"
            self.config.llm_manager.get_chat_model.return_value.invoke.return_value = mock_response

            result = self.agent.formulate_query(
                input_text="test query",
                db_metadata={"available_collections": ["test_collection"]}
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["generated_query"], "Generated query text")
            self.assertEqual(result["original_input"], "test query")

        def test_error_handling(self):
            self.config.llm_manager.get_chat_model.return_value.invoke.side_effect = Exception("Test error")

            result = self.agent.formulate_query("test query")

            self.assertFalse(result["success"])
            self.assertIn("Failed to parse LLM response", result["error"])
            self.assertEqual(result["original_input"], "test query")

        def test_prepare_vector_db_info(self):
            state = {
                "available_collections": ["col1", "col2"],
                "collection_schemas": {"col1": "schema1"},
                "metadata_fields": {"col1": ["field1"]},
                "sample_documents": {"col1": ["sample1"]}
            }

            result = self.agent._prepare_vector_db_info(state)

            self.assertEqual(result["available_collections"], ["col1", "col2"])
            self.assertEqual(result["collection_schemas"], {"col1": "schema1"})
            self.assertEqual(result["metadata_fields"], {"col1": ["field1"]})
            self.assertEqual(result["sample_documents"], {"col1": ["sample1"]})

        unittest.main()
