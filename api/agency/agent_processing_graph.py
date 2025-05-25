from langgraph.graph import StateGraph
import datetime
from typing import Dict, Any, Optional


class AgentProcessingGraph:
    """Creates a graph-based processing pipeline for the agent."""

    def __init__(self, dependencies: Dict[str, Any]):
        """Initialize with required dependencies."""
        self.dependencies = dependencies
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the processing graph."""
        # Create a state graph with Dictionary state
        graph = StateGraph(Dict)

        # Add nodes
        graph.add_node("process_input", self._process_input)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("generate_response", self._generate_response)

        # Add edges
        graph.add_edge("process_input", "execute_tools")
        graph.add_edge("execute_tools", "generate_response")

        # Set entry point
        graph.set_entry_point("process_input")

        # Compile the graph
        return graph.compile()

    def _process_input(self, state: Dict) -> Dict:
        """
        Process the user input to enhance it with context and prepare it for tool execution.

        This method:
        1. Extracts user message and metadata
        2. Updates the context manager with the new message
        3. Retrieves relevant conversation history
        4. Detects user intent and suggests relevant tools
        5. Enhances the message with additional context
        6. Ensures the user exists in the database

        Args:
            state: The current state dictionary containing at least the 'message' and 'user_id'

        Returns:
            Updated state dictionary with enhanced message and additional metadata
        """
        # Extract dependencies
        context_manager = self.dependencies["context_manager"]
        conversation_search = self.dependencies["conversation_search"]
        tool_registry = self.dependencies["tool_registry"]
        conn = self.dependencies["conn"]

        # Extract inputs from state
        message = state["message"]
        user_id = state.get("user_id", "default_user")
        history = state.get("history", [])

        # Log the incoming message
        logger = self.dependencies.get("logger")
        if logger:
            logger.info(f"Processing message from user {user_id}: {message[:50]}...")

        # Update context with user message
        context_manager.update_with_user_message(message)

        # Ensure user exists in database
        self._ensure_user_exists(user_id, conn)

        # Extract message features (optional - for more sophisticated processing)
        message_features = self._extract_message_features(message)

        # Get relevant conversation history
        relevant_history = conversation_search._run(
            query=message,
            user_id=user_id,
            limit=5  # Configurable number of relevant history items
        )

        # Detect user intent and suggest tools
        matched_intents = tool_registry.detect_intent(message)
        intent_context = ""
        suggested_tools = []

        if matched_intents:
            intent_contexts = []
            for intent_name, intent in matched_intents:
                intent_contexts.append(
                    f"This message appears to be about {intent.description}. "
                    f"Consider using the {intent.tool_name} tool."
                )
                suggested_tools.append(intent.tool_name)
            intent_context = "\n".join(intent_contexts)

        # Detect if message matches any workflow patterns
        workflow_orchestrator = self.dependencies.get("workflow_orchestrator")
        workflow_match = None
        if workflow_orchestrator:
            workflow_match = self._detect_workflow_match(message, workflow_orchestrator)

        # Enhance message with context
        enhanced_message = message

        # Add relevant conversation history if available
        if relevant_history:
            enhanced_message += f"\n\nContext from past conversations:\n{relevant_history}"

        # Add intent suggestions if available
        if intent_context:
            enhanced_message += f"\n\n{intent_context}"

        # Add workflow suggestion if matched
        if workflow_match:
            enhanced_message += f"\n\nThis appears to be a request for a {workflow_match} session."

        # Update state with processed information
        updated_state = {
            **state,
            "enhanced_message": enhanced_message,
            "original_message": message,
            "user_id": user_id,
            "history": history,
            "relevant_history": relevant_history,
            "message_features": message_features,
            "suggested_tools": suggested_tools,
            "workflow_match": workflow_match,
            "processing_timestamp": str(datetime.datetime.utcnow()),
            "processing_metadata": {
                "has_relevant_history": bool(relevant_history),
                "has_intent_match": bool(matched_intents),
                "message_length": len(message),
                "suggested_workflow": workflow_match
            }
        }

        # Log completion of processing
        if logger:
            logger.debug(f"Input processing complete. Enhanced message length: {len(enhanced_message)}")

        return updated_state

    # TODO: Move this to Helper class
    def _ensure_user_exists(self, user_id: str, conn) -> None:
        """Ensure the user exists in the database."""
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
            conn.commit()

    # TODO: Move to Helper class
    def _extract_message_features(self, message: str) -> Dict[str, Any]:
        """
        Extract features from the message for advanced processing.

        This could include:
        - Sentiment analysis
        - Topic detection
        - Question detection
        - Command detection
        """
        features = {
            "length": len(message),
            "is_question": "?" in message,
            "contains_greeting": any(greeting in message.lower()
                                     for greeting in ["hello", "hi", "hey", "greetings"]),
            "contains_farewell": any(farewell in message.lower()
                                     for farewell in ["bye", "goodbye", "see you", "talk later"]),
            "word_count": len(message.split())
        }

        # Add more sophisticated features here if needed

        return features

    # TODO: Helper
    def _detect_workflow_match(self, message: str, workflow_orchestrator) -> Optional[str]:
        """
        Detect if the message matches any workflow patterns.

        Args:
            message: The user message
            workflow_orchestrator: The workflow orchestrator instance

        Returns:
            Workflow type name if matched, None otherwise
        """
        # Simple keyword matching - could be replaced with more sophisticated matching
        workflow_patterns = {
            "goal_review": ["review goals", "goal progress", "how am i doing", "goal status"],
            "reflection_session": ["reflect", "reflection", "think about", "looking back"],
            "planning_session": ["plan", "planning", "future", "next steps", "schedule"]
        }

        message_lower = message.lower()

        for workflow_type, patterns in workflow_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return workflow_type

        return None

    def _execute_tools(self, state: Dict) -> Dict:
        """Execute tools based on the processed input."""
        # Implementation...
        return state

    def _generate_response(self, state: Dict) -> Dict:
        """Generate the final response."""
        # Implementation...
        return state

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs through the graph."""
        return self.graph.invoke(inputs)