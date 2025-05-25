from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Optional, Dict, Any, Tuple
import sqlite3
import datetime
import uuid
import logging

from api.agency.writer import WriterAgent
from api.agency.critic import CriticAgent
from api.agency.agent_processing_graph import AgentProcessingGraph
from api.toolbox import (
    tool_registry,
    ToolExecutionContext
)
from api.services import ConversationContextManager, WorkflowOrchestrator
from api.api_configurator import APIConfigurator
from api.interfaces import ChatResponse

logger = logging.getLogger(__name__)


class ExecutiveCoachAgent:
    """
    Executive coach agent that uses a chain of specialized components:
    1. Tool selection and execution
    2. Context management
    3. Response writing and critique

    Built with modern LangChain patterns using the LCEL framework.
    """

    def __init__(self, config: APIConfigurator):
        self.config = config
        self.prompt_manager = config.prompt_manager
        self.llm_provider = config.llm_provider

        # Initialize databases
        self.init_databases()

        # Create context manager
        self.context_manager = ConversationContextManager(self.llm_provider)

        # Create tools registry
        self.tool_registry = tool_registry(self.vector_db, self.conn)

        # Get specific tools needed for workflow orchestrator
        self.goal_tracker = self.tool_registry.get_tool("goal_tracker")
        self.journal_manager = self.tool_registry.get_tool("journal_manager")
        self.conversation_search = self.tool_registry.get_tool("conversation_search")

        # Create workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            goal_tracker=self.goal_tracker,
            journal_manager=self.journal_manager,
            conversation_search=self.conversation_search,
            llm_provider=self.llm_provider,
            context_manager=self.context_manager
        )

        self.processing_graph = AgentProcessingGraph({
            "context_manager": self.context_manager,
            "conversation_search": self.conversation_search,
            # ... other dependencies
        })

        # Create executive coach agent with tools
        self._create_agent()

        # Create writer and critic agents
        self.writer_agent = WriterAgent(self.llm_provider)
        self.critic_agent = CriticAgent(self.llm_provider)

        # Create the main agent chain
        self._create_agent_chain()

    def _create_agent(self):
        """Create the main tool-using agent."""
        # Get the main system prompt
        system_prompt_template = self.prompt_manager.get_prompt("executive_coach_system")

        # Use OpenAI functions agent if available, fallback to React
        try:
            # First try to create an OpenAI functions agent
            self.agent = create_openai_functions_agent(
                llm=self.llm_provider.get_llm("default",
                                              **self.prompt_manager.get_llm_settings("executive_coach_system")),
                tools=self.tool_registry.get_tools(),
                prompt=system_prompt_template
            )
        except Exception as e:
            logger.info(f"Falling back to React agent: {e}")
            # Fallback to React agent
            self.agent = create_react_agent(
                llm=self.llm_provider.get_llm("default",
                                              **self.prompt_manager.get_llm_settings("executive_coach_system")),
                tools=self.tool_registry.get_tools(),
                prompt=system_prompt_template
            )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tool_registry.get_tools(),
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _create_agent_chain(self):
        """Create the complete agent processing chain using LCEL."""

        # Define the input processing function
        def process_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Process and enhance the user input."""
            message = inputs["message"]
            user_id = inputs.get("user_id", "default_user")

            # Update context with user message
            self.context_manager.update_with_user_message(message)

            # Ensure user exists
            self._ensure_user_exists(user_id)

            # Get relevant conversation history
            relevant_history = self.conversation_search._run(message, user_id)

            # Determine user intent
            intent_context = self._determine_user_intent(message)

            # Enhance message with context
            enhanced_message = message
            if relevant_history:
                enhanced_message += f"\n\nContext from past conversations:\n{relevant_history}"
            if intent_context:
                enhanced_message += f"\n\n{intent_context}"

            return {
                "enhanced_message": enhanced_message,
                "user_id": user_id,
                "original_message": message,
                "history": inputs.get("history", [])
            }

        # Define the tool execution function
        def execute_tools(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Execute tools based on the user input."""
            enhanced_message = inputs["enhanced_message"]
            user_id = inputs["user_id"]

            # Capture tool outputs
            tool_outputs = {}
            with ToolExecutionContext() as tool_context:
                agent_result = self.agent_executor.invoke({
                    "input": enhanced_message,
                    "user_id": user_id
                })
                tool_outputs = tool_context.get_tool_outputs()

            # Update context with tool outputs
            for tool_name, output in tool_outputs.items():
                self.context_manager.update_with_tool_output(tool_name, output)

            return {
                **inputs,
                "tool_outputs": tool_outputs,
                "agent_result": agent_result.get("output", "")
            }

        # Define the response generation function
        def generate_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a response using writer and critic agents."""
            message = inputs["original_message"]
            user_id = inputs["user_id"]
            tool_outputs = inputs["tool_outputs"]
            history = inputs.get("history", [])

            # Convert history to messages
            chat_history = []
            for h in history:
                if len(h) == 2:
                    human_msg, ai_msg = h
                    chat_history.append(HumanMessage(content=human_msg))
                    chat_history.append(AIMessage(content=ai_msg))

            # Get writing context
            writing_context = self.context_manager.get_writing_context()

            # Generate draft with writer agent
            draft_response = self.writer_agent.run(
                user_message=message,
                context=writing_context,
                tool_outputs=tool_outputs,
                chat_history=chat_history
            )

            # Evaluate with critic
            evaluation = self.critic_agent.evaluate(
                user_message=message,
                draft_response=draft_response,
                context=self.context_manager.get_critic_context()
            )

            # Use revised response if needed
            final_response = evaluation.revised_response if evaluation.needs_revision else draft_response

            # Store the interaction
            self._store_interaction(user_id, message, final_response)

            return {
                **inputs,
                "final_response": final_response
            }

        # Create the chain
        self.chain = (
                RunnableLambda(process_input)
                | RunnableLambda(execute_tools)
                | RunnableLambda(generate_response)
        )

    def init_databases(self):
        """Initialize database connections."""
        # SQLite for structured data
        self.conn = sqlite3.connect("executive_coach.db")
        self._init_sql_tables()

        # Vector DB for semantic search
        self.embeddings = OpenAIEmbeddings(api_key=self.config.llm_client_config.openai_api_key)
        self.vector_db = Chroma(
            collection_name="conversation_history",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def _init_sql_tables(self):
        """Initialize SQLite tables."""
        cursor = self.conn.cursor()

        # Create database tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            user_id TEXT UNIQUE,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            target_date TIMESTAMP,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            goal_id INTEGER,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (goal_id) REFERENCES goals (id)
        )
        ''')

        self.conn.commit()

    def run(self, message: str, history: Optional[List[List[str]]] = None,
            user_id: str = "default_user") -> ChatResponse:
        """Process a user message and generate a response."""
        try:
            # Run the chain
            result = self.chain.invoke({
                "message": message,
                "history": history,
                "user_id": user_id
            })

            # Return the final response
            return ChatResponse(response=result["final_response"])
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return ChatResponse(
                response="I'm sorry, I encountered an error while processing your request. "
                         "Please try again or contact support if the issue persists."
            )

    def _determine_user_intent(self, message: str) -> str:
        """Determine user intent from message."""
        matched_intents = self.tool_registry.detect_intent(message)

        if not matched_intents:
            return ""

        intent_contexts = []
        for intent_name, intent in matched_intents:
            intent_contexts.append(
                f"This message appears to be about {intent.description}. "
                f"Consider using the {intent.tool_name} tool."
            )

        return "\n".join(intent_contexts)

    def _ensure_user_exists(self, user_id: str):
        """Ensure user exists in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
            self.conn.commit()

    def _store_interaction(self, user_id: str, user_message: str, bot_message: str):
        """Store conversation in vector database."""
        combined_text = f"User: {user_message}\nBot: {bot_message}"

        self.vector_db.add_texts(
            texts=[combined_text],
            metadatas=[{
                "user_id": user_id,
                "timestamp": str(datetime.datetime.utcnow()),
                "type": "conversation"
            }],
            ids=[str(uuid.uuid4())]
        )
        self.vector_db.persist()