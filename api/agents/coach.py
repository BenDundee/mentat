from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from typing import List, Dict, Any, Optional
import sqlite3
import datetime
import uuid

# Import your custom tool classes
from .tools import GoalTrackerTool, JournalManagerTool, ConversationSearchTool
from .llm_provider import LLMProvider
from api.util import PromptManager


class ExecutiveCoachAgent:
    def __init__(self, llm_config: Optional[Dict[str, Dict[str, Any]]] = None, prompts_dir: str = "prompts"):
        # Initialize prompt manager
        self.prompt_manager = PromptManager(prompts_dir)

        # Initialize LLM provider
        self.llm_provider = LLMProvider(llm_config)

        # Initialize databases and tools
        self.init_databases()
        self.init_tools()

        # Get the main system prompt
        system_prompt_template = self.prompt_manager.get_prompt("executive_coach_system")

        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm_provider.get_llm("default",
                                          **self.prompt_manager.get_llm_settings("executive_coach_system")),
            tools=self.tools,
            prompt=system_prompt_template
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def init_tools(self):
        """Initialize all tools with their dependencies and appropriate LLMs."""
        # Get specialized LLMs for different tools
        creative_llm = self.llm_provider.get_llm("creative")  # For journal prompts
        analytical_llm = self.llm_provider.get_llm("analytical")  # For goal analysis

        # Create tool instances
        self.goal_tracker = GoalTrackerTool(
            db_connection=self.conn,
            llm=analytical_llm  # Use analytical LLM for goal tracking
        )

        self.journal_manager = JournalManagerTool(
            db_connection=self.conn,
            vector_db=self.vector_db,
            llm=creative_llm  # Use creative LLM for journal prompts
        )

        self.conversation_search = ConversationSearchTool(
            vector_db=self.vector_db
        )

        # Create workflow orchestrator with access to LLM provider
        self.workflow_orchestrator = self._create_workflow_orchestrator()

        # Register all tools with the agents
        self.tools = [
            self.goal_tracker,
            self.journal_manager,
            self.conversation_search,
            self.workflow_orchestrator
        ]

    def _create_workflow_orchestrator(self):
        """Create a meta-tool that can orchestrate multi-step workflows."""

        class WorkflowOrchestratorTool(BaseTool):
            # ... similar to before, but with LLM provider instead of single LLM

            def __init__(self, goal_tracker, journal_manager, conversation_search, llm_provider):
                super().__init__()
                self.goal_tracker = goal_tracker
                self.journal_manager = journal_manager
                self.conversation_search = conversation_search
                self.llm_provider = llm_provider

                # Define available workflows
                self.workflows = {
                    "goal_review": self._goal_review_workflow,
                    "reflection_session": self._reflection_workflow,
                    "planning_session": self._planning_workflow
                }

            def _goal_review_workflow(self, parameters):
                """Review goals and suggest next steps."""
                user_id = parameters.get("user_id", "default_user")

                # Step 1: Get current goals
                goals_result = self.goal_tracker._run(f"list goals", user_id=user_id)

                # Step 2: Find relevant past conversations about these goals
                goal_context = self.conversation_search._run(f"goals progress", user_id=user_id)

                # Step 3: Generate personalized reflection using analytical LLM
                analytical_llm = self.llm_provider.get_llm("analytical")

                prompt = f"""Based on the following information about the user's goals:

                {goals_result}

                And relevant past discussions:
                {goal_context}

                Provide a thoughtful review of their goal progress, highlighting:
                1. Areas of success
                2. Opportunities for improvement
                3. Suggested next steps
                4. Reflective questions to consider
                """

                from langchain.schema import HumanMessage
                response = analytical_llm([HumanMessage(content=prompt)])

                # Step 4: Suggest a journal prompt as follow-up using creative LLM
                journal_prompt = self.journal_manager._run(f"create prompt for goal reflection", user_id=user_id)

                return f"{response.content}\n\n{journal_prompt}"

            # ... other workflow methods

        # Create and return the workflow orchestrator with LLM provider
        return WorkflowOrchestratorTool(
            goal_tracker=self.goal_tracker,
            journal_manager=self.journal_manager,
            conversation_search=self.conversation_search,
            llm_provider=self.llm_provider
        )

    def init_databases(self):
        """Initialize database connections."""
        # SQLite for structured data
        self.conn = sqlite3.connect("executive_coach.db")
        self._init_sql_tables()

        # Vector DB for semantic search
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = Chroma(
            collection_name="conversation_history",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def _init_sql_tables(self):
        """Initialize SQLite tables."""
        cursor = self.conn.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            user_id TEXT UNIQUE,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create goals table
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

        # Create journal_entries table
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

    def run(self, message: str, history: Optional[List[List[str]]] = None, user_id: str = "default_user"):
        """Run the executive coach agents with the given input."""
        # Ensure user exists in DB
        self._ensure_user_exists(user_id)

        # Update in-memory conversation buffer with history if provided
        if history:
            for human_msg, ai_msg in history:
                self.memory.chat_memory.add_user_message(human_msg)
                self.memory.chat_memory.add_ai_message(ai_msg)

        # Get relevant past conversations from vector store
        relevant_history = self.conversation_search._run(message, user_id)

        # Add relevant history as context
        enhanced_message = message
        if relevant_history:
            enhanced_message += f"\n\nContext from past conversations:\n{relevant_history}"

        # Additional context for tool selection
        intent_context = self._determine_user_intent(message)
        if intent_context:
            enhanced_message += f"\n\n{intent_context}"

        # Run the agents with user_id in the metadata for tool access
        response = self.agent_executor.run(
            input=enhanced_message,
            user_id=user_id  # Pass user_id to all tools
        )

        # Store this interaction in vector DB
        self._store_interaction(user_id, message, response)

        return response

    def _determine_user_intent(self, message: str) -> str:
        """Use a simple keyword-based approach to help guide tool selection."""
        message_lower = message.lower()

        if any(kw in message_lower for kw in ["goal", "objective", "target", "achieve"]):
            return "This message appears to be about goals. Consider using the GoalTracker tool."

        elif any(kw in message_lower for kw in ["journal", "reflect", "write", "thought"]):
            return "This message appears to be about journaling. Consider using the Journal tool."

        elif any(kw in message_lower for kw in ["remember", "last time", "previously", "you said"]):
            return "This message references past conversations. Consider using the SearchPastConversations tool."

        elif any(kw in message_lower for kw in ["review", "progress", "how am i doing"]):
            return "This message is requesting a progress review. Consider using the WorkflowOrchestrator with the goal_review workflow."

        return ""

    def _ensure_user_exists(self, user_id: str):
        """Make sure the user exists in the SQL database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
            self.conn.commit()

    def _store_interaction(self, user_id: str, user_message: str, bot_message: str):
        """Store conversation in vector DB for semantic search."""
        # Format the conversation text
        combined_text = f"User: {user_message}\nBot: {bot_message}"

        # Store in vector DB with metadata
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