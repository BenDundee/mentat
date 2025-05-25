from langgraph.graph import StateGraph
from typing import Dict, TypedDict, List, Union, Any
import uuid


class CoachState(TypedDict):
    """State for the executive coach workflow."""
    user_id: str
    input: str
    goal_info: Union[str, None]
    journal_info: Union[str, None]
    conversation_history: Union[str, None]
    workflow_type: str
    output: Union[str, None]


class WorkflowOrchestrator:
    """Manages different workflows using LangGraph."""

    def __init__(self, goal_tracker, journal_manager, conversation_search, llm_provider):
        self.goal_tracker = goal_tracker
        self.journal_manager = journal_manager
        self.conversation_search = conversation_search
        self.llm_provider = llm_provider

        # Create workflow graphs
        self.workflows = {
            "goal_review": self._create_goal_review_workflow(),
            "reflection_session": self._create_reflection_workflow(),
            "planning_session": self._create_planning_workflow()
        }

    def _create_goal_review_workflow(self):
        """Create goal review workflow graph."""
        # Define the workflow as a graph
        workflow = StateGraph(CoachState)

        # Define the nodes for this workflow
        workflow.add_node("fetch_goals", self._fetch_goals)
        workflow.add_node("get_conversation_context", self._get_conversation_context)
        workflow.add_node("analyze_progress", self._analyze_goal_progress)
        workflow.add_node("generate_journal_prompt", self._generate_journal_prompt)
        workflow.add_node("final_response", self._create_final_response)

        # Define the edges (workflow)
        workflow.add_edge("fetch_goals", "get_conversation_context")
        workflow.add_edge("get_conversation_context", "analyze_progress")
        workflow.add_edge("analyze_progress", "generate_journal_prompt")
        workflow.add_edge("generate_journal_prompt", "final_response")

        # Set the entry point
        workflow.set_entry_point("fetch_goals")

        # Compile the workflow
        return workflow.compile()

    def _create_reflection_workflow(self):
        """Create reflection session workflow graph."""
        # Similar pattern to goal review workflow
        workflow = StateGraph(CoachState)
        # Add nodes and edges for reflection workflow
        # ...
        return workflow.compile()

    def _create_planning_workflow(self):
        """Create planning session workflow graph."""
        # Similar pattern to goal review workflow
        workflow = StateGraph(CoachState)
        # Add nodes and edges for planning workflow
        # ...
        return workflow.compile()

    # Node implementations
    def _fetch_goals(self, state: CoachState) -> Dict:
        """Node to fetch user's goals."""
        user_id = state["user_id"]
        goals_result = self.goal_tracker._run("list goals", user_id=user_id)
        return {"goal_info": goals_result}

    def _get_conversation_context(self, state: CoachState) -> Dict:
        """Node to get relevant conversation history."""
        user_id = state["user_id"]
        goal_context = self.conversation_search._run("goals progress", user_id=user_id)
        return {"conversation_history": goal_context}

    def _analyze_goal_progress(self, state: CoachState) -> Dict:
        """Node to analyze goal progress with analytical LLM."""
        analytical_llm = self.llm_provider.get_llm("analytical")

        prompt = f"""Based on the following information about the user's goals:

        {state["goal_info"]}

        And relevant past discussions:
        {state["conversation_history"]}

        Provide a thoughtful review of their goal progress, highlighting:
        1. Areas of success
        2. Opportunities for improvement
        3. Suggested next steps
        4. Reflective questions to consider
        """

        from langchain.schema import HumanMessage
        response = analytical_llm([HumanMessage(content=prompt)])

        return {"output": response.content}

    def _generate_journal_prompt(self, state: CoachState) -> Dict:
        """Node to generate a journal prompt."""
        user_id = state["user_id"]
        journal_prompt = self.journal_manager._run("create prompt for goal reflection", user_id=user_id)

        # Append to the existing output
        return {"journal_info": journal_prompt}

    def _create_final_response(self, state: CoachState) -> Dict:
        """Create the final combined response."""
        return {"output": f"{state['output']}\n\n{state['journal_info']}"}

    def run_workflow(self, workflow_type: str, user_input: str, user_id: str = "default_user") -> str:
        """Run a specific workflow."""
        if workflow_type not in self.workflows:
            return f"Workflow '{workflow_type}' not found. Available workflows: {', '.join(self.workflows.keys())}"

        # Initialize state
        initial_state = {
            "user_id": user_id,
            "input": user_input,
            "goal_info": None,
            "journal_info": None,
            "conversation_history": None,
            "workflow_type": workflow_type,
            "output": None
        }

        # Run the workflow
        workflow = self.workflows[workflow_type]
        final_state = workflow.invoke(initial_state)

        return final_state["output"]