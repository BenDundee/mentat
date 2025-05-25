from langgraph.graph import StateGraph
from typing import Dict, Any, List
import json


class WorkflowOrchestrator:
    """Manages different workflows using LangGraph with ConversationContextManager integration."""

    def __init__(self, goal_tracker, journal_manager, conversation_search, llm_provider, context_manager):
        self.goal_tracker = goal_tracker
        self.journal_manager = journal_manager
        self.conversation_search = conversation_search
        self.llm_provider = llm_provider
        self.context_manager = context_manager  # Now uses ConversationContextManager

        # Create workflow graphs
        self.workflows = {
            "goal_review": self._create_goal_review_workflow(),
            "reflection_session": self._create_reflection_workflow(),
            "planning_session": self._create_planning_workflow()
        }

    def _create_goal_review_workflow(self):
        """Create goal review workflow graph."""
        # Define the workflow as a graph
        workflow = StateGraph(Dict)  # Use a simple Dict instead of CoachState

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
        workflow = StateGraph(Dict)
        # Add nodes and edges for reflection workflow
        # ...
        return workflow.compile()

    def _create_planning_workflow(self):
        """Create planning session workflow graph."""
        # Similar pattern to goal review workflow
        workflow = StateGraph(Dict)
        # Add nodes and edges for planning workflow
        # ...
        return workflow.compile()

    # Node implementations
    def _fetch_goals(self, state: Dict) -> Dict:
        """Node to fetch user's goals and update context manager."""
        user_id = state["user_id"]
        goals_result = self.goal_tracker._run("list goals", user_id=user_id)

        # Parse goals from the result and update context manager
        try:
            # Assuming goals_result contains a structured representation of goals
            goals = self._extract_goals_from_result(goals_result)
            self.context_manager.current_context["goals"] = goals
        except Exception as e:
            print(f"Error extracting goals: {e}")

        return {"goal_info": goals_result}

    def _extract_goals_from_result(self, goals_result: str) -> List[str]:
        """Extract goal titles from the goal tracker result."""
        try:
            # Attempt to extract structured data - adapt this based on your goal format
            if "[" in goals_result and "]" in goals_result:
                # Try to extract JSON list
                import re
                goals_match = re.search(r'\[(.*)\]', goals_result)
                if goals_match:
                    goals_text = f"[{goals_match.group(1)}]"
                    return json.loads(goals_text)

            # Fallback: simple line splitting
            return [g.strip() for g in goals_result.split("\n")
                    if g.strip() and not g.strip().startswith("-")]
        except:
            # If parsing fails, return goal text as is
            return [goals_result]

    def _get_conversation_context(self, state: Dict) -> Dict:
        """Node to get relevant conversation history and update context."""
        user_id = state["user_id"]
        goal_context = self.conversation_search._run("goals progress", user_id=user_id)

        # Update context manager with relevant history
        if "themes" in self.context_manager.current_context:
            # Extract themes from the conversation history
            themes = self.context_manager._extract_themes(goal_context)
            for theme in themes:
                if theme not in self.context_manager.current_context["themes"]:
                    self.context_manager.current_context["themes"].append(theme)

        return {"conversation_history": goal_context}

    def _analyze_goal_progress(self, state: Dict) -> Dict:
        """Node to analyze goal progress with analytical LLM."""
        analytical_llm = self.llm_provider.get_llm("analytical")

        # Use context from the context manager
        goals = self.context_manager.current_context.get("goals", [])
        themes = self.context_manager.current_context.get("themes", [])

        prompt = f"""Based on the following information about the user's goals:

        {state["goal_info"]}

        And relevant past discussions:
        {state["conversation_history"]}

        Recent conversation themes: {', '.join(themes)}

        Provide a thoughtful review of their goal progress, highlighting:
        1. Areas of success
        2. Opportunities for improvement
        3. Suggested next steps
        4. Reflective questions to consider
        """

        from langchain.schema import HumanMessage
        response = analytical_llm([HumanMessage(content=prompt)])

        # Extract insights from the analysis and update context manager
        insights = self.context_manager._extract_insights("goal_analysis", response.content)
        if insights:
            self.context_manager.current_context["recent_insights"].extend(insights)

        return {"output": response.content}

    def _generate_journal_prompt(self, state: Dict) -> Dict:
        """Node to generate a journal prompt informed by context."""
        user_id = state["user_id"]

        # Use recent insights to inform the journal prompt
        recent_insights = self.context_manager.current_context.get("recent_insights", [])
        insights_context = ""
        if recent_insights:
            insights_context = f"Based on these insights: {', '.join(recent_insights[-3:])}, "

        # Enhance the journal prompt request with context
        journal_prompt_request = f"{insights_context}create prompt for goal reflection"
        journal_prompt = self.journal_manager._run(journal_prompt_request, user_id=user_id)

        # Add to context manager
        self.context_manager.update_with_tool_output("journal_manager", journal_prompt)

        return {"journal_info": journal_prompt}

    def _create_final_response(self, state: Dict) -> Dict:
        """Create the final combined response, leveraging context insights."""
        # Get any additional context that might enhance the response
        recent_insights = self.context_manager.current_context.get("recent_insights", [])
        themes = self.context_manager.current_context.get("themes", [])

        insight_section = ""
        if recent_insights:
            insight_section = "\n\nKey Insights:\n- " + "\n- ".join(recent_insights[-3:])

        response = f"{state['output']}\n\n{state['journal_info']}{insight_section}"

        # Store this comprehensive output in the context manager
        self.context_manager.update_with_tool_output("workflow_orchestrator", response)

        return {"output": response}

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

        # After workflow completes, extract any final insights
        if final_state.get("output"):
            self.context_manager._extract_insights(
                "workflow_" + workflow_type,
                final_state["output"]
            )

        return final_state["output"]