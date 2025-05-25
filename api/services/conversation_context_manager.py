class ConversationContextManager:
    """Manages conversation context throughout the interaction lifecycle."""

    def __init__(self, llm_provider):
        self.llm = llm_provider.get_llm("default")
        self.current_context = {
            "goals": [],
            "themes": [],
            "recent_insights": [],
            "tool_outputs": {},
            "user_profile": {},
        }

        # Create the insight extraction prompt
        self.insight_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Extract 2-3 key insights as a JSON list of strings."),
            HumanMessage(content="Tool output from {tool_name}: {output}")
        ])

        # Create the theme extraction prompt
        self.theme_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Identify 1-2 conversation themes as a JSON list of strings."),
            HumanMessage(content="User message: {message}")
        ])

        # Create chains
        self.insight_chain = self.insight_prompt | self.llm | StrOutputParser()
        self.theme_chain = self.theme_prompt | self.llm | StrOutputParser()

    def update_with_tool_output(self, tool_name: str, output: str) -> None:
        """Update context with output from a tool."""
        self.current_context["tool_outputs"][tool_name] = output

        # Extract insights from tool output
        try:
            insights_json = self.insight_chain.invoke({
                "tool_name": tool_name,
                "output": output
            })

            import json
            insights = json.loads(insights_json)
            if isinstance(insights, list):
                self.current_context["recent_insights"].extend(insights)
        except:
            pass  # Silently fail on extraction errors

    def update_with_user_message(self, message: str) -> None:
        """Update context based on user message."""
        # Extract themes, update user profile, etc.
        new_themes = self._extract_themes(message)
        if new_themes:
            for theme in new_themes:
                if theme not in self.current_context["themes"]:
                    self.current_context["themes"].append(theme)

    def get_writing_context(self) -> dict:
        """Get relevant context for the writer agent."""
        # Filter and organize context specifically for response writing
        return {
            "goals": self.current_context["goals"],
            "recent_insights": self.current_context["recent_insights"][-3:],
            "tool_outputs": self.current_context["tool_outputs"],
            "themes": self.current_context["themes"][-5:],
        }

    def get_critic_context(self) -> dict:
        """Get relevant context for the critic agent."""
        # May include different/additional context for evaluation
        return {
            "goals": self.current_context["goals"],
            "user_profile": self.current_context["user_profile"],
            "themes": self.current_context["themes"],
        }

    def _extract_insights(self, tool_name: str, output: str) -> List[str]:
        """Extract key insights from tool output."""
        # Use the LLM to extract insights
        prompt = f"""
        Extract 2-3 key insights from this {tool_name} output:

        {output}

        Return only the insights as a JSON list of strings.
        """
        try:
            response = self.llm.invoke(prompt)
            import json
            return json.loads(response)
        except:
            return []

    def _extract_themes(self, message: str) -> List[str]:
        """Extract themes from user message."""
        # Similar LLM-based extraction
        prompt = f"""
        Identify 1-2 conversation themes from this message:

        {message}

        Return only the themes as a JSON list of strings.
        """
        try:
            response = self.llm.invoke(prompt)
            import json
            return json.loads(response)
        except:
            return []