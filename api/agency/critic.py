from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
import json


class CriticEvaluation(BaseModel):
    """Schema for critic evaluation results"""
    relevance_score: int = Field(description="Rating 1-5 on how relevant the response is to user's query")
    actionability_score: int = Field(description="Rating 1-5 on how actionable the advice is")
    empathy_score: int = Field(description="Rating 1-5 on level of empathy shown")
    clarity_score: int = Field(description="Rating 1-5 on clarity and structure")
    needs_revision: bool = Field(description="Whether the response needs revision")
    improvement_reasons: Optional[List[str]] = Field(description="Reasons for needed improvements")
    revised_response: Optional[str] = Field(description="Revised response if needed")


class CriticAgent:
    """Agent responsible for evaluating and improving responses."""

    def __init__(self, llm_provider):
        # Use a model optimized for analytical thinking
        self.llm = llm_provider.get_llm("analytical", temperature=0.2)

        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=CriticEvaluation)

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
                You are an executive coach critic agent. Your role is to evaluate responses
                for quality and suggest improvements where needed.

                Analyze the draft response based on these criteria:
                1. Relevance: Does it directly address the user's needs?
                2. Actionability: Does it provide clear, executable advice?
                3. Empathy: Does it acknowledge the user's situation appropriately?
                4. Clarity: Is it clear and well-structured?

                {self.parser.get_format_instructions()}
            """),
            HumanMessage(content="""
                User message: {user_message}

                Draft response: {draft_response}

                User goals: {goals}

                Evaluate this response and determine if it needs revision.
                If it does, provide specific reasons and a revised version.
            """)
        ])

        # Create the chain
        self.chain = (
                {"user_message": RunnablePassthrough(),
                 "draft_response": RunnablePassthrough(),
                 "goals": RunnablePassthrough()}
                | self.prompt
                | self.llm
        )

    def evaluate(self,
                 user_message: str,
                 draft_response: str,
                 context: Dict[str, Any]) -> CriticEvaluation:
        """
        Evaluate a draft response and suggest improvements if needed.

        Args:
            user_message: The user's message
            draft_response: The draft response to evaluate
            context: Dictionary containing relevant context including goals

        Returns:
            CriticEvaluation object with scores and potential revisions
        """
        # Format goals for the prompt
        goals = ", ".join(context.get("goals", ["No specific goals"]))

        # Run the chain
        result = self.chain.invoke({
            "user_message": user_message,
            "draft_response": draft_response,
            "goals": goals
        })

        # Parse the result
        try:
            # Try to extract JSON from the response
            json_str = result.content if hasattr(result, 'content') else str(result)

            # Find JSON in the response if it's embedded in text
            import re
            json_match = re.search(r'```json\n([\s\S]*?)\n```', json_str)
            if json_match:
                json_str = json_match.group(1)

            # Parse as JSON and then into our Pydantic model
            json_obj = json.loads(json_str)
            return CriticEvaluation(**json_obj)
        except Exception as e:
            # Fallback for parsing errors
            return CriticEvaluation(
                relevance_score=3,
                actionability_score=3,
                empathy_score=3,
                clarity_score=3,
                needs_revision=False,
                improvement_reasons=["Error parsing critic output: " + str(e)],
                revised_response=None
            )