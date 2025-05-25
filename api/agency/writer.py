from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional


class WriterAgent:
    """Agent responsible for crafting well-structured responses based on tool outputs."""

    def __init__(self, llm_provider):
        # Use a model optimized for creative writing
        self.llm = llm_provider.get_llm("creative", temperature=0.7)

        # Create the prompt template for the writer
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are an executive coach writer agent. Your role is to craft well-structured,
                personalized responses that help professionals achieve their goals.

                Your writing should:
                1. Acknowledge the user's specific situation and needs
                2. Incorporate relevant insights from tool outputs
                3. Use a supportive yet challenging tone
                4. Provide actionable advice
                5. Be concise but thorough
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="""
                User message: {user_message}

                Context: {context}

                Tool outputs: {tool_outputs}

                Draft a response that addresses the user's needs while incorporating
                relevant information from the tool outputs and context.
            """)
        ])

        # Create the chain
        self.chain = (
                {"user_message": RunnablePassthrough(),
                 "context": RunnablePassthrough(),
                 "tool_outputs": RunnablePassthrough(),
                 "chat_history": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def run(self,
            user_message: str,
            context: Dict[str, Any],
            tool_outputs: Dict[str, str],
            chat_history: Optional[List] = None) -> str:
        """
        Generate a response using the writer agent.

        Args:
            user_message: The user's message
            context: Dictionary containing relevant context
            tool_outputs: Dictionary of tool outputs by tool name
            chat_history: Optional chat history

        Returns:
            Crafted response as string
        """
        # Format tool outputs for the prompt
        formatted_tool_outputs = "\n\n".join(
            [f"{tool_name}: {output}" for tool_name, output in tool_outputs.items()]
        )

        # Format context for the prompt
        formatted_context = "\n".join(
            [f"{key}: {value}" for key, value in context.items()]
        )

        # Default empty chat history if none provided
        if chat_history is None:
            chat_history = []

        # Run the chain
        return self.chain.invoke({
            "user_message": user_message,
            "context": formatted_context,
            "tool_outputs": formatted_tool_outputs,
            "chat_history": chat_history
        })