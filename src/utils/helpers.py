from atomic_agents.lib.components.agent_memory import Message
from src.types.chat import SimpleMessageContentIOSchema


def get_message(role: str, message: str):
    return Message(role=role, content=SimpleMessageContentIOSchema(content=message))