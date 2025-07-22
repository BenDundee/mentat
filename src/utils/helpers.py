from atomic_agents.lib.components.agent_memory import Message
from src.interfaces.chat import SimpleMessageContentIOSchema
from typing import Tuple


def get_message(role: str, message: str, turn_id: str = None):
    return Message(role=role, content=SimpleMessageContentIOSchema(content=message), turn_id=turn_id)


def strip_message(message: Message) -> Tuple[str, str]:
    return message.role, str(message.content)