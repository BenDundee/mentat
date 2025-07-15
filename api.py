from flask import Flask, request, jsonify
import logging

from atomic_agents.lib.components.agent_memory import Message

from src.configurator import Configurator
from src.controller import Controller
from src.utils import get_message


app = Flask(__name__)


logger = logging.getLogger(__name__)
fmt = '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)

config = Configurator()
controller = Controller(config)

@app.route("/chat", methods=["POST"])
def chat():
    input_message = request.json.get("message")
    history = request.json.get("history")
    file = request.files.get("file")

    # TODO: consider managing `turn_id` in `ConversationService`
    input_history = []
    if history:
        input_history = [
            get_message(role=h.get("role"), message=h.get("content"), turn_id=f"{i}") for (i, h) in enumerate(history)
        ]
    if input_message:
        next_i = f"{len(input_history)}"
        input_message = get_message(message=input_message.get("content"), role=input_message.get("role"), turn_id=next_i)

    # Do something with text and file (e.g., RAG processing)
    conversation_id = request.json.get("session_id")  # TODO: Manage conversation ID in a better way.
    response = controller.get_response(input_message, input_history, conversation_id)
    if file:
        response += f" and received file: {file.filename}"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(
        host=config.deployment_config.api_cfg.host,
        port=config.deployment_config.api_cfg.port,
        debug=config.deployment_config.api_cfg.debug)
