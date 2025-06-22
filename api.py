from flask import Flask, request, jsonify
import logging

from atomic_agents.lib.components.agent_memory import Message

from src.configurator import Configurator
from src.controller import Controller


app = Flask(__name__)


logger = logging.getLogger(__name__)
fmt = '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)

config = Configurator()
controller = Controller(config)

@app.route("/chat", methods=["POST"])
def chat():
    input = request.json.get("message")
    history = request.json.get("history")
    conversation_id = request.json.get("session_id")
    file = request.files.get("file")

    # TODO: move to adapter in `utils` module
    input_message = Message()
    input_history = []
    if history:
        input_history = [Message(content=h.get("content"), role=h.get("role"), turn_id=i) for (i, h) in enumerate(history)]
    if input:
        next_i = len(input_history)
        input_message = Message(content=input.get("content"), role=input.get("role"), turn_id=next_i)

    # Do something with text and file (e.g., RAG processing)
    response = controller.get_response(input_message, input_history, conversation_id)
    if file:
        response += f" and received file: {file.filename}"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(
        host=config.deployment_config.api_cfg.host,
        port=config.deployment_config.api_cfg.port,
        debug=config.deployment_config.api_cfg.debug)
