from flask import Flask, request, jsonify
import logging

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
    input = request.json.get("messages")
    file = request.files.get("file")

    # Do something with text and file (e.g., RAG processing)
    response = controller.get_response(input)
    if file:
        response += f" and received file: {file.filename}"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(
        host=config.deployment_config.api_cfg.host,
        port=config.deployment_config.api_cfg.port,
        debug=config.deployment_config.api_cfg.debug)
