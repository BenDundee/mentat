import gradio as gr
import requests
import logging

from src.configurator import Configurator


logger = logging.getLogger(__name__)
fmt = '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)


config = Configurator()


def chat_with_backend(user_input, message_history):
    # Add current user message to the history
    message_history.append({"role": "user", "content": user_input})

    try:
        # Send full history to backend
        response = requests.post(config.deployment_config.api, json={"messages": message_history})
        assistant_reply = response.json().get("response", "[No response]")
    except Exception as e:
        assistant_reply = f"[Error] {str(e)}"

    # Append assistant's reply to the history
    message_history.append({"role": "assistant", "content": assistant_reply})

    # Return formatted messages for Chatbot display
    chatbot_display = [
        (m["content"], message_history[i + 1]["content"])
        for i, m in enumerate(message_history[:-1])
        if m["role"] == "user"
    ]

    return chatbot_display, message_history


def clear_chat():
    return [], []  # Empty display and message history


if __name__ == "__main__":

    with gr.Blocks() as demo:
        gr.Markdown("## Executive Coach Assistant")

        chatbot = gr.Chatbot()
        with gr.Column(scale=5):  # Make textbox take more space
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Ask your coaching assistant..."
            )
        with gr.Column(scale=1):  # Keep buttons narrow
            submit = gr.Button("Send")
            clear = gr.Button("Clear Conversation")

        message_history = gr.State([])  # List of {"role": ..., "content": ...}

        submit.click(chat_with_backend, inputs=[msg, message_history], outputs=[chatbot, message_history])
        msg.submit(chat_with_backend, inputs=[msg, message_history], outputs=[chatbot, message_history])

        # Clear button logic
        clear.click(fn=clear_chat, inputs=[], outputs=[chatbot, message_history])

    demo.launch(
        server_name=config.deployment_config.app_cfg.host,
        server_port=config.deployment_config.app_cfg.port,
        debug=config.deployment_config.app_cfg.debug
    )

