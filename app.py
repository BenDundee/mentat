import gradio as gr
import requests # For making HTTP requests to your API
import logging

# Configure logging for the Gradio app
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

# Define the URL for your FastAPI backend
# This should match the host and port where your api.py is running
API_URL = "http://127.0.0.1:8000/chat"

def call_chat_api(user_message: str, chat_history: list[list[str | None]] | None) -> gr.ChatMessage:
    """
    Calls the backend chat API to get a response.

    :param user_message: The latest message from the user.
    :param chat_history: The existing chat history in the format [user_msg, bot_msg], ...
    :return: The bot's response as a string.
    """
    payload = {
        "message": user_message,
        "history": chat_history if chat_history else [] # API expects history or empty list
    }
    logger.info(f"Sending payload to API: {payload}")
    err = None
    api_response = None
    try:
        response = requests.post(API_URL, json=payload, timeout=60) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        api_response = response.json()
    except requests.exceptions.Timeout:
        logger.error(f"API call timed out: {API_URL}")
        err = "Error: The request to the API timed out."
    except requests.exceptions.HTTPError as e:
        logger.error(f"API call failed with HTTPError: {e.response.status_code} - {e.response.text}")
        err = f"Error: API request failed (HTTP {e.response.status_code}). Please check API server logs."
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed with RequestException: {e}")
        err = f"Error: Could not connect to the API. Ensure the API server is running at {API_URL}."
    except Exception as e:
        logger.error(f"An unexpected error occurred while calling the API: {e}")
        err = "Error: An unexpected error occurred while processing your request."

    if err:
        return gr.ChatMessage(content=err, role="system")
    else:
        return gr.ChatMessage(content=api_response.get("content", ""), role=api_response.get("role", "assistant"))

def chat_interface_fn(user_message: str, chat_history: list[gr.ChatMessage] | None) -> gr.ChatMessage:
    """
    This function is called by the Gradio ChatInterface for each new message.
    It updates the chat history with the user's message and the bot's response.

    :param user_message: The new message input by the user.
    :param chat_history: The current chat history. Gradio provides this as a list of
                         [user_msg, bot_msg] pairs.
    :return: The updated chat_history, which Gradio uses to update the Chatbot display.
    """
    if chat_history is None:
        chat_history = []

    logger.info(f"User message: {user_message}")
    logger.info(f"Current chat history: {chat_history}")

    # Get the bot's response from the API
    # The `chat_history` from Gradio is already in the List[List[str]] format
    # that our API's ChatRequest model expects.
    bot_response = call_chat_api(user_message, chat_history)
    #chat_history.extend([gr.ChatMessage(content=user_message, role="user"), bot_response])
    logger.info(f"Updated chat history: {chat_history}")
    return bot_response

# Create the Gradio ChatInterface
iface = gr.ChatInterface(
    fn=chat_interface_fn,
    title="Executive Coach AI",
    description="Chat with the AI-powered Executive Coach. Type your message below and press Enter.",
    chatbot=gr.Chatbot(
        height=600,
        label="Chat Session",
        type="messages",
        show_label=True,
        bubble_full_width=False,
        avatar_images=(None, "https://gravatar.com/avatar/99999999999999999999999999999999?d=mp&s=40") # User, Bot (example avatar)
    ),
    textbox=gr.Textbox(
        placeholder="Type your message here...",
        show_label=False,
        container=False,
        scale=7
    ),
)

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    logger.info(f"Make sure your API server (api.py) is running and accessible at {API_URL}")
    # To make it accessible on the network, you might use server_name="0.0.0.0"
    # For development, localhost is usually fine.
    iface.launch()