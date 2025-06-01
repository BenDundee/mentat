from fastapi import FastAPI, HTTPException
import logging

from langchain_core.messages import BaseMessage

from api.interfaces import ChatRequest, ChatResponse
from api.api_configurator import APIConfigurator
from api.controller import Controller

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

app = FastAPI(title="Executive Coach API")
config = APIConfigurator()
controller = Controller(config)

@app.post(
    "/chat",
    response_model=BaseMessage,
    summary="Chat endpoint for processing chat requests and producing responses"
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse: #BaseMessage:
    """
    Chat endpoint for processing user chat requests and generating responses.

    This asynchronous method interfaces with the processing controller to deal
    with incoming user messages, maintain user conversation history, and
    generate appropriate responses. It also manages error handling by returning
    HTTPExceptions in case of unexpected issues during processing.

    Args:
        request (ChatRequest): Input request containing the user message,
        conversation history, and user ID.

    Returns:
        BaseMessage: The computed response message object.

    Raises:
        HTTPException: If an error occurs during processing, this exception
        is raised with a 500 status code and a detailed error message.
    """
    try:
        state = controller.process_message(
            user_message=request.message,
            history=request.history or [],
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return ChatResponse(
            message=state.response.content if state.response else "Sorry, I couldn't process your request.",
            session_id=state.conversation_id or request.session_id,  # Return the session ID for continuity
            detected_intent=state.detected_intent.value if state.detected_intent else None,
            success=not state.errors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)