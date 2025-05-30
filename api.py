from fastapi import FastAPI, HTTPException
import logging

from langchain_core.messages import ChatMessage, AIMessage, BaseMessage

from api.interfaces import ChatRequest
from api.api_configurator import APIConfigurator
from api.controller import Controller

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

app = FastAPI(title="Executive Coach API")
config = APIConfigurator()
controller = Controller(config)

@app.post("/chat", response_model=BaseMessage, summary="Chat endpoint for processing chat requests and producing responses")
async def chat_endpoint(request: ChatRequest) -> BaseMessage:
    """
    Handles the chat endpoint for processing chat requests and producing responses.
    The endpoint expects a `ChatRequest` object and returns a LangChain `ChatMessage` object.
    In case of internal errors, an HTTP 500 status code is returned.

    :param request: A `ChatRequest` object containing the message to process,
        optional chat history, and the user ID.
    :type request: ChatRequest

    :return: A LangChain `ChatMessage` object containing the generated response.
    :rtype: ChatMessage

    :raises HTTPException: If an internal server error occurs.
    """
    try:
        out = controller.process_message(
            user_message=request.message,
            history=request.history,
            user_id=request.user_id
        )
        return out.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)