from fastapi import FastAPI, HTTPException, Depends
import logging

from api.interfaces import ChatResponse, ChatRequest
from api.api_configurator import APIConfigurator
from api.workflows import WorkflowOrchestrator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

app = FastAPI(title="Executive Coach API")
config = APIConfigurator()
orchestrator = WorkflowOrchestrator(llm_provider=config.llm_provider)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Handles the chat endpoint for processing chat requests and producing responses.
    The endpoint expects a `ChatRequest` object and returns a `ChatResponse` object.
    In case of internal errors, an HTTP 500 status code is returned.

    :param request: A `ChatRequest` object containing the message to process,
        optional chat history, and the user ID.
    :type request: ChatRequest

    :return: A `ChatResponse` object containing the generated response.
    :rtype: ChatResponse

    :raises HTTPException: If an internal server error occurs.
    """
    try:
        out = orchestrator.process_message(user_message=request.message, user_id=request.user_id)
        return ChatResponse(response=out.get("output", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
