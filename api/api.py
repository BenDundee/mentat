from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Any, Optional
from api.agents import ExecutiveCoachAgent
import sqlite3

from api.config import Configurator
from api.types import ChatResponse, ChatRequest


app = FastAPI(title="Executive Coach API")


def create_executive_coach_agent(config_path: Optional[str] = None) -> ExecutiveCoachAgent:
    """Create an ExecutiveCoachAgent with the specified configuration."""
    # Load configuration
    config = Configurator.load_config(config_path)

    # Extract LLM configuration
    llm_config = config.get("llm", {})

    # Create agents with LLM configuration
    agent = ExecutiveCoachAgent(llm_config=llm_config)

    return agent


coach_agent = create_executive_coach_agent()


def get_db():
    conn = sqlite3.connect("executive_coach.db")
    try:
        yield conn
    finally:
        conn.close()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = coach_agent.run(
            message=request.message,
            history=request.history,
            user_id=request.user_id
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals", response_model=Goal)
async def create_goal(goal: GoalCreate, conn: sqlite3.Connection = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO goals (user_id, title, description, target_date) 
           VALUES (?, ?, ?, ?) RETURNING *""",
        (goal.user_id, goal.title, goal.description, goal.target_date)
    )
    new_goal = cursor.fetchone()
    conn.commit()

    # Convert row to Goal model
    columns = [col[0] for col in cursor.description]
    goal_dict = {key: value for key, value in zip(columns, new_goal)}

    return Goal(**goal_dict)


@app.get("/goals", response_model=List[Goal])
async def list_goals(user_id: str = "default_user", conn: sqlite3.Connection = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,))
    goals = cursor.fetchall()

    # Convert rows to Goal models
    columns = [col[0] for col in cursor.description]
    result = []
    for goal in goals:
        goal_dict = {key: value for key, value in zip(columns, goal)}
        result.append(Goal(**goal_dict))

    return result


@app.post("/journal")
async def create_journal_entry(entry: JournalEntry, conn: sqlite3.Connection = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO journal_entries (user_id, goal_id, content) VALUES (?, ?, ?)",
        (entry.user_id, entry.goal_id, entry.content)
    )
    conn.commit()

    # Also store in vector DB for semantic search
    coach_agent._store_interaction(
        user_id=entry.user_id,
        user_message="Journal Entry",
        bot_message=entry.content
    )

    return {"status": "success", "message": "Journal entry created"}


@app.get("/journal/prompts")
async def get_journal_prompts(user_id: str = "default_user"):
    prompt = coach_agent._journal_prompt("generate journal prompt", user_id)
    return {"prompt": prompt}