from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.quiz_service import classify_topics, generate_quiz
from app.utils.logging import api_logger

router = APIRouter(prefix="/quiz", tags=["quiz"])

class QuizRequest(BaseModel):
    topic: Optional[str] = None
    num_questions: Optional[int] = 5
    user_id: Optional[str] = None

@router.get("/topics")
async def get_topics(user_id: Optional[str] = None):
    try:
        topics = classify_topics(user_id=user_id)
        return {"topics": topics}
    except Exception as e:
        api_logger.error(f"Failed to fetch topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_quiz_endpoint(request: QuizRequest):
    try:
        quiz_data = generate_quiz(topic=request.topic, num_questions=request.num_questions or 5, user_id=request.user_id)
        return quiz_data
    except Exception as e:
        api_logger.error(f"Failed to generate quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))
