from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, chat, notes, quiz
from app.utils.logging import api_logger

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("Pre-warming EmbeddingService...")
    try:
        from app.services.embeddings import embedding_service
        embedding_service.embed_query("warmup")
        api_logger.info("EmbeddingService pre-warmed successfully!")
    except Exception as e:
        api_logger.warning(f"Failed to pre-warm EmbeddingService: {e}")
        
    api_logger.info("Starting Telegram Bot...")
    try:
        from app.services.telegram_bot import telegram_bot_service
        await telegram_bot_service.start()
    except Exception as e:
        api_logger.error(f"Failed to start Telegram bot: {e}")

    yield

    try:
        from app.services.telegram_bot import telegram_bot_service
        await telegram_bot_service.stop()
    except Exception as e:
        api_logger.error(f"Error stopping Telegram bot: {e}")

app = FastAPI(
    title="AI Tutor Agent API",
    description="Backend API for processing textbook images and RAG interactions.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for the simple frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a local tool
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(notes.router)
app.include_router(quiz.router)

@app.get("/")
def read_root():
    return {"status": "AI Tutor Agent Backend is running."}

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    api_logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
