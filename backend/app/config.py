import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Construct path to .env file relative to this file
env_path = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    vision_model: str = "qwen3-vl:4b"
    chat_model: str = "qwen2.5:1.5b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    telegram_bot_token: str = ""
    max_upload_size_mb: int = 30
    
    # Resolving absolute path for qdrant to be relative to the backend folder
    qdrant_path: str = "./data/qdrant_db"
    
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    host: str = "127.0.0.1"
    port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=env_path if env_path.exists() else ".env", 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

    def get_qdrant_abs_path(self):
        backend_dir = Path(__file__).parent.parent
        return str(backend_dir / self.qdrant_path)

settings = Settings()
