from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ProcessResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class SourceChunk(BaseModel):
    text: str
    document_id: str
    original_filename: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_time: str
    status: str

class NoteRequest(BaseModel):
    document_id: Optional[str] = None
    topic: Optional[str] = None

class NoteResponse(BaseModel):
    notes: str
    source_documents: List[str]
