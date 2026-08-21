from fastapi import APIRouter, HTTPException
from app.models.schemas import NoteRequest, NoteResponse
from app.services.rag import generate_notes_from_docs
from app.utils.logging import api_logger

router = APIRouter(prefix="/notes", tags=["notes"])

@router.post("", response_model=NoteResponse)
async def generate_notes(request: NoteRequest):
    api_logger.info("Generating notes...")
    try:
        notes = generate_notes_from_docs(request.document_id, request.topic)
        return NoteResponse(notes=notes, source_documents=[request.document_id] if request.document_id else [])
    except Exception as e:
        api_logger.error(f"Notes generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
