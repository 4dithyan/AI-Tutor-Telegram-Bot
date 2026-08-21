from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.rag import process_rag_query, process_rag_query_stream
from app.utils.logging import api_logger

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    api_logger.info(f"Received chat question: {request.question}")
    
    try:
        answer, sources_data = process_rag_query(request.question, user_id=request.user_id)
        
        sources = []
        for s in sources_data:
            sources.append(SourceChunk(
                text=s["text"],
                document_id=s["document_id"],
                original_filename=s["original_filename"],
                score=s["score"]
            ))
            
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        api_logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    api_logger.info(f"Received streaming chat question: {request.question} (user: {request.user_id})")
    
    try:
        stream_gen, sources_data = process_rag_query_stream(request.question, user_id=request.user_id)
        
        sources = []
        for s in sources_data:
            sources.append({
                "original_filename": s["original_filename"],
                "document_id": s["document_id"],
                "score": s.get("score", 0.0)
            })
            
        def event_generator():
            # 1. Send sources metadata first
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # 2. Yield LLM tokens as they generate
            for token in stream_gen:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                
            # 3. Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        api_logger.error(f"Chat stream failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

