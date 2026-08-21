from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
import os
import shutil
import uuid

from app.models.schemas import ProcessResponse, DocumentInfo
from app.services.vision import vision_service
from app.services.document_processor import clean_text, extract_text_from_pdf
from app.services.chunker import semantic_chunk_text
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.utils.logging import upload_logger

from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    upload_logger.info(f"Received {len(files)} files for upload.")
    saved_files = []
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    
    for file in files:
        filename_lower = file.filename.lower()
        is_image = file.content_type.startswith("image/") or filename_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'))
        is_pdf = file.content_type == "application/pdf" or filename_lower.endswith(".pdf")
        
        if not (is_image or is_pdf):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a supported image or PDF.")
            
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        size = 0
        with open(file_path, "wb") as buffer:
            while content_chunk := await file.read(1024 * 1024):
                size += len(content_chunk)
                if size > max_bytes:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds maximum allowed size of {settings.max_upload_size_mb}MB.")
                buffer.write(content_chunk)
                
        saved_files.append({"filename": file.filename, "path": file_path, "size_mb": round(size / (1024 * 1024), 2)})
        
    return {"message": f"Successfully uploaded {len(saved_files)} files.", "files": saved_files}

@router.post("/process", response_model=ProcessResponse)
async def process_document(filename: str, user_id: Optional[str] = "web_user"):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found in uploads directory.")
        
    document_id = str(uuid.uuid4())
    upload_logger.info(f"Starting processing for {filename} (Doc ID: {document_id}, User: {user_id})")
    
    try:
        # 1. Text Extraction (PDF vs Image)
        if filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = vision_service.extract_text_from_image(file_path)
            
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from document.")
            
        # 2. Text Cleaning
        cleaned_text = clean_text(raw_text)
        
        # 3. Chunking
        chunks = semantic_chunk_text(cleaned_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create chunks from text.")
            
        # 4. Embeddings
        embeddings = embedding_service.embed_texts(chunks)
        
        # 5. Vector Store Insertion with user_id
        vector_store.insert_chunks(
            document_id=document_id,
            original_filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            user_id=user_id or "web_user"
        )
        
        return ProcessResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
            message="Processing successful"
        )
    except Exception as e:
        upload_logger.error(f"Processing failed for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("", response_model=List[DocumentInfo])
async def list_documents(user_id: Optional[str] = None):
    docs = vector_store.list_documents(user_id=user_id)
    result = []
    for d in docs:
        result.append(DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            upload_time="Unknown",
            status="Processed"
        ))
    return result

@router.delete("/clear/all")
async def clear_all_documents(user_id: Optional[str] = None):
    try:
        if user_id:
            vector_store.clear_user_documents(user_id=user_id)
            return {"message": f"All documents for user '{user_id}' cleared successfully!"}
        else:
            vector_store.clear_all()
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
            return {"message": "All documents cleared successfully!"}
    except Exception as e:
        upload_logger.error(f"Failed to clear documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    try:
        vector_store.delete_document(document_id)
        return {"message": f"Document {document_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
