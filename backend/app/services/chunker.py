from app.config import settings
from app.utils.logging import chunk_logger
from typing import List
import re

def semantic_chunk_text(text: str) -> List[str]:
    """
    Chunks text semantically by splitting into paragraphs/sections 
    and combining them until CHUNK_SIZE is reached.
    """
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    
    if not text:
        return []

    # Split by double newlines (paragraphs/headers)
    paragraphs = re.split(r'\n{2,}', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding the paragraph exceeds the chunk size, we save the current chunk
        # and start a new one, keeping a bit of overlap if possible.
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Simple overlap: take the last `chunk_overlap` characters of the previous chunk 
            # to start the new chunk to maintain context.
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    chunk_logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks
