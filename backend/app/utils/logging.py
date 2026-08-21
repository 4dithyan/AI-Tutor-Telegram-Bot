import logging
import sys

def setup_logger(name: str, prefix: str) -> logging.Logger:
    """
    Setup a simple logger with a specific prefix.
    Examples of prefix: [UPLOAD], [OCR/VISION], [CHUNKING], etc.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs if handlers already exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f'%(asctime)s {prefix} %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Pre-configured loggers for different modules
upload_logger = setup_logger("upload", "[UPLOAD]")
vision_logger = setup_logger("vision", "[OCR/VISION]")
chunk_logger = setup_logger("chunking", "[CHUNKING]")
embed_logger = setup_logger("embedding", "[EMBEDDING]")
db_logger = setup_logger("vector_store", "[VECTOR STORE]")
retrieval_logger = setup_logger("retrieval", "[RETRIEVAL]")
llm_logger = setup_logger("llm", "[LLM]")
api_logger = setup_logger("api", "[API]")
