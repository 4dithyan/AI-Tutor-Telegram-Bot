from fastembed import TextEmbedding
from app.config import settings
from app.utils.logging import embed_logger
import time
from typing import List

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model
        embed_logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        start = time.time()
        # The model is loaded lazily on first instantiation. 
        # For a 12GB RAM machine, FastEmbed is very memory efficient.
        self.model = TextEmbedding(model_name=self.model_name)
        embed_logger.info(f"Embedding model loaded in {time.time() - start:.2f}s")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        Returns a list of vectors (list of floats).
        """
        start = time.time()
        # FastEmbed returns a generator of numpy arrays, we convert to list of floats for Qdrant
        embeddings_generator = self.model.embed(texts)
        embeddings = [vec.tolist() for vec in embeddings_generator]
        embed_logger.info(f"Embedded {len(texts)} chunks in {time.time() - start:.2f}s")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        start = time.time()
        embeddings_generator = self.model.query_embed(text)
        embeddings = [vec.tolist() for vec in embeddings_generator]
        embed_logger.info(f"Embedded query in {time.time() - start:.3f}s")
        return embeddings[0] if embeddings else []

# Global singleton instance to reuse the loaded model
embedding_service = EmbeddingService()
