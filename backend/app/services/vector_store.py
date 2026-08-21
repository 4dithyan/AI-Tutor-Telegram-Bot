from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.utils.logging import db_logger
import uuid
import time
from typing import List, Dict, Any, Optional

COLLECTION_NAME = "textbook_chunks"
# FastEmbed BAAI/bge-small-en-v1.5 has dimension 384
EMBEDDING_DIM = 384 

class VectorStore:
    def __init__(self):
        db_path = settings.get_qdrant_abs_path()
        db_logger.info(f"Initializing Qdrant local store at: {db_path}")
        self.client = QdrantClient(path=db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if COLLECTION_NAME not in collection_names:
                db_logger.info(f"Creating collection '{COLLECTION_NAME}' with dimension {EMBEDDING_DIM}")
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
            else:
                db_logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception as e:
            db_logger.error(f"Failed to ensure collection exists: {str(e)}")

    def insert_chunks(self, document_id: str, original_filename: str, chunks: List[str], embeddings: List[List[float]], user_id: str = "default_user"):
        if not chunks:
            db_logger.warning("No chunks to insert.")
            return

        db_logger.info(f"Inserting {len(chunks)} chunks into vector store for doc {document_id} (user: {user_id})")
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "document_id": document_id,
                "original_filename": original_filename,
                "text": chunk,
                "chunk_index": i,
                "user_id": user_id
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            
        start = time.time()
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        db_logger.info(f"Insertion completed in {time.time() - start:.2f}s")

    def search(self, query_vector: List[float], top_k: int = 3, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        db_logger.info(f"Searching for top {top_k} matching chunks (user: {user_id or 'all'}).")
        start = time.time()
        
        query_filter = None
        if user_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            )
            
        search_result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        db_logger.info(f"Search completed in {time.time() - start:.2f}s")
        
        results = []
        for hit in search_result.points:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "document_id": hit.payload.get("document_id", ""),
                "original_filename": hit.payload.get("original_filename", ""),
                "user_id": hit.payload.get("user_id", "default_user")
            })
        return results

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, str]]:
        try:
            scroll_filter = None
            if user_id:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        )
                    ]
                )
                
            records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            docs = {}
            for record in records:
                doc_id = record.payload.get("document_id")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": record.payload.get("original_filename", "unknown"),
                        "user_id": record.payload.get("user_id", "default_user")
                    }
            return list(docs.values())
        except Exception as e:
            db_logger.error(f"Error listing documents: {str(e)}")
            return []

    def delete_document(self, document_id: str):
        db_logger.info(f"Deleting document {document_id}")
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )

    def clear_user_documents(self, user_id: str):
        db_logger.info(f"Clearing documents for user {user_id}...")
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        )
                    ]
                ),
            )
            db_logger.info(f"Cleared all documents for user {user_id}")
        except Exception as e:
            db_logger.error(f"Error clearing documents for user {user_id}: {e}")
            raise e

    def clear_all(self):
        db_logger.info("Clearing all collections and documents from Qdrant vector store...")
        try:
            self.client.delete_collection(collection_name=COLLECTION_NAME)
            self._ensure_collection()
            db_logger.info("Vector store collection cleared and re-created successfully!")
        except Exception as e:
            db_logger.error(f"Error clearing vector store: {str(e)}")
            raise e

# Global singleton
from qdrant_client.http import models
vector_store = VectorStore()
