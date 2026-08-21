from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.services.llm import llm_service
from app.config import settings
from app.utils.logging import retrieval_logger
from typing import Dict, Any, Tuple, List, Generator

from typing import Dict, Any, Tuple, List, Generator, Optional

RAG_SYSTEM_PROMPT = """
You are an AI Textbook Study Assistant developed by Adithyan (https://adithyan-portfolio.pages.dev). 
Your primary role is to answer questions strictly based on the provided textbook context or educational study topics.

Follow these strict rules:
1. Answer primarily from the retrieved textbook context in a simple, beginner-friendly language.
2. Use clear explanations, formulas, and examples when helpful.
3. RESTRICTION RULE: If the user asks something completely off-topic (such as sports scores, gossip, jokes, or non-educational topics), respond politely:
   "⚠️ I am your AI Textbook Tutor developed by Adithyan (https://adithyan-portfolio.pages.dev). I am restricted to answering questions related to your uploaded textbook materials and educational study topics. Please ask a study-related question!"
4. If the answer is not available in the uploaded material but is an educational question, answer based on general educational knowledge while noting it is outside the uploaded textbook.
"""

def process_rag_query(question: str, user_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    retrieval_logger.info(f"Processing RAG query: {question} (user: {user_id or 'all'})")
    
    # 1. Embed the user question
    query_vector = embedding_service.embed_query(question)
    
    # 2. Retrieve relevant chunks for user
    top_chunks = vector_store.search(query_vector=query_vector, top_k=settings.top_k, user_id=user_id)
    
    if not top_chunks:
        retrieval_logger.warning("No relevant chunks found in the database.")
        return "I don't have any uploaded textbook material to answer this question. Please upload some images or PDFs first.", []
        
    # 3. Construct context
    context_str = "--- TEXTBOOK CONTEXT ---\n"
    for i, chunk in enumerate(top_chunks):
        context_str += f"Source [{i+1}] (File: {chunk['original_filename']}):\n{chunk['text']}\n\n"
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{question}"
    
    # 4. Generate Answer
    answer = llm_service.generate_response(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt)
    
    return answer, top_chunks

def process_rag_query_stream(question: str, user_id: Optional[str] = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    retrieval_logger.info(f"Processing RAG streaming query: {question} (user: {user_id or 'all'})")
    
    # 1. Embed the user question
    query_vector = embedding_service.embed_query(question)
    
    # 2. Retrieve relevant chunks for user
    top_chunks = vector_store.search(query_vector=query_vector, top_k=settings.top_k, user_id=user_id)
    
    if not top_chunks:
        def empty_gen():
            yield "I don't have any uploaded textbook material to answer this question. Please upload some images or PDFs first."
        return empty_gen(), []
        
    # 3. Construct context
    context_str = "--- TEXTBOOK CONTEXT ---\n"
    for i, chunk in enumerate(top_chunks):
        context_str += f"Source [{i+1}] (File: {chunk['original_filename']}):\n{chunk['text']}\n\n"
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{question}"
    
    # 4. Stream generator
    stream_gen = llm_service.generate_response_stream(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt)
    return stream_gen, top_chunks


def generate_notes_from_docs(document_id: str = None, topic: str = None) -> str:
    # A simple implementation for notes generation.
    # In a real scenario, you'd fetch all chunks for a document or specific chunks matching a topic.
    # For now, if topic is provided, search it. If document_id is provided, you might want to fetch those chunks directly.
    # Let's keep it simple: if there's a topic, search and summarize.
    
    if topic:
        query_vector = embedding_service.embed_query(topic)
        chunks = vector_store.search(query_vector, top_k=5)
    else:
        # Just grab random/first few chunks (requires different Qdrant query, simplified here)
        # Using a dummy query to just get some content
        query_vector = embedding_service.embed_query("textbook notes summary")
        chunks = vector_store.search(query_vector, top_k=5)
        
    if not chunks:
        return "No documents found to generate notes."
        
    context_str = "\n".join([c['text'] for c in chunks])
    
    prompt = """
    Create concise, structured study notes based on the following textbook content.
    Include key definitions, formulas, and bullet points.
    """
    
    user_prompt = f"Content:\n{context_str}\n\nPlease generate notes."
    return llm_service.generate_response(system_prompt=prompt, user_prompt=user_prompt)
