import re
from typing import Dict, Any, Tuple, List, Generator, Optional
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.services.llm import llm_service
from app.config import settings
from app.utils.logging import retrieval_logger

RAG_SYSTEM_PROMPT = """
You are an AI Textbook Study Assistant developed by Adithyan (https://adithyan-portfolio.pages.dev). 
Your primary role is to answer questions strictly based on the provided textbook context or educational study topics.

Follow these strict rules:
1. OFF-TOPIC RESTRICTION RULE: If the user asks something completely non-educational (such as sports scores, celebrity gossip, movies, jokes, or non-study chat), output EXACTLY:
   "⚠️ OFF_TOPIC_RESTRICTION"

2. CONTENT VALIDATION RULE:
   - If the answer is present in the provided textbook context, answer clearly based on that context.
   - If the user's question is educational BUT NOT mentioned or covered in the provided textbook context, start your response with:
     "⚠️ <b>Outside PDF/Image Topic Alert</b>\n<i>Note: This question is not covered in your uploaded textbook documents/images. Below is general educational guidance:</i>\n\n"
"""

def process_rag_query(question: str, user_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    retrieval_logger.info(f"Processing RAG query: {question} (user: {user_id or 'all'})")
    
    # 1. Embed the user question
    query_vector = embedding_service.embed_query(question)
    
    # 2. Retrieve relevant chunks for user
    top_chunks = vector_store.search(query_vector=query_vector, top_k=settings.top_k, user_id=user_id)
    
    if not top_chunks:
        retrieval_logger.info("No uploaded textbook chunks found for user.")
        general_prompt = f"Answer the student's question if educational. Question: {question}"
        general_answer = llm_service.generate_response(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=general_prompt).strip()
        
        if "OFF_TOPIC_RESTRICTION" in general_answer:
            return (
                "⚠️ <b>Topic Restriction Alert!</b>\n\n"
                "I am your AI Textbook Tutor developed by <a href=\"https://adithyan-portfolio.pages.dev\">Adithyan</a>. "
                "I am restricted to answering questions related to educational study topics. Please ask a study-related question!",
                []
            )
            
        outside_msg = (
            "⚠️ <b>Outside PDF/Image Topic Alert</b>\n"
            "<i>Note: You have not uploaded any textbook documents/images yet. Below is general educational guidance:</i>\n\n"
            f"{general_answer}\n\n"
            "💡 <i>Tip: Upload a textbook photo or PDF anytime to ask questions specifically from your materials!</i>"
        )
        return outside_msg, []
        
    # 3. Construct context
    context_str = "--- TEXTBOOK CONTEXT ---\n"
    for i, chunk in enumerate(top_chunks):
        context_str += f"Source [{i+1}] (File: {chunk['original_filename']}):\n{chunk['text']}\n\n"
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{question}"
    
    # 4. Generate Answer with content validation
    answer = llm_service.generate_response(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt).strip()
    
    if "OFF_TOPIC_RESTRICTION" in answer:
        return (
            "⚠️ <b>Topic Restriction Alert!</b>\n\n"
            "I am your AI Textbook Tutor developed by <a href=\"https://adithyan-portfolio.pages.dev\">Adithyan</a>. "
            "I am restricted to answering questions related to educational study topics. Please ask a study-related question!",
            []
        )
        
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


def generate_notes_from_docs(document_id: str = None, topic: str = None, user_id: Optional[str] = None) -> str:
    """
    Generates structured, clean bullet-point study notes from uploaded documents.
    """
    if topic:
        query_vector = embedding_service.embed_query(topic)
        chunks = vector_store.search(query_vector, top_k=8, user_id=user_id)
    else:
        query_vector = embedding_service.embed_query("textbook main concepts definitions formulas study notes summary")
        chunks = vector_store.search(query_vector, top_k=8, user_id=user_id)
        
    if not chunks:
        return "📂 No uploaded documents found to generate study notes. Please upload a textbook photo or PDF first!"
        
    context_str = "\n\n".join([f"Source ({c['original_filename']}):\n{c['text']}" for c in chunks])
    
    system_prompt = (
        "You are an expert study guide creator. "
        "Create clear, beautifully structured study notes covering ALL main topics present in the textbook context. "
        "STRICT FORMATTING RULES:\n"
        "1. Do NOT use markdown asterisks (`*` or `**`) anywhere in the text.\n"
        "2. Use clean bullet points starting with `• ` for key points and findings.\n"
        "3. Group information logically under clear section titles.\n"
        "4. Include all key definitions, formulas, and main topic findings."
    )
    
    user_prompt = f"Textbook Context:\n{context_str}\n\nPlease generate structured study notes."
    
    notes = llm_service.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
    
    # Post-process cleanup to strip any raw markdown asterisks
    cleaned_notes = re.sub(r'\*+', '', notes)
    cleaned_notes = re.sub(r'^\s*[-+]\s+', '• ', cleaned_notes, flags=re.MULTILINE)
    return cleaned_notes.strip()
