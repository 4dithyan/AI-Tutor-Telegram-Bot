import json
import re
from typing import List, Dict, Any, Optional
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.services.llm import llm_service
from app.utils.logging import api_logger

def classify_topics(user_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Analyzes the indexed textbook chunks in Qdrant for a specific user and classifies them into main topics.
    """
    api_logger.info(f"Classifying textbook topics for user: {user_id or 'all'}...")
    docs = vector_store.list_documents(user_id=user_id)
    if not docs:
        return []

    # Get sample chunks across documents for user
    sample_query_vec = embedding_service.embed_query("textbook main concepts chapters topics summary study")
    top_chunks = vector_store.search(sample_query_vec, top_k=8, user_id=user_id)
    
    if not top_chunks:
        return []

    context_str = "\n\n".join([f"Document ({c['original_filename']}):\n{c['text'][:350]}" for c in top_chunks])

    system_prompt = (
        "You are an educational curriculum organizer. "
        "Analyze the provided textbook context and classify it into 3 to 6 key educational topics/chapters directly present in the text. "
        "Return ONLY a JSON array of strings containing the topic titles. "
        "Example format: [\"1. Topic A\", \"2. Topic B\", \"3. Topic C\"]"
    )
    
    user_prompt = f"Textbook Context:\n{context_str}\n\nPlease output the JSON array of topics."

    try:
        raw_response = llm_service.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
        # Extract JSON array
        match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if match:
            topics_array = json.loads(match.group(0))
            return [{"id": f"topic-{i+1}", "title": str(title).strip()} for i, title in enumerate(topics_array)]
    except Exception as e:
        api_logger.error(f"Topic classification failed: {e}")

    # Fallback default topics if JSON parsing fails
    return [
        {"id": "topic-1", "title": "Overall Chapter Concepts"},
        {"id": "topic-2", "title": "Core Definitions & Formulas"},
        {"id": "topic-3", "title": "Key Principles & Examples"}
    ]

def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()

    # 1. Direct parse
    try:
        res = json.loads(text)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # 2. JSONDecoder raw_decode starting from first '{' (ignores trailing extra data)
    start_idx = text.find('{')
    if start_idx != -1:
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3. Substring from first '{' to last '}'
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            res = json.loads(text[start_idx:end_idx+1])
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    return None

def synthesize_fallback_questions(top_chunks: List[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
    """
    Guaranteed fallback MCQ synthesizer directly from context chunks if LLM JSON output fails.
    """
    questions = []
    sentences = []
    for c in top_chunks:
        text = c.get("text", "")
        filename = c.get("original_filename", "textbook")
        for s in re.split(r'[.\n]', text):
            s_clean = s.strip()
            if len(s_clean) > 25:
                sentences.append((s_clean, filename))
                
    if not sentences:
        return []
        
    target = max(1, min(len(sentences), count))
    for i in range(target):
        sent, fname = sentences[i]
        words = [w for w in sent.split() if len(w) > 4]
        key_term = words[0] if words else "the concept"
        
        q_text = f"According to your uploaded document ({fname}), which statement is correct regarding {key_term}?"
        
        opts = [
            f"A) {sent[:80]}...",
            f"B) An unrelated assumption about {key_term}",
            f"C) The inverse application of {key_term}",
            f"D) None of the above"
        ]
        
        questions.append({
            "id": i + 1,
            "question": q_text,
            "options": opts,
            "correct": "A",
            "explanation": f"Stated directly in {fname}: '{sent}'"
        })
        
    return questions

def generate_quiz(topic: str = None, num_questions: int = 5, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a structured multiple-choice quiz (MCQ) for a specific topic or overall content using user's uploaded documents.
    Adaptively sizes question count based on uploaded content density.
    """
    api_logger.info(f"Generating quiz for topic: {topic or 'Overall Textbook'} (user: {user_id or 'all'}, count: {num_questions})")
    
    if topic and topic.strip().lower() != "overall":
        query_str = f"{topic} definitions key facts concept principles formulas terms"
    else:
        query_str = "textbook main concepts key definitions formulas principles terms summary"

    # Fetch more chunks for 10/15 question quizzes
    fetch_k = max(10, num_questions * 2)
    query_vec = embedding_service.embed_query(query_str)
    top_chunks = vector_store.search(query_vec, top_k=fetch_k, user_id=user_id)
    
    if not top_chunks:
        return {
            "topic": topic or "Overall Textbook",
            "questions": []
        }
        
    # Calculate available content density and adapt question count
    total_chars = sum(len(c.get("text", "")) for c in top_chunks)
    if total_chars < 600:
        target_count = min(num_questions, 3)
    elif total_chars < 1500:
        target_count = min(num_questions, 5)
    else:
        target_count = num_questions
        
    context_str = "\n\n".join([f"Source Document ({c['original_filename']}):\n{c['text']}" for c in top_chunks])
    
    system_prompt = f"""
You are a precise educational exam creator. 
STRICT MANDATE: Generate up to {target_count} Multiple Choice Questions (MCQs) BASED STRICTLY AND ONLY ON THE PROVIDED TEXTBOOK CONTEXT BELOW. 
Do NOT ask generic questions or questions about topics outside the provided context.
Every question MUST have EXPLICITLY 4 options: A), B), C), and D).

Output ONLY valid JSON with no markdown formatting, using this exact schema:
{{
  "topic": "{topic or 'Overall Textbook'}",
  "questions": [
    {{
      "id": 1,
      "question": "Question text directly testing the provided textbook context?",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct": "B",
      "explanation": "Exact explanation based on the textbook context."
    }}
  ]
}}
"""

    user_prompt = f"PROVIDED TEXTBOOK CONTEXT:\n{context_str}\n\nPlease generate the {target_count} MCQs in valid JSON format based ONLY on the context above."
    
    questions = []
    try:
        raw_response = llm_service.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
        quiz_data = extract_json_payload(raw_response)
        
        if quiz_data:
            raw_qs = quiz_data.get("questions") or quiz_data.get("quiz") or []
            
            option_keys = ["A", "B", "C", "D"]
            for idx, q in enumerate(raw_qs, 1):
                q_text = q.get("question") or f"Question {idx}"
                raw_opts = q.get("options") or []
                
                # Normalize options to exactly 4 options A, B, C, D
                norm_opts = []
                for i in range(4):
                    if i < len(raw_opts):
                        opt_str = str(raw_opts[i]).strip()
                        if not (opt_str.startswith("A)") or opt_str.startswith("B)") or opt_str.startswith("C)") or opt_str.startswith("D)")):
                            opt_str = f"{option_keys[i]}) {opt_str}"
                    else:
                        opt_str = f"{option_keys[i]}) Option {i+1}"
                    norm_opts.append(opt_str)
                    
                # Normalize correct answer key
                raw_corr = str(q.get("correct") or q.get("answer") or q.get("correct_answer") or "A").strip().upper()
                corr_key = "A"
                for k in option_keys:
                    if k in raw_corr:
                        corr_key = k
                        break
                        
                questions.append({
                    "id": idx,
                    "question": q_text,
                    "options": norm_opts,
                    "correct": corr_key,
                    "explanation": q.get("explanation") or "Extracted directly from textbook content."
                })
    except Exception as e:
        api_logger.error(f"Quiz LLM generation failed: {e}")

    # Fallback to direct sentence synthesizer if LLM generated 0 questions
    if not questions:
        api_logger.info("Using fallback MCQ synthesizer directly from context chunks...")
        questions = synthesize_fallback_questions(top_chunks, target_count)

    return {
        "topic": topic or "Overall Textbook",
        "questions": questions
    }
