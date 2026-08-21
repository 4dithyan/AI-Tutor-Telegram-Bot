import ollama
from app.config import settings
from app.utils.logging import llm_logger
import time
from typing import Generator

class LLMService:
    def __init__(self):
        self.model = settings.chat_model
        self.client = ollama.Client(host=settings.ollama_base_url)

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        llm_logger.info(f"Generating LLM response using model {self.model}...")
        start = time.time()
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                keep_alive="60m",
                options={
                    "num_predict": 350,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
            answer = response['message']['content']
            llm_logger.info(f"LLM generated response in {time.time() - start:.2f}s")
            return answer
        except Exception as e:
            llm_logger.error(f"LLM generation failed: {str(e)}")
            raise e

    def generate_response_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        llm_logger.info(f"Generating streaming LLM response using model {self.model}...")
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                stream=True,
                keep_alive="60m",
                options={
                    "num_predict": 350,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
            for chunk in response:
                token = chunk.get('message', {}).get('content', '')
                if token:
                    yield token
        except Exception as e:
            llm_logger.error(f"Streaming LLM generation failed: {str(e)}")
            raise e

llm_service = LLMService()


