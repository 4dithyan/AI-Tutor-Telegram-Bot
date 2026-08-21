import ollama
import base64
import io
import time
import os
from PIL import Image
from app.config import settings
from app.utils.logging import vision_logger

VISION_PROMPT = """
You are an expert educational textbook extractor.
Please read this page carefully and extract the educational content accurately.
Preserve:
- Headings
- Paragraphs
- Bullet points
- Formulas
- Definitions
- Examples
- Important concepts
- Tables when possible

Do not hallucinate information that is not visible in the image.
If there is text that is unreadable or uncertain, clearly mark it as [UNREADABLE].
Output the content in clean Markdown format without unnecessary commentary.
"""

def optimize_image_for_vision(file_path: str, max_size: int = 1600) -> str:
    """
    Opens image, resizes if larger than max_size while maintaining aspect ratio,
    and returns base64 string. Reduces inference time dramatically.
    """
    with Image.open(file_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        width, height = img.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class VisionService:
    def __init__(self):
        self.model = settings.vision_model
        # Configure client base url
        self.client = ollama.Client(host=settings.ollama_base_url)

    def extract_text_from_image(self, file_path: str) -> str:
        """
        Optimizes image resolution, base64 encodes, and sends to qwen3-vl via ollama with memory pinning.
        """
        vision_logger.info(f"Extracting text from image: {file_path}")
        start = time.time()
        
        try:
            image_base64 = optimize_image_for_vision(file_path, max_size=1600)
            
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': VISION_PROMPT,
                    'images': [image_base64]
                }],
                keep_alive="60m",
                options={
                    "temperature": 0.1,
                    "num_predict": 2048
                }
            )
            
            extracted_text = response['message']['content']
            vision_logger.info(f"Extraction completed in {time.time() - start:.2f}s. Extracted {len(extracted_text)} chars.")
            return extracted_text
        except Exception as e:
            vision_logger.error(f"Failed to extract text from image: {str(e)}")
            raise e

vision_service = VisionService()

