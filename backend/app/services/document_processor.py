import re
import os
import tempfile
import pymupdf as fitz
from pypdf import PdfReader
from app.services.vision import vision_service
from app.utils.logging import upload_logger

def clean_text(text: str) -> str:
    """
    Cleans the raw text extracted from vision model or PDF parser.
    """
    if not text:
        return ""
    
    # Remove large amounts of whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    Attempts direct text extraction per page using PyPDF.
    If a page has minimal or no text (< 50 chars), renders the page as an image
    and passes it to vision_service for OCR extraction.
    """
    upload_logger.info(f"Extracting text from PDF: {file_path}")
    doc_pages_text = []
    
    try:
        pdf_reader = PdfReader(file_path)
        total_pages = len(pdf_reader.pages)
        upload_logger.info(f"PDF has {total_pages} pages.")
        
        fitz_doc = fitz.open(file_path)
        
        for i, page in enumerate(pdf_reader.pages):
            page_num = i + 1
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            
            # Fallback to Vision OCR if text is missing or sparse (scanned page)
            if len(extracted) < 50:
                upload_logger.info(f"Page {page_num} has minimal text ({len(extracted)} chars). Running Vision OCR fallback...")
                try:
                    fitz_page = fitz_doc[i]
                    pix = fitz_page.get_pixmap(dpi=150)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                        tmp_path = tmp_img.name
                        pix.save(tmp_path)
                    
                    ocr_text = vision_service.extract_text_from_image(tmp_path)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
                    if ocr_text.strip():
                        extracted = ocr_text.strip()
                except Exception as ocr_err:
                    upload_logger.error(f"Vision OCR fallback failed for page {page_num}: {ocr_err}")
            
            if extracted:
                doc_pages_text.append(f"--- Page {page_num} ---\n{extracted}")
        
        fitz_doc.close()
        full_text = "\n\n".join(doc_pages_text)
        return clean_text(full_text)
    except Exception as e:
        upload_logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        raise e
