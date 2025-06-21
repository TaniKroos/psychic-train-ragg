# document_processor.py

import fitz  # PyMuPDF
from typing import List
from io import BytesIO

def extract_text_from_pdf_file(file_bytes: BytesIO) -> str:
    """
    Extracts all text from a PDF file uploaded via FastAPI (BytesIO).
    """
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text.strip()

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Splits text into chunks of approximately `max_tokens` words.
    """
    import re

    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count <= max_tokens:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
