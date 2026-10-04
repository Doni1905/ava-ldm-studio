"""
src/asr/postprocess.py
=======================
Text normalization and post-processing for ASR outputs.
"""

import re

def normalize_text(text: str) -> str:
    """
    Basic text normalization for Tamil/general ASR output.
    - Remove extra whitespaces
    - Strip leading/trailing whitespaces
    - (Optional future: remove specific punctuation or apply Tamil-specific normalizations)
    """
    if not text:
        return ""
        
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

