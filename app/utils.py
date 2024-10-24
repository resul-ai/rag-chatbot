import re
from typing import List, Dict, Any
import numpy as np

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        return text.strip()

    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean the text first
        text = TextProcessor.clean_text(text)
        
        # Split into sentences (roughly)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Join the current chunk and add it to chunks
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_size = 0
                overlap_chunk = []
                
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks