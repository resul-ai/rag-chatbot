from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize
import re
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    A text processor that hierarchically splits text into chunks while preserving context.
    Follows a hierarchy of: pages -> paragraphs -> sentences -> token-based splits.
    Ensures no chunk exceeds the specified maximum token limit.
    """
    
    def __init__(self, max_tokens: int = 768):
        """
        Initialize TextProcessor with maximum tokens per chunk.
        Downloads required NLTK data if not already present.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk (default: 768)
        """
        self.max_tokens = max_tokens
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Ensure language-specific resources are available
        try:
            nltk.data.find('tokenizers/punkt/PY3/english.pickle')
        except LookupError:
            logger.info("Downloading English language model...")
            nltk.download('punkt', quiet=True)
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks hierarchically while preserving context.
        
        Args:
            text (str): Input text to be split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
            
        # Basic cleaning (minimal)
        text = self._basic_clean(text)
        
        # Check for page breaks (form feed character)
        if '\f' in text:
            pages = text.split('\f')
            chunks = []
            for page in pages:
                if page.strip():  # Skip empty pages
                    if self._estimate_tokens(page) <= self.max_tokens:
                        chunks.append(page.strip())
                    else:
                        # Split page into smaller chunks
                        chunks.extend(self._split_large_text(page))
            return chunks
            
        return self._split_large_text(text)
    
    def _basic_clean(self, text: str) -> str:
        """
        Perform minimal text cleaning without losing content.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Reduce multiple consecutive newlines to double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace from lines while preserving paragraphs
        lines = [line.strip() for line in text.splitlines()]
        return '\n'.join(line for line in lines)
    
    def _split_large_text(self, text: str) -> List[str]:
        """
        Split large text into smaller chunks by paragraphs first.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # If paragraph alone exceeds limit
            if self._estimate_tokens(para) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split paragraph into sentences
                chunks.extend(self._split_paragraph(para))
            else:
                if current_chunk and self._estimate_tokens(current_chunk + "\n" + para) > self.max_tokens:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n" + para if current_chunk else para)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using blank lines as separators.
        
        Args:
            text (str): Text to split into paragraphs
            
        Returns:
            List[str]: List of paragraphs
        """
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    def _split_paragraph(self, paragraph: str) -> List[str]:
        """
        Split paragraph into smaller chunks by sentences.
        If NLTK sentence tokenization fails, falls back to simple splitting.
        
        Args:
            paragraph (str): Paragraph to split
            
        Returns:
            List[str]: List of chunks
        """
        try:
            sentences = sent_tokenize(paragraph)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {str(e)}")
            # Fallback to simple sentence splitting
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If single sentence exceeds limit
            if self._estimate_tokens(sentence) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split sentence by token limit
                chunks.extend(self._split_by_tokens(sentence))
            else:
                if current_chunk and self._estimate_tokens(current_chunk + " " + sentence) > self.max_tokens:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk = (current_chunk + " " + sentence if current_chunk else sentence)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Split text by estimated token count while trying to preserve word boundaries.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of chunks
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if self._estimate_tokens(current_chunk + " " + word) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = (current_chunk + " " + word if current_chunk else word)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        Uses a simple approximation: ~4 characters per token.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        return len(text.strip()) // 4 + 1  # +1 for safety margin


if __name__ == "__main__":
    # Test examples
    processor = TextProcessor(max_tokens=20)  # Small token limit for testing
    
    print("Initializing text processor and downloading required NLTK data...")
    
    # Test 1: Page-based splitting (PDF-like content)
    print("\nTest 1: Page-based splitting")
    pdf_like_text = "Page 1 content.\nMore content here.\f\nPage 2 content.\nExtra content.\f\nPage 3."
    chunks = processor.split_into_chunks(pdf_like_text)
    print(f"Input length: {len(pdf_like_text)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Test 2: Paragraph-based splitting
    print("\nTest 2: Paragraph-based splitting")
    paragraphs = """First paragraph with enough text to demonstrate splitting.
    
    Second paragraph that contains multiple sentences. This is another sentence.
    
    Third small paragraph."""
    chunks = processor.split_into_chunks(paragraphs)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Test 3: Long sentence splitting
    print("\nTest 3: Long sentence splitting")
    long_sentence = "This is an extremely long sentence that contains many words and should be split into multiple chunks because it exceeds our token limit and needs to be handled properly by the processor while maintaining as much coherence as possible."
    chunks = processor.split_into_chunks(long_sentence)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Test 4: Mixed content
    print("\nTest 4: Mixed content")
    mixed_text = """Short paragraph.

    A longer paragraph with multiple sentences. This should be split carefully.
    
    Final short line."""
    chunks = processor.split_into_chunks(mixed_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Test 5: Edge cases
    print("\nTest 5: Edge cases")
    edge_cases = [
        "",  # Empty string
        "Short text",  # Text below limit
        "a" * 2000,  # Very long string without spaces
        "\n\n\n",  # Only newlines
        "   ",  # Only spaces
    ]
    
    for case in edge_cases:
        print(f"\nInput: '{case}'")
        chunks = processor.split_into_chunks(case)
        print(f"Chunks: {chunks}")