from typing import List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    A text processor that splits text into chunks while preserving sentence integrity.
    Uses NLTK for sentence and word tokenization.
    """
    
    def __init__(self, max_tokens: int = 768):
        """
        Initialize TextProcessor with maximum tokens per chunk.
        Downloads required NLTK data if not present.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk
        """
        self.max_tokens = max_tokens
        
        # Download all required NLTK data
        requirements = [
            'punkt',
            'averaged_perceptron_tagger',
            'punkt_tab',
            'english.pickle'
        ]
        
        for requirement in requirements:
            try:
                # Special handling for english.pickle
                if requirement == 'english.pickle':
                    try:
                        nltk.data.find('tokenizers/punkt/english.pickle')
                    except LookupError:
                        nltk.download('punkt')
                else:
                    try:
                        nltk.data.find(f'tokenizers/{requirement}')
                    except LookupError:
                        nltk.download(requirement, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {requirement}: {str(e)}")
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving sentence integrity.
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
            
        text = self._basic_clean(text)
        
        # Handle page breaks first
        if '\f' in text:
            pages = text.split('\f')
            chunks = []
            for page in pages:
                if page.strip():
                    chunks.extend(self._process_text(page))
            return chunks
        
        return self._process_text(text)
    
    def _basic_clean(self, text: str) -> str:
        """
        Perform basic text cleaning.
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _process_text(self, text: str) -> List[str]:
        """
        Process text by splitting into paragraphs first, then sentences.
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if not paragraphs:
            return []
            
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            # Get sentences in paragraph
            try:
                sentences = sent_tokenize(paragraph)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {str(e)}")
                # Fallback to simple sentence splitting
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
            
            paragraph_processed = False
            
            for sentence in sentences:
                # Count tokens in sentence
                try:
                    sentence_tokens = word_tokenize(sentence)
                except Exception as e:
                    logger.warning(f"NLTK word tokenization failed: {str(e)}")
                    sentence_tokens = sentence.split()
                
                sentence_token_count = len(sentence_tokens)
                
                # If single sentence exceeds max tokens
                if sentence_token_count > self.max_tokens:
                    # Save current chunk if exists
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    
                    # Add long sentence as its own chunk
                    chunks.append(sentence)
                    paragraph_processed = True
                    continue
                
                # If adding this sentence would exceed the limit
                if current_tokens + sentence_token_count > self.max_tokens:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    # Start new chunk with current sentence
                    current_chunk = [sentence]
                    current_tokens = sentence_token_count
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_tokens += sentence_token_count
                
                paragraph_processed = True
            
            # Add paragraph break if not at the end
            if paragraph_processed and current_chunk and paragraphs.index(paragraph) < len(paragraphs) - 1:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
        
        # Add final chunk if exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    processor = TextProcessor(max_tokens=50)  # Small token limit for testing
    
    print("Running text processor tests...\n")
    
    def test_and_print(name: str, text: str):
        print(f"\n=== Test: {name} ===")
        try:
            token_count = len(word_tokenize(text)) if text.strip() else 0
            print(f"Input text ({token_count} tokens):")
        except Exception:
            print(f"Input text (token count unavailable):")
        print(text[:100] + "..." if len(text) > 100 else text)
        
        chunks = processor.split_into_chunks(text)
        print(f"\nNumber of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                chunk_tokens = len(word_tokenize(chunk))
                print(f"\nChunk {i} ({chunk_tokens} tokens):")
            except Exception:
                print(f"\nChunk {i} (token count unavailable):")
            print(chunk)
    
    # Test cases
    
    # Test 1: Normal paragraphs
    test_and_print("Normal Paragraphs", """
    First paragraph with regular content that should be processed normally.
    
    Second paragraph with some more content. This paragraph has multiple sentences. Each sentence should be properly handled.
    
    Third small paragraph with just one sentence.
    """)
    
    # Test 2: Long continuous text
    test_and_print("Long Continuous", "This is a very long sentence. " * 20)
    
    # Test 3: Text with page breaks
    test_and_print("Page Breaks", 
        "Page 1 has this content. Another sentence here.\f\n" + 
        "Page 2 starts here. It continues with more text.\f\n" + 
        "Page 3 is short.")
    
    # Test 4: Very long word sequence
    test_and_print("Long Word", "supercalifragilisticexpialidocious " * 30)
    
    # Test 5: Mixed content with varying sentence lengths
    test_and_print("Mixed Content", """
    Short sentence here. Another short one.
    
    This is a much longer sentence that contains many more words and should probably be in its own chunk because it exceeds our token limit. This is a shorter follow-up sentence.
    
    Final short paragraph.
    """)