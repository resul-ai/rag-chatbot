from typing import List, Optional, Dict, Any
import PyPDF2
from docx import Document
from .text_processor import TextProcessor
from .vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            'txt': self._process_txt,
            'pdf': self._process_pdf,
            'doc': self._process_doc,
            'docx': self._process_doc
        }
        self.text_processor = TextProcessor()  # default max_tokens=768 ile
        self.vector_store = VectorStore()
        
        # Load existing vector store if available
        self.vector_store.load()

    def process_document(self, file, file_type: str) -> Dict[str, Any]:
        """Process document and add to vector store"""
        try:
            # Validate file type
            if file_type.lower() not in self.supported_formats:
                return {
                    'status': 'error',
                    'error': f'Unsupported file type: {file_type}',
                    'metadata': {'filename': file.name}
                }

            # Extract text
            text = self.supported_formats[file_type.lower()](file)
            
            if not text:
                return {
                    'status': 'error',
                    'error': 'No text could be extracted from the document',
                    'metadata': {'filename': file.name}
                }
            
            # Create metadata
            metadata = {
                'filename': file.name,
                'file_type': file_type,
                'file_size': file.size
            }
            
            # Process text into chunks - using enhanced TextProcessor
            chunks = self.text_processor.split_into_chunks(text)
            
            if not chunks:
                return {
                    'status': 'error',
                    'error': 'No valid text chunks could be extracted',
                    'metadata': metadata
                }
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadata)
            
            return {
                'status': 'success',
                'chunks': len(chunks),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'metadata': {'filename': getattr(file, 'name', 'unknown')}
            }

    def _process_txt(self, file) -> str:
        """Process txt files"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error processing TXT: {str(e)}")
            return ""

    def _process_pdf(self, file) -> str:
        """Process pdf files"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\f"  # Add form feed character to mark page breaks
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return ""

    def _process_doc(self, file) -> str:
        """Process doc/docx files"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing DOC: {str(e)}")
            return ""