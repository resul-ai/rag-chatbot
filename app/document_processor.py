import os
from typing import List, Optional, Dict, Any
import PyPDF2
from docx import Document
from .utils import TextProcessor
from .vector_store import VectorStore

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            'txt': self._process_txt,
            'pdf': self._process_pdf,
            'doc': self._process_doc,
            'docx': self._process_doc
        }
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        
        # Try to load existing vector store
        self.vector_store.load()

    def process_document(self, file, file_type: str) -> Dict[str, Any]:
        """Process the uploaded document and add to vector store"""
        try:
            # Extract text from document
            if file_type.lower() not in self.supported_formats:
                return {
                    'status': 'error',
                    'error': f'Unsupported file type: {file_type}',
                    'metadata': {'filename': file.name}
                }

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
            
            # Split text into chunks
            chunks = self.text_processor.split_into_chunks(text)
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadata)
            
            # Save updated vector store
            self.vector_store.save()
            
            return {
                'status': 'success',
                'chunks': len(chunks),
                'metadata': metadata
            }
            
        except Exception as e:
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
            print(f"Error processing TXT: {str(e)}")
            return ""

    def _process_pdf(self, file) -> str:
        """Process pdf files"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
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
            print(f"Error processing DOC: {str(e)}")
            return ""