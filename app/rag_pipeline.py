from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from sentence_transformers import SentenceTransformer
import json
import time

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_response(self, 
                         query: str, 
                         chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline
        """
        try:
            # Check if there are any documents in vector store
            if not self.vector_store.has_documents():
                return {
                    'status': 'error',
                    'error': 'No documents available',
                    'response': 'Please upload some documents first.',
                    'sources': [],
                    'timestamp': time.time()
                }

            # Get relevant documents
            relevant_docs = self.vector_store.search(query, k=3)
            
            if not relevant_docs:
                return {
                    'status': 'error',
                    'error': 'No relevant documents found',
                    'response': 'I could not find any relevant information to answer your question.',
                    'sources': [],
                    'timestamp': time.time()
                }

            # Format context from relevant documents
            context = "\n".join([doc['text'] for doc in relevant_docs])
            
            # Format source information
            sources = []
            for doc in relevant_docs:
                if doc['metadata']['filename'] not in sources:
                    sources.append(doc['metadata']['filename'])

            # For now, we'll return a simple response with context
            response = {
                'status': 'success',
                'response': f"Based on the documents, here is what I found:\n\n{context}",
                'sources': sources,
                'timestamp': time.time()
            }
            
            return response

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': 'An error occurred while processing your question.',
                'sources': [],
                'timestamp': time.time()
            }