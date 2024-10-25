from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging
import hashlib
from datetime import datetime
from .vector_store import VectorStore
from .llm.base import BaseLLM, Message, LLMResponse
from .llm.prompts import RAGPromptBuilder
from .cache_manager import CacheManager, CachePrefix

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    Handles document retrieval, caching, and LLM response generation
    """
    
    def __init__(self, vector_store: VectorStore, llm: BaseLLM):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store: Vector store instance for document retrieval
            llm: Language model instance for response generation
        """
        self.vector_store = vector_store
        self.llm = llm
        self.cache_manager = CacheManager()

    def _generate_chunk_fingerprint(self, chunk_text: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique fingerprint for a document chunk
        
        Args:
            chunk_text: The text content of the chunk
            metadata: Document metadata
            
        Returns:
            str: Unique fingerprint for the chunk
        """
        content = (
            f"{chunk_text}"
            f"{metadata.get('filename', '')}"
            f"{metadata.get('file_size', '')}"
            f"{metadata.get('chunk_id', '')}"
            f"{metadata.get('last_modified', '')}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_cache_key(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Generate cache key for query-document combination
        
        Args:
            query: User query
            relevant_docs: List of relevant document chunks
            
        Returns:
            str: Cache key
        """
        # Generate unique fingerprints for each chunk
        chunk_fingerprints = [
            self._generate_chunk_fingerprint(doc['text'], doc['metadata'])
            for doc in relevant_docs
        ]
        # Sort for consistency
        chunk_fingerprints.sort()
        # Combine query with chunk fingerprints
        content = f"{query}:{'|'.join(chunk_fingerprints)}"
        return self.cache_manager.generate_key(CachePrefix.QUERY_RESPONSE, content)

    async def generate_response(self, 
                              query: str,
                              chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User query
            chat_history: Optional chat history for context
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Validate document store
            if not self.vector_store.has_documents():
                return {
                    'status': 'success',
                    'response': 'Please upload some documents first.',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Validate LLM connection
            if not await self.llm.validate_connection():
                return {
                    'status': 'error',
                    'response': 'Language model service is currently unavailable.',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Get relevant documents
            relevant_docs = self.vector_store.search(query, k=3)
            if not relevant_docs:
                return {
                    'status': 'success',
                    'response': 'No relevant information found in the documents.',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Check cache first
            cache_key = self._generate_cache_key(query, relevant_docs)
            cached_response = self.cache_manager.get_cache(cache_key)
            if cached_response:
                logger.info("Cache hit: Using cached response")
                return cached_response

            # Generate source identifiers for response
            sources = []
            for doc in relevant_docs:
                chunk_id = self._generate_chunk_fingerprint(doc['text'], doc['metadata'])[:8]
                source = f"{doc['metadata']['filename']} (chunk: {chunk_id})"
                sources.append(source)
            sources = list(set(sources))  # Remove duplicates

            # Prepare context for LLM
            context_parts = []
            for doc in relevant_docs:
                chunk_id = self._generate_chunk_fingerprint(doc['text'], doc['metadata'])[:8]
                context_parts.append(
                    f"From {doc['metadata']['filename']} (chunk: {chunk_id}):\n{doc['text']}"
                )
            context = "\n\n".join(context_parts)

            # Process chat history
            history_messages = []
            if chat_history:
                history_messages = [
                    Message(role=msg['role'], content=msg['content'])
                    for msg in chat_history
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg
                ]

            # Build messages for LLM
            messages = RAGPromptBuilder.build_rag_messages(
                question=query,
                context=context,
                chat_history=history_messages
            )

            try:
                # Get LLM response
                response = await self.llm.chat(messages=messages, stream=True)
                
                # Prepare result
                result = {
                    'status': 'success',
                    'response': response,  # AsyncGenerator for streaming
                    'sources': sources,
                    'timestamp': datetime.now().timestamp()
                }

                # Cache only non-streaming responses
                if not isinstance(response, AsyncGenerator):
                    self.cache_manager.set_cache(cache_key, result)

                return result
                
            except Exception as e:
                logger.error(f"LLM error: {str(e)}")
                return {
                    'status': 'error',
                    'response': 'Error generating response from language model.',
                    'sources': sources,
                    'timestamp': datetime.now().timestamp()
                }

        except Exception as e:
            logger.error(f"RAG pipeline error: {str(e)}")
            return {
                'status': 'error',
                'response': 'Error processing request in RAG pipeline.',
                'sources': [],
                'timestamp': datetime.now().timestamp()
            }

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Add any cleanup logic here
            pass
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")