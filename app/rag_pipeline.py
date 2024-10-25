from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging
from datetime import datetime
from .vector_store import VectorStore
from .llm.base import BaseLLM, Message, LLMResponse
from .llm.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, llm: BaseLLM):
        self.vector_store = vector_store
        self.llm = llm
        
    async def generate_response(self, 
                              query: str,
                              chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate response using RAG pipeline"""
        try:
            # Check vector store first
            if not self.vector_store.has_documents():
                return {
                    'status': 'success',  # error yerine success
                    'response': 'I notice you haven\'t uploaded any documents yet. Please upload some documents so I can assist you better with specific information from them. In the meantime, is there anything else I can help you with?',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Validate LLM connection
            is_connected = await self.llm.validate_connection()
            if not is_connected:
                return {
                    'status': 'error',
                    'error': 'LLM service is not available',
                    'response': 'The language model service is currently unavailable. Please try again later.',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Get relevant documents
            relevant_docs = self.vector_store.search(query, k=3)
            
            if not relevant_docs:
                return {
                    'status': 'error',
                    'error': 'No relevant documents found',
                    'response': 'I could not find any relevant information in the documents to answer your question.',
                    'sources': [],
                    'timestamp': datetime.now().timestamp()
                }

            # Format context
            context = "\n\n".join([
                f"From {doc['metadata']['filename']}:\n{doc['text']}"
                for doc in relevant_docs
            ])

            # Convert chat history to Message objects if provided
            history_messages = []
            if chat_history:
                for msg in chat_history:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        history_messages.append(Message(
                            role=msg['role'],
                            content=msg['content']
                        ))

            # Build messages using RAGPromptBuilder
            messages = RAGPromptBuilder.build_rag_messages(
                question=query,
                context=context,
                chat_history=history_messages
            )

            # Get sources
            sources = list(set(
                doc['metadata']['filename'] 
                for doc in relevant_docs
            ))

            try:
                messages = RAGPromptBuilder.build_rag_messages(
                question=query,
                context=context,
                chat_history=history_messages
                )
                # Get response from LLM with streaming enabled
                response = await self.llm.chat(messages=messages, stream=True)

                return {
                        'status': 'success',
                        'response': response,  # This is now an AsyncGenerator
                        'sources': sources,
                        'timestamp': datetime.now().timestamp()
                        }
                
            except Exception as e:
                logger.error(f"LLM error: {str(e)}")
                return {
                    'status': 'error',
                    'error': f'LLM error: {str(e)}',
                    'response': 'An error occurred while generating the response.',
                    'sources': sources,
                    'timestamp': datetime.now().timestamp()
                }

        except Exception as e:
            logger.error(f"RAG pipeline error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'response': 'An error occurred while processing your question.',
                'sources': [],
                'timestamp': datetime.now().timestamp()
            }