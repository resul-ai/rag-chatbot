from typing import List, Optional
from pydantic import BaseModel
from .base import Message

class RAGPromptBuilder:
    """Builds RAG prompts in chat format"""
    
    SYSTEM_TEMPLATE = """You are a helpful assistant answering questions based on provided documents.
Your task is to provide accurate, relevant answers using only the context provided.
If the context doesn't contain enough information to answer the question, say so.
Keep your answers concise and to the point."""

    @classmethod
    def build_rag_messages(cls,
                          question: str,
                          context: str,
                          chat_history: Optional[List[Message]] = None) -> List[Message]:
        """Build messages list for RAG"""
        messages = [
            Message(role="system", content=cls.SYSTEM_TEMPLATE),
            Message(role="user", content=f"""Context:
{context}

Question: {question}

Instructions:
1. Answer based solely on the provided context
2. If you cannot answer from the context, say so
3. Be concise and direct
4. Use quotes when referencing the context""")
        ]
        
        if chat_history:
            # Insert chat history before the current question
            messages[1:1] = chat_history
            
        return messages