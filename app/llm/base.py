from typing import Dict, Optional, Any, List, AsyncGenerator, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel

class Message(BaseModel):
    """Chat message format"""
    role: str
    content: str

class LLMResponse(BaseModel):
    """Standard response format for all LLM implementations"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLM(ABC):
    """Base class for all LLM implementations"""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def chat(self, 
                  messages: List[Message],
                  temperature: float = 0.7,
                  stream: bool = False) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """
        Generate response using chat format
        
        Args:
            messages: List of chat messages
            temperature: Controls randomness (0.0 to 1.0)
            stream: Whether to stream the response
            
        Returns:
            Either a single LLMResponse or an AsyncGenerator of LLMResponses if streaming
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the LLM service is accessible"""
        pass