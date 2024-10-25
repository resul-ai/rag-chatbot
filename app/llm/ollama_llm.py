from typing import List, Union, AsyncGenerator, Dict
from ollama import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseLLM, LLMResponse, Message
import logging
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM):
    def __init__(self, 
                 model: str = "llama3.2", 
                 host: str = "http://localhost:11434",
                 **kwargs):
        super().__init__(model, **kwargs)
        self.client = AsyncClient(host=host)
        self.host = host
        self._ensure_event_loop()

    @contextmanager
    def _ensure_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            yield loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            yield loop

    async def validate_connection(self) -> bool:
        with self._ensure_event_loop():
            try:
                response = await self.client.list()
                models = response.get('models', [])
                available_models = [m.get('name') for m in models]
                logger.info(f"Available models: {available_models}")
                
                # Check if model exists (with or without tag)
                model_exists = any(m.startswith(self.model) for m in available_models)
                
                if not model_exists:
                    logger.warning(f"Model {self.model} not found")
                    return False
                    
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {str(e)}")
                return False

    async def chat(self,
                messages: List[Message],
                stream: bool = True) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat response using Ollama with streaming enabled by default"""
        try:
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            response = await self.client.chat(
                model=self.model,
                messages=formatted_messages,
                stream=stream
            )

            if stream:
                async def response_generator():
                    async for chunk in response:
                        if isinstance(chunk, dict) and 'message' in chunk:
                            yield LLMResponse(
                                content=chunk['message']['content'],
                                model=self.model,
                                usage={"total_tokens": chunk.get('eval_count', 0)}
                            )
                return response_generator()
            else:
                return LLMResponse(
                    content=response['message']['content'],
                    model=self.model,
                    usage={"total_tokens": response.get('eval_count', 0)}
                )

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            raise