import asyncio
from app.llm.ollama_llm import OllamaLLM
from app.llm.base import Message
import logging

logging.basicConfig(level=logging.INFO)

async def test_ollama_connection():
    # Initialize LLM
    llm = OllamaLLM(
        model="llama3.2",
        host="http://ollama:11434"
    )
    
    # Test connection
    print("Testing connection...")
    is_connected = await llm.validate_connection()
    print(f"Connection status: {is_connected}")
    
    if is_connected:
        # Test simple query
        messages = [
            Message(role="user", content="Hi, how are you?")
        ]
        
        print("\nTesting chat...")
        try:
            response = await llm.chat(messages=messages)
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"Chat error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_ollama_connection())