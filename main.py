import streamlit as st
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from asyncio import AbstractEventLoop
from typing import AsyncGenerator

from app.document_processor import DocumentProcessor
from app.llm.base import LLMResponse
from app.rag_pipeline import RAGPipeline
from app.chat_manager import ChatManager
from app.database import Database
from app.llm.ollama_llm import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_TYPES = ["txt", "pdf", "doc", "docx"]

def check_file_size(file) -> bool:
    """Check if file size is less than MAX_FILE_SIZE"""
    return file.size <= MAX_FILE_SIZE

def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

async def init_llm() -> OllamaLLM:
    """Initialize and validate LLM connection"""
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        host=os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )
    
    for i in range(3):
        try:
            is_connected = await llm.validate_connection()
            if is_connected:
                logger.info("Successfully connected to Ollama")
                return llm
            logger.warning(f"Attempt {i+1}: Failed to connect to Ollama, retrying...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Connection attempt {i+1} failed: {str(e)}")
            if i < 2:
                await asyncio.sleep(5)
    
    logger.error("Failed to connect to Ollama after multiple attempts")
    return llm

def get_async_loop() -> AbstractEventLoop:
    """Get or create an event loop"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def initialize_session_state():
    """Initialize session state variables"""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'llm' not in st.session_state:
        loop = get_async_loop()
        st.session_state.llm = loop.run_until_complete(init_llm())
        
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            vector_store=st.session_state.document_processor.vector_store,
            llm=st.session_state.llm
        )
    
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    if 'database' not in st.session_state:
        st.session_state.database = Database()
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = st.session_state.chat_manager.create_session()
    
    if 'active_files' not in st.session_state:
        st.session_state.active_files = {}


def remove_file(file_id: str):
    """Remove file from vector store"""
    try:
        vector_store = st.session_state.document_processor.vector_store
        if vector_store.remove_document(file_id):
            # Update session state
            st.session_state.active_files = {
                k: v for k, v in st.session_state.active_files.items()
                if k != file_id
            }
            
            # Clear chat if no documents remain
            if not vector_store.has_documents():
                clear_current_chat()
                create_new_chat()
            
            # Force sidebar refresh
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error removing file: {str(e)}")
        st.error(f"Error removing file: {str(e)}")

def create_new_chat():
    """Create new chat and update session state"""
    if st.session_state.chat_manager:
        new_chat_id = st.session_state.chat_manager.create_session()
        st.session_state.current_chat_id = new_chat_id

def clear_current_chat():
    """Clear current chat history"""
    if st.session_state.database and st.session_state.current_chat_id:
        st.session_state.database.clear_chat_history(st.session_state.current_chat_id)

def process_stream_response(response_stream: AsyncGenerator) -> str:
    """Process streaming response"""
    loop = get_async_loop()
    full_response = ""
    
    async def collect_stream():
        nonlocal full_response
        async for chunk in response_stream:
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                yield full_response
    
    try:
        return loop.run_until_complete(collect_stream())
    except Exception as e:
        logger.error(f"Error processing stream: {str(e)}")
        return ""

async def process_query(pipeline: RAGPipeline, 
                       query: str, 
                       chat_history: list) -> Dict[str, Any]:
    """Process query using RAG pipeline"""
    try:
        return await pipeline.generate_response(query, chat_history)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'response': 'An error occurred while processing your question.',
            'sources': [],
            'timestamp': datetime.now().timestamp()
        }

def render_chat_interface():
    st.title("🤖 RAG Chatbot")
    
    # Get chat history
    chat_history = st.session_state.database.get_chat_history(
        st.session_state.current_chat_id
    )
    
    # Display chat history
    for message in chat_history:
        with st.chat_message("user"):
            st.write(message['query'])
        with st.chat_message("assistant"):
            st.write(message['response'])
            if message['sources']:
                st.caption(f"Sources: {', '.join(message['sources'])}")

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            async def generate_response():
                nonlocal full_response
                response = await process_query(
                    st.session_state.rag_pipeline,
                    query,
                    chat_history
                )
                
                if response['status'] == 'success':
                    if isinstance(response['response'], AsyncGenerator):
                        async for chunk in response['response']:
                            if isinstance(chunk, LLMResponse):
                                full_response += chunk.content
                                # Streamlit'in stream yazma özelliğini kullanalım
                                placeholder.markdown(full_response)
                                await asyncio.sleep(0.005) 
                    else:
                        full_response = response['response']
                        placeholder.markdown(full_response)

                    if response.get('sources'):
                        st.caption(f"Sources: {', '.join(response['sources'])}")
                    
                    # Save to database
                    st.session_state.database.save_chat_message(
                        st.session_state.current_chat_id,
                        query,
                        {
                            'response': full_response,
                            'sources': response.get('sources', []),
                            'timestamp': response['timestamp']
                        }
                    )
                else:
                    st.error(response.get('error', 'An error occurred'))

            # Run the async function
            loop = get_async_loop()
            loop.run_until_complete(generate_response())

def render_sidebar():
    """Render sidebar content"""
    with st.sidebar:
        st.title("📚 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key="file_uploader"  # Unique key eklendi
        )
        
        # New chat and Clear chat buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("🆕 New Chat", key="new_chat", on_click=create_new_chat)
        with col2:
            st.button("🗑️ Clear Chat", key="clear_chat", on_click=clear_current_chat)
        
        # Show active documents
        st.write("---")
        st.write("📁 Active Documents:")
        
        if 'document_processor' not in st.session_state:
            return

        vector_store = st.session_state.document_processor.vector_store
        doc_info = vector_store.get_document_info()
        
        # Initialize active_files in session state if not exists
        if 'active_files' not in st.session_state:
            st.session_state.active_files = {}

        if not doc_info:
            st.write("No documents loaded")
        else:
            # Create a container for document list
            doc_container = st.container()
            
            with doc_container:
                for doc in doc_info:
                    file_id = doc['file_id']
                    
                    # Skip if already in active_files
                    if file_id in st.session_state.active_files:
                        continue
                        
                    # Add to active_files
                    st.session_state.active_files[file_id] = doc['filename']
                    
                # Display active files from session state
                for file_id, filename in st.session_state.active_files.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {filename}")
                        doc_data = next((d for d in doc_info if d['file_id'] == file_id), None)
                        if doc_data:
                            st.caption(f"Chunks: {doc_data['chunk_count']}")
                    with col2:
                        # Use unique key for each delete button
                        if st.button("🗑️", key=f"remove_{file_id}", help="Remove document"):
                            remove_file(file_id)
        
        # Process uploads
        if uploaded_files:
            st.write("---")
            st.write("📤 Processing Uploads:")
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name in [v for v in st.session_state.active_files.values()]:
                    st.warning(f"⚠️ {uploaded_file.name} already exists")
                    continue
                    
                if not check_file_size(uploaded_file):
                    st.error(f"❌ {uploaded_file.name} (Too large)")
                    continue
                
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    result = st.session_state.document_processor.process_document(
                        uploaded_file,
                        file_extension
                    )
                    
                    if result['status'] == 'success':
                        st.success(f"✅ {uploaded_file.name}")
                        st.caption(f"Chunks: {result['chunks']}")
                        # Force sidebar refresh after successful upload
                        st.rerun()
                    else:
                        st.error(f"❌ {uploaded_file.name}")
                        st.caption(f"Error: {result['error']}")
                
                except Exception as e:
                    logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                    st.error(f"❌ {uploaded_file.name}")
                    st.caption(f"Error: {str(e)}")

def main():
    """Main application entry point"""
    try:
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide"
        )

        initialize_session_state()
        render_sidebar()
        render_chat_interface()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()