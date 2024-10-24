import streamlit as st
from app.document_processor import DocumentProcessor
from app.rag_pipeline import RAGPipeline
from app.chat_manager import ChatManager
from app.database import Database
import os
from datetime import datetime

def check_file_size(file):
    """Check if file size is less than 5MB"""
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
    return file.size <= MAX_FILE_SIZE

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def initialize_session_state():
    """Initialize session state variables"""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.document_processor.vector_store
        )
    
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    if 'database' not in st.session_state:
        st.session_state.database = Database()
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = "default"
    
    if 'active_files' not in st.session_state:
        st.session_state.active_files = {}

def create_new_chat():
    """Create new chat and update session state"""
    if st.session_state.chat_manager:
        new_chat_id = st.session_state.chat_manager.create_session()
        st.session_state.current_chat_id = new_chat_id

def clear_current_chat():
    """Clear current chat history"""
    if st.session_state.database and st.session_state.current_chat_id:
        st.session_state.database.clear_chat_history(st.session_state.current_chat_id)

def remove_file(file_id: str):
    """Remove file from vector store"""
    if file_id in st.session_state.get('file_to_remove', {}):
        filename = st.session_state.file_to_remove[file_id]
        vector_store = st.session_state.document_processor.vector_store
        
        if vector_store.remove_document(file_id):
            if file_id in st.session_state.active_files:
                del st.session_state.active_files[file_id]
            vector_store.save()
            
            # Clear chat history when all documents are removed
            if not vector_store.has_documents():
                clear_current_chat()
                create_new_chat()

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Create sidebar
    with st.sidebar:
        st.title("📚 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents (max 5 files, each under 5MB)", 
            type=["txt", "pdf", "doc", "docx"],
            accept_multiple_files=True
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
        
        # Get document info from vector store
        vector_store = st.session_state.document_processor.vector_store
        doc_info = vector_store.get_document_info()
        
        if not doc_info:
            st.write("No documents loaded")
        else:
            for doc in doc_info:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc['filename']}")
                    st.caption(f"Chunks: {doc['chunk_count']}")
                with col2:
                    # Store file info in session state for removal
                    st.session_state.file_to_remove = {
                        doc['file_id']: doc['filename']
                    }
                    st.button(
                        "🗑️",
                        key=f"remove_{doc['file_id']}",
                        on_click=remove_file,
                        args=(doc['file_id'],)
                    )
        
        # Process new uploads
        if uploaded_files:
            st.write("---")
            st.write("📤 Processing Uploads:")
            
            for uploaded_file in uploaded_files:
                if not check_file_size(uploaded_file):
                    st.error(f"❌ {uploaded_file.name} (Too large)")
                    continue
                
                # Process the file
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    result = st.session_state.document_processor.process_document(
                        uploaded_file,
                        file_extension
                    )
                    
                    if result.get('status') == 'success':
                        st.success(f"✅ {uploaded_file.name}")
                        st.caption(f"Chunks: {result.get('chunks', 0)}")
                    else:
                        st.error(f"❌ {uploaded_file.name}")
                        st.caption(f"Error: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}")
                    st.caption(f"Error: {str(e)}")

    # Main chat interface
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
            response = st.session_state.rag_pipeline.generate_response(
                query,
                chat_history
            )
            
            if response['status'] == 'success':
                st.write(response['response'])
                if response.get('sources'):
                    st.caption(f"Sources: {', '.join(response['sources'])}")
                
                # Save to database
                st.session_state.database.save_chat_message(
                    st.session_state.current_chat_id,
                    query,
                    response
                )
            else:
                st.error(response.get('error', 'An error occurred'))

if __name__ == "__main__":
    main()