# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application that enables document-based question answering through an intuitive web interface. The application allows users to upload documents, processes them into a vector store, and answers questions based on the document content using advanced language models.

## Features

- 📄 **Document Processing**
  - Support for PDF, DOC, DOCX, and TXT files
  - Intelligent text chunking with sentence boundary preservation
  - Automatic document vectorization and indexing

- 🤖 **Advanced QA Capabilities**
  - RAG-based question answering
  - Context-aware responses
  - Source attribution for answers
  - Chat history management

- 💻 **User Interface**
  - Clean, intuitive Streamlit interface
  - Real-time document management
  - Interactive chat interface
  - Document upload progress tracking

- 🔧 **Technical Features**
  - FAISS vector store for efficient similarity search
  - Redis-based response caching
  - SQLite chat history persistence
  - Ollama LLM integration
  - Docker containerization

## System Architecture

```
├── app/
│   ├── llm/           # Language model integration
│   ├── cache_manager  # Redis caching system
│   ├── vector_store   # FAISS vector store
│   ├── text_processor # Document processing
│   └── rag_pipeline   # Core RAG implementation
├── data/              # Persistent storage
├── docker/            # Container configuration
└── main.py           # Application entry point
```

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- 10GB+ free disk space

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Start the application using Docker Compose:
```bash
docker-compose up --build
```

3. Access the web interface:
- Open your browser and navigate to `http://localhost:8501`

## Usage Guide

1. **Upload Documents**
   - Click the "Upload Documents" button in the sidebar
   - Select one or more supported documents
   - Wait for processing completion

2. **Ask Questions**
   - Type your question in the chat input
   - The system will search relevant documents
   - View the answer with source references

3. **Manage Documents**
   - View active documents in the sidebar
   - Remove documents as needed
   - Start new chat sessions

## Configuration

Key environment variables (set in `docker-compose.yml`):

```yaml
OLLAMA_HOST: http://ollama:11434
OLLAMA_MODEL: llama3.2
REDIS_URL: redis://redis:6379
DEBUG: true
```

## Development Setup

For local development without Docker:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start required services (Redis, Ollama) separately

4. Run the application:
```bash
streamlit run main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed by R. Caliskan*