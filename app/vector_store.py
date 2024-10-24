import os
from typing import List, Dict, Optional, Set
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import hashlib

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize vector store with a specific embedding model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Dict] = []
        self.file_ids: Set[str] = set()  # Track active file IDs
        
        # Create data directory if it doesn't exist
        os.makedirs('data/vector_store', exist_ok=True)

    def _generate_file_id(self, filename: str, filesize: int) -> str:
        """Generate unique file ID"""
        return hashlib.md5(f"{filename}_{filesize}".encode()).hexdigest()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text"""
        return self.model.encode([text])[0]

    def add_documents(self, chunks: List[str], metadata: Dict) -> str:
        """Add document chunks to the vector store"""
        # Generate file ID
        file_id = self._generate_file_id(metadata['filename'], metadata['file_size'])
        
        # Store file ID
        self.file_ids.add(file_id)
        
        # Process each chunk
        embeddings = []
        start_idx = len(self.documents)
        
        for i, chunk in enumerate(chunks):
            embedding = self._generate_embedding(chunk)
            embeddings.append(embedding)
            
            # Store document info
            self.documents.append({
                'file_id': file_id,
                'chunk_id': i,
                'text': chunk,
                'metadata': metadata
            })

        # Add embeddings to FAISS index
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
        
        return file_id

    def remove_document(self, file_id: str) -> bool:
        """Remove document and its chunks from vector store"""
        if file_id not in self.file_ids:
            return False
        
        # Get indices of chunks to remove
        indices_to_remove = [
            i for i, doc in enumerate(self.documents)
            if doc['file_id'] == file_id
        ]
        
        if not indices_to_remove:
            return False
        
        # Create new index and documents list without the removed file
        new_index = faiss.IndexFlatL2(self.dimension)
        new_documents = []
        
        # Copy embeddings and documents that we want to keep
        embeddings_to_keep = []
        
        for i, doc in enumerate(self.documents):
            if i not in indices_to_remove:
                # Get embedding from original index
                embedding = self.index.reconstruct(i)
                embeddings_to_keep.append(embedding)
                new_documents.append(doc)
        
        # Add kept embeddings to new index
        if embeddings_to_keep:
            embeddings_array = np.array(embeddings_to_keep).astype('float32')
            new_index.add(embeddings_array)
        
        # Update index and documents
        self.index = new_index
        self.documents = new_documents
        self.file_ids.remove(file_id)
        
        return True

    def get_document_info(self) -> List[Dict]:
        """Get information about all stored documents"""
        unique_docs = {}
        for doc in self.documents:
            file_id = doc['file_id']
            if file_id not in unique_docs:
                unique_docs[file_id] = {
                    'file_id': file_id,
                    'filename': doc['metadata']['filename'],
                    'file_size': doc['metadata']['file_size'],
                    'chunk_count': 1
                }
            else:
                unique_docs[file_id]['chunk_count'] += 1
        
        return list(unique_docs.values())

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self._generate_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        
        # Get corresponding documents
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

    def save(self, filepath: str = 'data/vector_store/store.pkl') -> None:
        """Save the vector store to disk"""
        state = {
            'documents': self.documents,
            'file_ids': self.file_ids,
            'index': faiss.serialize_index(self.index)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str = 'data/vector_store/store.pkl') -> bool:
        """Load the vector store from disk"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.documents = state['documents']
            self.file_ids = state.get('file_ids', set())  # Backward compatibility
            self.index = faiss.deserialize_index(state['index'])
            return True
        except (FileNotFoundError, EOFError):
            return False
        
    def has_documents(self) -> bool:
        """Check if there are any documents in the store"""
        return len(self.documents) > 0 and self.index.ntotal > 0