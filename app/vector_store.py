import os
from typing import List, Dict, Optional, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from .cache_manager import CacheManager, CachePrefix

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store implementation using FAISS"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 store_path: str = 'data/vector_store/store.pkl',
                 index_type: str = 'L2',
                 force_reload: bool = False):  # Yeni parametre ekledik
        """
        Initialize vector store
        
        Args:
            model_name: Name of the sentence transformer model
            store_path: Path to save/load vector store
            index_type: FAISS index type ('L2' or 'IP' for inner product)
            force_reload: Force reinitialization of the store
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        if index_type == 'IP':
            self.index = faiss.IndexFlatIP(self.dimension)
        else:  # Default to L2
            self.index = faiss.IndexFlatL2(self.dimension)
            
        self.documents = []
        self.document_map = {}  # file_id -> List[indices]
        self.store_path = Path(store_path)
        self.cache_manager = CacheManager()
        
        # Create data directory
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing store or initialize new one
        if not force_reload and self.store_path.exists():
            if not self.load():
                logger.warning("Failed to load existing store, initializing new one")
                self._initialize_new_store()
        else:
            self._initialize_new_store()


    def _initialize_new_store(self):
        """Initialize a new vector store"""
        self.documents = []
        self.document_map = {}
        self.index = faiss.IndexFlatL2(self.dimension)
        self.save()
        logger.info("Initialized new vector store")


    def get_document_info(self) -> List[Dict[str, Any]]:
        """
        Get information about stored documents
        
        Returns:
            List[Dict]: List of document information
        """
        try:
            unique_docs = {}
            for file_id, indices in self.document_map.items():
                if indices:  # If document has any chunks
                    doc = self.documents[indices[0]]
                    unique_docs[file_id] = {
                        'file_id': file_id,
                        'filename': doc['metadata']['filename'],
                        'file_size': doc['metadata']['file_size'],
                        'chunk_count': len(indices),
                        'last_modified': doc['metadata'].get('last_modified', '')
                    }
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return []


    def _generate_file_id(self, filename: str, file_size: int, last_modified: float) -> str:
        """
        Generate unique file ID
        
        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            last_modified: Last modified timestamp
            
        Returns:
            str: Unique file identifier
        """
        content = f"{filename}:{file_size}:{last_modified}"
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _generate_chunk_id(self, text: str, metadata: Dict) -> str:
        """
        Generate unique chunk ID for caching
        
        Args:
            text: Chunk text content
            metadata: Chunk metadata
            
        Returns:
            str: Unique chunk identifier
        """
        content = (
            f"{text}"
            f"{metadata.get('filename', '')}"
            f"{metadata.get('file_size', '')}"
            f"{metadata.get('chunk_id', '')}"
            f"{metadata.get('last_modified', '')}"
        )
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _compute_embedding(self, text: str, metadata: Dict) -> np.ndarray:
        """
        Compute embedding with caching support
        
        Args:
            text: Text to embed
            metadata: Document metadata
            
        Returns:
            np.ndarray: Computed embedding
        """
        chunk_id = self._generate_chunk_id(text, metadata)
        cache_key = self.cache_manager.generate_key(CachePrefix.VECTOR_EMBEDDING, chunk_id)
        
        # Check cache first
        cached_data = self.cache_manager.get_cache(cache_key)
        if cached_data and 'embedding' in cached_data:
            logger.debug(f"Using cached embedding for chunk {chunk_id[:8]}")
            return np.array(cached_data['embedding']).astype('float32')
        
        # Compute new embedding
        embedding = self.model.encode(text, convert_to_tensor=False)
        
        # Cache the embedding
        self.cache_manager.set_cache(cache_key, {
            'embedding': embedding.tolist(),
            'metadata': metadata,
            'timestamp': datetime.now().timestamp()
        })
        
        return embedding.astype('float32')

    def save(self) -> bool:
        """
        Save vector store state to disk
        
        Returns:
            bool: Success status
        """
        try:
            state = {
                'documents': self.documents,
                'document_map': self.document_map,
                'index_data': faiss.serialize_index(self.index),
                'model_name': self.model_name,
                'timestamp': datetime.now().timestamp()
            }
            
            with open(self.store_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Vector store saved to {self.store_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    

    def load(self) -> bool:
        """
        Load vector store state from disk
        
        Returns:
            bool: Success status
        """
        try:
            if not self.store_path.exists():
                logger.info(f"No existing vector store found at {self.store_path}")
                return False
            
            with open(self.store_path, 'rb') as f:
                state = pickle.load(f)
            
            # Model compatibility check - sadece boyut kontrolü yapalım
            stored_dim = len(state['documents'][0]['embedding']) if state['documents'] else self.dimension
            if stored_dim != self.dimension:
                logger.error(f"Dimension mismatch: stored={stored_dim}, current={self.dimension}")
                return False
            
            self.documents = state.get('documents', [])
            self.document_map = state.get('document_map', {})
            
            # Recreate index with stored embeddings
            if self.documents:
                embeddings = [doc['embedding'] for doc in self.documents]
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.array(embeddings).astype('float32'))
            
            logger.info(f"Vector store loaded from {self.store_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False


    def add_documents(self, chunks: List[str], metadata: Dict) -> str:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of text chunks
            metadata: Document metadata
            
        Returns:
            str: File identifier
        """
        try:
            # Generate unique file ID
            file_id = self._generate_file_id(
                metadata['filename'],
                metadata['file_size'],
                metadata.get('last_modified', datetime.now().timestamp())
            )
            
            # Remove existing document if present
            if file_id in self.document_map:
                self.remove_document(file_id)
            
            # Process chunks
            embeddings = []
            start_idx = len(self.documents)
            chunk_indices = []
            
            for i, chunk in enumerate(chunks):
                # Update metadata for chunk
                chunk_metadata = {
                    **metadata,
                    'chunk_id': i,
                    'timestamp': datetime.now().timestamp()
                }
                
                # Compute embedding
                embedding = self._compute_embedding(chunk, chunk_metadata)
                embeddings.append(embedding)
                
                # Store document info
                self.documents.append({
                    'file_id': file_id,
                    'chunk_id': i,
                    'text': chunk,
                    'metadata': chunk_metadata
                })
                chunk_indices.append(start_idx + i)
            
            # Update index
            if embeddings:
                embeddings_array = np.array(embeddings)
                self.index.add(embeddings_array)
                
            # Update document map
            self.document_map[file_id] = chunk_indices
            
            # Save state
            self.save()
            
            logger.info(f"Added document {metadata['filename']} with {len(chunks)} chunks")
            return file_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def remove_document(self, file_id: str) -> bool:
        """
        Remove document from vector store
        
        Args:
            file_id: File identifier
            
        Returns:
            bool: Success status
        """
        if file_id not in self.document_map:
            return False
            
        try:
            # Get indices to remove
            indices_to_remove = set(self.document_map[file_id])
            
            # Create new index
            if isinstance(self.index, faiss.IndexFlatIP):
                new_index = faiss.IndexFlatIP(self.dimension)
            else:
                new_index = faiss.IndexFlatL2(self.dimension)
            
            # Copy data excluding removed document
            new_documents = []
            new_document_map = {}
            embeddings_to_keep = []
            new_idx = 0
            
            for old_idx, doc in enumerate(self.documents):
                if old_idx not in indices_to_remove:
                    # Get embedding
                    embedding = self.index.reconstruct(old_idx)
                    embeddings_to_keep.append(embedding)
                    
                    # Update document
                    new_documents.append(doc)
                    
                    # Update document map
                    doc_id = doc['file_id']
                    if doc_id not in new_document_map:
                        new_document_map[doc_id] = []
                    new_document_map[doc_id].append(new_idx)
                    new_idx += 1
            
            # Update index
            if embeddings_to_keep:
                embeddings_array = np.array(embeddings_to_keep)
                new_index.add(embeddings_array)
            
            # Update state
            self.index = new_index
            self.documents = new_documents
            self.document_map = new_document_map
            
            # Remove embeddings from cache
            for doc in self.documents:
                if doc['file_id'] == file_id:
                    chunk_id = self._generate_chunk_id(doc['text'], doc['metadata'])
                    cache_key = self.cache_manager.generate_key(
                        CachePrefix.VECTOR_EMBEDDING, 
                        chunk_id
                    )
                    self.cache_manager.delete_cache(cache_key)
            
            # Save state
            self.save()
            
            logger.info(f"Removed document {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {str(e)}")
            return False

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents
        """
        try:
            # Compute query embedding
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            
            # Search index
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                min(k, len(self.documents))
            )
            
            # Collect results
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def has_documents(self) -> bool:
        """
        Check if there are any documents in the store
        
        Returns:
            bool: True if documents exist
        """
        return bool(self.documents and self.index.ntotal > 0)