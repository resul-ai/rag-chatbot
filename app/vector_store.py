import os
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize vector store"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.document_map = {}  # file_id -> List[indices]
        self.store_path = 'data/vector_store/store.pkl'
        
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        
    def save(self, filepath: Optional[str] = None) -> bool:
        """Save vector store to disk"""
        try:
            save_path = filepath or self.store_path
            state = {
                'documents': self.documents,
                'document_map': self.document_map,
                'index_data': faiss.serialize_index(self.index)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Vector store saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self, filepath: Optional[str] = None) -> bool:
        """Load vector store from disk"""
        try:
            load_path = filepath or self.store_path
            
            if not os.path.exists(load_path):
                logger.info(f"No existing vector store found at {load_path}")
                return False
            
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            self.documents = state['documents']
            self.document_map = state.get('document_map', {})  # Backward compatibility
            self.index = faiss.deserialize_index(state['index_data'])
            
            logger.info(f"Vector store loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def _generate_file_id(self, filename: str, filesize: int) -> str:
        """Generate unique file ID"""
        return hashlib.md5(f"{filename}_{filesize}".encode()).hexdigest()

    def add_documents(self, chunks: List[str], metadata: Dict) -> str:
        """Add document chunks to vector store"""
        try:
            file_id = self._generate_file_id(metadata['filename'], metadata['file_size'])
            
            # Remove existing document if present
            if file_id in self.document_map:
                self.remove_document(file_id)
            
            # Process chunks
            embeddings = []
            start_idx = len(self.documents)
            chunk_indices = []
            
            for i, chunk in enumerate(chunks):
                embedding = self.model.encode(chunk, convert_to_tensor=False)
                embeddings.append(embedding)
                
                self.documents.append({
                    'file_id': file_id,
                    'chunk_id': i,
                    'text': chunk,
                    'metadata': metadata
                })
                chunk_indices.append(start_idx + i)
            
            # Update index
            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                self.index.add(embeddings_array)
                
            # Update document map
            self.document_map[file_id] = chunk_indices
            
            # Save state after adding documents
            self.save()
            
            logger.info(f"Added document {metadata['filename']} with {len(chunks)} chunks")
            return file_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def remove_document(self, file_id: str) -> bool:
        """Remove document from vector store"""
        if file_id not in self.document_map:
            return False
            
        try:
            # Get indices to remove
            indices_to_remove = set(self.document_map[file_id])
            
            # Create new index and documents list
            new_index = faiss.IndexFlatL2(self.dimension)
            new_documents = []
            new_document_map = {}
            
            # Copy data excluding removed document
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
                embeddings_array = np.array(embeddings_to_keep).astype('float32')
                new_index.add(embeddings_array)
            
            # Update state
            self.index = new_index
            self.documents = new_documents
            self.document_map = new_document_map
            
            # Save state after removing document
            self.save()
            
            logger.info(f"Removed document {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {str(e)}")
            return False

    def get_document_info(self) -> List[Dict]:
        """Get information about stored documents"""
        try:
            unique_docs = {}
            for file_id, indices in self.document_map.items():
                if indices:  # If document has any chunks
                    doc = self.documents[indices[0]]
                    unique_docs[file_id] = {
                        'file_id': file_id,
                        'filename': doc['metadata']['filename'],
                        'file_size': doc['metadata']['file_size'],
                        'chunk_count': len(indices)
                    }
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return []

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                min(k, len(self.documents))
            )
            
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def has_documents(self) -> bool:
        """Check if there are any documents in the store"""
        return bool(self.documents and self.index.ntotal > 0)