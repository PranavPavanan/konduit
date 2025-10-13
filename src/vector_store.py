"""
FAISS vector store implementation for embeddings storage and retrieval
"""

import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    url: str
    text: str
    score: float
    chunk_index: int

class VectorStore:
    """FAISS-based vector store for embeddings"""
    
    def __init__(self, index_path: str = "vector_index", dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.is_trained = False
        
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index"""
        # Use IndexFlatIP for cosine similarity (after normalization)
        index = faiss.IndexFlatIP(self.dimension)
        return index
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def add_embeddings(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]):
        """Add embeddings and metadata to the index"""
        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_index()
        
        # Add embeddings to index
        self.index.add(normalized_embeddings.astype('float32'))
        
        # Store metadata
        self.chunks_metadata.extend(chunks_metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for similar embeddings"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        normalized_query = self._normalize_embeddings(query_embedding)
        
        # Search
        scores, indices = self.index.search(normalized_query.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                metadata = self.chunks_metadata[idx]
                result = SearchResult(
                    chunk_id=metadata.get('chunk_id', ''),
                    url=metadata.get('url', ''),
                    text=metadata.get('text', ''),
                    score=float(score),
                    chunk_index=int(idx)
                )
                results.append(result)
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save_index(self):
        """Save the index and metadata to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save metadata
            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            logger.info(f"Vector index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            raise
    
    def load_index(self) -> bool:
        """Load the index and metadata from disk"""
        try:
            faiss_path = f"{self.index_path}.faiss"
            pkl_path = f"{self.index_path}.pkl"
            
            if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
                logger.info("No existing index found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata
            with open(pkl_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            logger.info(f"Vector index loaded from {self.index_path} ({self.index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.index is None:
            return {"vector_count": 0, "dimension": self.dimension}
        
        return {
            "vector_count": self.index.ntotal,
            "dimension": self.dimension,
            "is_trained": self.is_trained
        }
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.chunks_metadata.clear()
        self.is_trained = False
        logger.info("Vector store cleared")
    
    def merge_consecutive_chunks(self, results: List[SearchResult], 
                               max_merge_distance: int = 2) -> List[SearchResult]:
        """Merge consecutive chunks from the same page for better context"""
        if not results:
            return results
        
        # Group results by URL
        url_groups = {}
        for result in results:
            url = result.url
            if url not in url_groups:
                url_groups[url] = []
            url_groups[url].append(result)
        
        merged_results = []
        
        for url, url_results in url_groups.items():
            # Sort by chunk index
            url_results.sort(key=lambda x: x.chunk_index)
            
            # Merge consecutive chunks
            current_group = [url_results[0]]
            
            for i in range(1, len(url_results)):
                current_chunk = url_results[i]
                last_chunk = current_group[-1]
                
                # Check if chunks are consecutive and from same page
                if (current_chunk.chunk_index - last_chunk.chunk_index <= max_merge_distance and
                    current_chunk.url == last_chunk.url):
                    current_group.append(current_chunk)
                else:
                    # Finalize current group
                    if len(current_group) > 1:
                        merged_result = self._merge_chunk_group(current_group)
                        merged_results.append(merged_result)
                    else:
                        merged_results.append(current_group[0])
                    
                    # Start new group
                    current_group = [current_chunk]
            
            # Add final group
            if len(current_group) > 1:
                merged_result = self._merge_chunk_group(current_group)
                merged_results.append(merged_result)
            else:
                merged_results.append(current_group[0])
        
        # Sort by score (highest first)
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Merged {len(results)} results into {len(merged_results)} groups")
        return merged_results
    
    def _merge_chunk_group(self, chunks: List[SearchResult]) -> SearchResult:
        """Merge a group of consecutive chunks"""
        if not chunks:
            return None
        
        # Use the highest scoring chunk as base
        base_chunk = max(chunks, key=lambda x: x.score)
        
        # Merge text content
        merged_text = " ".join([chunk.text for chunk in chunks])
        
        # Create merged result
        merged_result = SearchResult(
            chunk_id=f"{base_chunk.chunk_id}_merged",
            url=base_chunk.url,
            text=merged_text,
            score=base_chunk.score,  # Use highest score
            chunk_index=base_chunk.chunk_index
        )
        
        return merged_result
