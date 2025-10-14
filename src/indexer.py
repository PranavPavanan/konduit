"""
Indexer service that coordinates content chunking and vector storage
"""

import logging
import time
from typing import List, Dict, Any
from src.chunker import ChunkerService, TextChunk
from src.vector_store import VectorStore
from src.crawler import CrawledPage

logger = logging.getLogger(__name__)

class IndexerService:
    """Service for indexing crawled content into vector store"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vector_store: VectorStore = None):
        self.chunker = ChunkerService(embedding_model)
        self.vector_store = vector_store or VectorStore()
        self.embedding_model = embedding_model
        self._crawled_pages = []
        
    async def index_content(self, chunk_size: int = 500, chunk_overlap: int = 50, 
                           embedding_model: str = None, clear_existing: bool = False) -> Dict[str, Any]:
        """Index crawled content into vector store"""
        start_time = time.time()
        errors = []
        
        try:
            # Clear existing index if requested
            if clear_existing:
                logger.info("Clearing existing vector index...")
                self.vector_store.clear_and_rebuild()
            
            # Update embedding model if specified
            if embedding_model and embedding_model != self.embedding_model:
                self.chunker = ChunkerService(embedding_model)
                self.embedding_model = embedding_model
            
            # Get crawled pages from crawler service
            # This would typically be injected or accessed via a shared state
            # For now, we'll assume pages are passed in somehow
            pages = self._get_crawled_pages()
            
            if not pages:
                logger.warning("No crawled pages found to index")
                return {
                    "vector_count": 0,
                    "errors": ["No crawled pages found"],
                    "index_time_seconds": time.time() - start_time
                }
            
            logger.info(f"Starting indexing of {len(pages)} pages")
            
            # Convert crawled pages to dict format for chunker
            page_dicts = []
            for page in pages:
                page_dicts.append({
                    'url': page.url,
                    'title': page.title,
                    'content': page.content
                })
            
            # Create text chunks
            logger.info("Creating text chunks...")
            chunks = self.chunker.chunk_pages(page_dicts, chunk_size, chunk_overlap)
            
            if not chunks:
                logger.warning("No chunks created from pages")
                return {
                    "vector_count": 0,
                    "errors": ["No chunks created from pages"],
                    "index_time_seconds": time.time() - start_time
                }
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.chunker.get_embeddings(chunks)
            
            # Prepare metadata for vector store
            chunks_metadata = []
            for chunk in chunks:
                chunks_metadata.append({
                    'chunk_id': chunk.chunk_id,
                    'url': chunk.url,
                    'text': chunk.text,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'token_count': chunk.token_count
                })
            
            # Add to vector store
            logger.info("Adding to vector store...")
            self.vector_store.add_embeddings(embeddings, chunks_metadata)
            
            # Save index to disk
            logger.info("Saving index to disk...")
            self.vector_store.save_index()
            
            index_time = time.time() - start_time
            
            logger.info(f"Indexing completed: {len(chunks)} chunks indexed in {index_time:.2f}s")
            
            return {
                "vector_count": len(chunks),
                "errors": errors,
                "index_time_seconds": index_time
            }
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            errors.append(str(e))
            return {
                "vector_count": 0,
                "errors": errors,
                "index_time_seconds": time.time() - start_time
            }
    
    def _get_crawled_pages(self) -> List[CrawledPage]:
        """Get crawled pages from crawler service"""
        return self._crawled_pages
    
    def set_crawled_pages(self, pages: List[CrawledPage]):
        """Set crawled pages for indexing"""
        self._crawled_pages = pages
    
    def load_existing_index(self) -> bool:
        """Load existing vector index from disk"""
        return self.vector_store.load_index()
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_stats()
    
    def clear_index(self):
        """Clear the vector index"""
        self.vector_store.clear()
        logger.info("Vector index cleared")
