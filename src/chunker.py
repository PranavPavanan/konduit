"""
Content chunking and text processing module
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Data class for text chunks"""
    text: str
    url: str
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int

class ChunkerService:
    """Service for semantic chunking of text content"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}")
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks(self, text: str, url: str, chunk_size: int = 500, 
                      chunk_overlap: int = 50) -> List[TextChunk]:
        """Create overlapping text chunks"""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        start_char = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    url=url,
                    chunk_id=f"{url}_{chunk_id}",
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
                start_char = end_char - len(' '.join(overlap_sentences))
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_char = start_char + len(chunk_text)
            
            chunks.append(TextChunk(
                text=chunk_text,
                url=url,
                chunk_id=f"{url}_{chunk_id}",
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens
            ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap from the end of current chunk"""
        if not sentences or overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def chunk_pages(self, pages: List[Dict[str, Any]], chunk_size: int = 500, 
                   chunk_overlap: int = 50) -> List[TextChunk]:
        """Chunk multiple pages into text chunks"""
        all_chunks = []
        
        for page in pages:
            try:
                chunks = self._create_chunks(
                    page['content'], 
                    page['url'], 
                    chunk_size, 
                    chunk_overlap
                )
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks for {page['url']}")
            except Exception as e:
                logger.error(f"Failed to chunk page {page['url']}: {e}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_embeddings(self, chunks: List[TextChunk]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        self._load_model()
        
        texts = [chunk.text for chunk in chunks]
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        self._load_model()
        
        try:
            embedding = self.model.encode([query])
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
