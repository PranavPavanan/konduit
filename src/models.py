"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .config import get_config

# Get configuration for default values
config = get_config()
crawler_config = config.get_crawler_config()
chunking_config = config.get_chunking_config()
embeddings_config = config.get_embeddings_config()
qa_config = config.get_qa_config()

class CrawlRequest(BaseModel):
    """Request model for crawling endpoint"""
    start_url: HttpUrl
    max_pages: int = Field(
        default=crawler_config.get('default_max_pages', 10), 
        ge=1, 
        le=100
    )
    max_depth: int = Field(
        default=crawler_config.get('default_max_depth', 2), 
        ge=1, 
        le=5
    )
    crawl_delay_ms: int = Field(
        default=crawler_config.get('default_delay_ms', 1000), 
        ge=crawler_config.get('min_delay_ms', 100), 
        le=crawler_config.get('max_delay_ms', 5000)
    )

class CrawlResponse(BaseModel):
    """Response model for crawling endpoint"""
    page_count: int
    skipped_count: int
    urls: List[str]
    errors: List[str] = []
    crawl_time_seconds: float

class IndexRequest(BaseModel):
    """Request model for indexing endpoint"""
    chunk_size: int = Field(
        default=chunking_config.get('default_chunk_size', 500), 
        ge=chunking_config.get('min_chunk_size', 100), 
        le=chunking_config.get('max_chunk_size', 1000)
    )
    chunk_overlap: int = Field(
        default=chunking_config.get('default_chunk_overlap', 50), 
        ge=0, 
        le=200
    )
    embedding_model: str = Field(
        default=embeddings_config.get('model_name', 'all-MiniLM-L6-v2')
    )

class IndexResponse(BaseModel):
    """Response model for indexing endpoint"""
    vector_count: int
    errors: List[str] = []
    index_time_seconds: float

class AskRequest(BaseModel):
    """Request model for ask endpoint"""
    question: str = Field(
        ..., 
        min_length=qa_config.get('min_question_length', 1), 
        max_length=qa_config.get('max_question_length', 1000)
    )
    top_k: int = Field(
        default=qa_config.get('default_top_k', 5), 
        ge=1, 
        le=10
    )
    min_relevance: float = Field(
        default=qa_config.get('default_min_relevance', 0.5), 
        ge=0.0, 
        le=1.0, 
        description="Minimum relevance threshold (0.0-1.0)"
    )

class Source(BaseModel):
    """Source citation model"""
    url: str
    snippet: str
    relevance_score: float

class Timings(BaseModel):
    """Timing metrics model"""
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class AskResponse(BaseModel):
    """Response model for ask endpoint"""
    answer: str
    sources: List[Source]
    timings: Timings
    confidence: Optional[float] = None
