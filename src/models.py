"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class CrawlRequest(BaseModel):
    """Request model for crawling endpoint"""
    start_url: HttpUrl
    max_pages: int = Field(default=50, ge=1, le=100)
    max_depth: int = Field(default=3, ge=1, le=5)
    crawl_delay_ms: int = Field(default=1000, ge=100, le=5000)

class CrawlResponse(BaseModel):
    """Response model for crawling endpoint"""
    page_count: int
    skipped_count: int
    urls: List[str]
    errors: List[str] = []
    crawl_time_seconds: float

class IndexRequest(BaseModel):
    """Request model for indexing endpoint"""
    chunk_size: int = Field(default=500, ge=100, le=1000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

class IndexResponse(BaseModel):
    """Response model for indexing endpoint"""
    vector_count: int
    errors: List[str] = []
    index_time_seconds: float

class AskRequest(BaseModel):
    """Request model for ask endpoint"""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

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
