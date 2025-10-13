"""
Metrics and observability module for RAG service
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrawlMetrics:
    """Metrics for crawling operations"""
    pages_crawled: int = 0
    pages_skipped: int = 0
    total_crawl_time: float = 0.0
    avg_page_time: float = 0.0
    errors: List[str] = field(default_factory=list)

@dataclass
class IndexMetrics:
    """Metrics for indexing operations"""
    chunks_created: int = 0
    vectors_added: int = 0
    total_index_time: float = 0.0
    avg_chunk_time: float = 0.0
    errors: List[str] = field(default_factory=list)

@dataclass
class QAMetrics:
    """Metrics for Q&A operations"""
    questions_answered: int = 0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    refusal_rate: float = 0.0
    errors: List[str] = field(default_factory=list)

class MetricsCollector:
    """Collects and aggregates metrics for the RAG service"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_history: deque = deque(maxlen=max_history)
        self.crawl_metrics = CrawlMetrics()
        self.index_metrics = IndexMetrics()
        self.qa_metrics = QAMetrics()
        
        # Performance tracking
        self.retrieval_times: deque = deque(maxlen=100)
        self.generation_times: deque = deque(maxlen=100)
        self.total_times: deque = deque(maxlen=100)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    def start_request(self, request_id: str, endpoint: str) -> RequestMetrics:
        """Start tracking a new request"""
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            start_time=time.time()
        )
        return metrics
    
    def end_request(self, metrics: RequestMetrics, status: str = "success", 
                   error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """End tracking a request"""
        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.status = status
        metrics.error = error
        
        if metadata:
            metrics.metadata.update(metadata)
        
        self.request_history.append(metrics)
        
        # Track errors
        if error:
            self.error_counts[error] += 1
        
        logger.info(f"Request {metrics.request_id} completed in {metrics.duration_ms:.2f}ms with status {status}")
    
    def record_crawl_metrics(self, pages_crawled: int, pages_skipped: int, 
                           total_time: float, errors: List[str] = None):
        """Record crawling metrics"""
        self.crawl_metrics.pages_crawled += pages_crawled
        self.crawl_metrics.pages_skipped += pages_skipped
        self.crawl_metrics.total_crawl_time += total_time
        
        if pages_crawled > 0:
            self.crawl_metrics.avg_page_time = self.crawl_metrics.total_crawl_time / pages_crawled
        
        if errors:
            self.crawl_metrics.errors.extend(errors)
            for error in errors:
                self.error_counts[error] += 1
        
        logger.info(f"Crawl metrics: {pages_crawled} pages crawled, {pages_skipped} skipped in {total_time:.2f}s")
    
    def record_index_metrics(self, chunks_created: int, vectors_added: int, 
                           total_time: float, errors: List[str] = None):
        """Record indexing metrics"""
        self.index_metrics.chunks_created += chunks_created
        self.index_metrics.vectors_added += vectors_added
        self.index_metrics.total_index_time += total_time
        
        if chunks_created > 0:
            self.index_metrics.avg_chunk_time = self.index_metrics.total_index_time / chunks_created
        
        if errors:
            self.index_metrics.errors.extend(errors)
            for error in errors:
                self.error_counts[error] += 1
        
        logger.info(f"Index metrics: {chunks_created} chunks created, {vectors_added} vectors added in {total_time:.2f}s")
    
    def record_qa_metrics(self, retrieval_time: float, generation_time: float, 
                         total_time: float, was_refusal: bool = False, error: str = None):
        """Record Q&A metrics"""
        self.qa_metrics.questions_answered += 1
        
        # Track timing metrics
        self.retrieval_times.append(retrieval_time)
        self.generation_times.append(generation_time)
        self.total_times.append(total_time)
        
        # Update averages
        self.qa_metrics.avg_retrieval_time = statistics.mean(self.retrieval_times)
        self.qa_metrics.avg_generation_time = statistics.mean(self.generation_times)
        self.qa_metrics.avg_total_time = statistics.mean(self.total_times)
        
        # Update refusal rate
        if was_refusal:
            refusals = sum(1 for req in self.request_history 
                          if req.endpoint == "/ask" and req.metadata.get("was_refusal", False))
            self.qa_metrics.refusal_rate = refusals / max(1, self.qa_metrics.questions_answered)
        
        if error:
            self.qa_metrics.errors.append(error)
            self.error_counts[error] += 1
        
        logger.info(f"QA metrics: retrieval={retrieval_time:.2f}ms, generation={generation_time:.2f}ms, total={total_time:.2f}ms")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.total_times:
            return {"message": "No performance data available"}
        
        return {
            "retrieval_times": {
                "p50": self._percentile(self.retrieval_times, 50),
                "p95": self._percentile(self.retrieval_times, 95),
                "avg": statistics.mean(self.retrieval_times),
                "min": min(self.retrieval_times),
                "max": max(self.retrieval_times)
            },
            "generation_times": {
                "p50": self._percentile(self.generation_times, 50),
                "p95": self._percentile(self.generation_times, 95),
                "avg": statistics.mean(self.generation_times),
                "min": min(self.generation_times),
                "max": max(self.generation_times)
            },
            "total_times": {
                "p50": self._percentile(self.total_times, 50),
                "p95": self._percentile(self.total_times, 95),
                "avg": statistics.mean(self.total_times),
                "min": min(self.total_times),
                "max": max(self.total_times)
            }
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        recent_requests = list(self.request_history)[-10:]  # Last 10 requests
        
        return {
            "crawl_metrics": {
                "pages_crawled": self.crawl_metrics.pages_crawled,
                "pages_skipped": self.crawl_metrics.pages_skipped,
                "total_crawl_time": self.crawl_metrics.total_crawl_time,
                "avg_page_time": self.crawl_metrics.avg_page_time,
                "error_count": len(self.crawl_metrics.errors)
            },
            "index_metrics": {
                "chunks_created": self.index_metrics.chunks_created,
                "vectors_added": self.index_metrics.vectors_added,
                "total_index_time": self.index_metrics.total_index_time,
                "avg_chunk_time": self.index_metrics.avg_chunk_time,
                "error_count": len(self.index_metrics.errors)
            },
            "qa_metrics": {
                "questions_answered": self.qa_metrics.questions_answered,
                "avg_retrieval_time": self.qa_metrics.avg_retrieval_time,
                "avg_generation_time": self.qa_metrics.avg_generation_time,
                "avg_total_time": self.qa_metrics.avg_total_time,
                "refusal_rate": self.qa_metrics.refusal_rate,
                "error_count": len(self.qa_metrics.errors)
            },
            "performance_stats": self.get_performance_stats(),
            "error_counts": dict(self.error_counts),
            "recent_requests": [
                {
                    "request_id": req.request_id,
                    "endpoint": req.endpoint,
                    "duration_ms": req.duration_ms,
                    "status": req.status,
                    "error": req.error
                } for req in recent_requests
            ]
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.request_history.clear()
        self.crawl_metrics = CrawlMetrics()
        self.index_metrics = IndexMetrics()
        self.qa_metrics = QAMetrics()
        self.retrieval_times.clear()
        self.generation_times.clear()
        self.total_times.clear()
        self.error_counts.clear()
        logger.info("Metrics reset")

# Global metrics collector instance
metrics_collector = MetricsCollector()
