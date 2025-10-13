"""
RAG Service - Main FastAPI application
A modular Retrieval-Augmented Generation service with web crawling and vector search capabilities.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import time
import os
from contextlib import asynccontextmanager

from src.crawler import CrawlerService
from src.indexer import IndexerService
from src.qa_service import QAService
from src.vector_store import VectorStore
from src.llm_service import LLMService
from src.metrics import metrics_collector
from src.models import CrawlRequest, CrawlResponse, IndexRequest, IndexResponse, AskRequest, AskResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Global services
crawler_service = None
indexer_service = None
qa_service = None
vector_store = None
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global crawler_service, indexer_service, qa_service, vector_store, llm_service
    
    logger.info("Initializing RAG services...")
    
    # Initialize core services
    crawler_service = CrawlerService()
    vector_store = VectorStore()
    
    # Initialize Ollama LLM service with Qwen3 4B
    llm_service = LLMService(model_name="qwen3:4b", ollama_url="http://localhost:11434")
    
    # Initialize dependent services
    indexer_service = IndexerService(vector_store=vector_store)
    qa_service = QAService(vector_store, llm_service)
    
    # Try to load existing vector index
    if vector_store.load_index():
        logger.info("Loaded existing vector index")
    else:
        logger.info("No existing vector index found")
    
    logger.info("RAG services initialized successfully")
    yield
    
    logger.info("Shutting down RAG services...")

app = FastAPI(
    title="RAG Service",
    description="A modular Retrieval-Augmented Generation service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Service is running", "status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics and performance statistics"""
    return metrics_collector.get_all_metrics()

@app.get("/health")
async def health_check():
    """Detailed health check with service status"""
    try:
        # Check if services are initialized
        services_status = {
            "crawler": crawler_service is not None,
            "indexer": indexer_service is not None,
            "qa_service": qa_service is not None,
            "vector_store": vector_store is not None,
            "llm_service": llm_service is not None
        }
        
        # Check vector store status
        vector_stats = vector_store.get_stats() if vector_store else {"vector_count": 0}
        
        # Check LLM availability and get model info
        llm_available = False
        llm_info = {"status": "not_available"}
        if llm_service and hasattr(llm_service, 'providers') and llm_service.providers:
            llm_available = llm_service.providers[0].is_available()
            if hasattr(llm_service.providers[0], 'get_model_info'):
                llm_info = llm_service.providers[0].get_model_info()
        
        return {
            "status": "healthy",
            "services": services_status,
            "vector_store": vector_stats,
            "llm_available": llm_available,
            "llm_info": llm_info,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """Crawl a website and extract content"""
    try:
        logger.info(f"Starting crawl for {request.start_url}")
        result = await crawler_service.crawl(
            start_url=str(request.start_url),
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            crawl_delay_ms=request.crawl_delay_ms
        )
        
        # Store crawled pages in indexer for later indexing
        crawled_pages = crawler_service.get_crawled_pages()
        indexer_service.set_crawled_pages(crawled_pages)
        
        logger.info(f"Crawl completed: {result['page_count']} pages processed")
        return CrawlResponse(**result)
    except Exception as e:
        logger.error(f"Crawl failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=IndexResponse)
async def index_content(request: IndexRequest):
    """Index crawled content into vector store"""
    try:
        logger.info("Starting content indexing")
        result = await indexer_service.index_content(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model
        )
        logger.info(f"Indexing completed: {result['vector_count']} vectors created")
        return IndexResponse(**result)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question and get an answer with sources"""
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        result = await qa_service.ask_question(
            question=request.question,
            top_k=request.top_k
        )
        logger.info(f"Question answered in {result['timings']['total_ms']}ms")
        return AskResponse(**result)
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )