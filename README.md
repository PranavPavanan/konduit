# Konduit RAG

A Retrieval-Augmented Generation (RAG) system that crawls websites, indexes content, and answers questions with source citations.

## Features

- **Intelligent Web Crawling**: Hybrid approach with automatic SPA/static page detection using BeautifulSoup and readability extraction
- **Modern Content Support**: Handles HTML pages and text files with content extraction and cleaning
- **Local-First Embeddings**: Uses sentence-transformers for fast, private embedding generation
- **Flexible LLM Integration**: Supports Ollama for local models with configurable parameters
- **Persistent Storage**: FAISS vector database with pickle serialization for easy development and testing
- **Multiple Interfaces**: REST API with comprehensive health checks and monitoring
- **Configurable Pipeline**: YAML-based configuration for all components

## Quick Start

### Prerequisites

1. **Install Python 3.8+** and create virtual environment:
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

2. **Install Ollama** (required for LLM):
   
   Download and install from https://ollama.com/download/windows
   
   Start Ollama service:
   ```cmd
   ollama serve
   ```
   
   Pull the required model:
   ```cmd
   ollama pull qwen3:4b
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

### Usage

**Option 1: Quick Start (Windows)**
```cmd
# Start the service
start_rag.bat

# Stop the service  
stop_rag.bat
```

**Option 2: Manual Start**
```cmd
# Activate virtual environment
venv\Scripts\activate.bat

# Start Ollama (in separate terminal)
ollama serve

# Start RAG service
python main.py
```

**API Usage:**
- **Health Check:** http://localhost:8000/health
- **API Documentation:** http://localhost:8000/docs
- **Ask Question:** POST http://localhost:8000/ask
- **Crawl Website:** POST http://localhost:8000/crawl
- **Index Content:** POST http://localhost:8000/index

## Architecture

### Pipeline Overview
```
URL → Crawler → Chunker → Embedder → Vector Store
                                           ↓
User Question → Embedder → Retriever → LLM → Answer + Citations
```

### Key Design Decisions

**Crawling**
- **BeautifulSoup + Readability**: Fast parsing for traditional HTML pages with content extraction
- **Configurable Delays**: Respectful crawling with configurable delays and user agents
- **Content Cleaning**: Automatic removal of navigation, ads, and boilerplate content
- **Tradeoff**: Focus on content quality over JavaScript-heavy SPA support

**Embeddings**
- **Sentence-Transformers**: Local embedding model (all-MiniLM-L6-v2) for privacy and cost savings
- **No API Costs**: All embedding generation happens locally
- **384-Dimensional**: Optimized for speed and memory usage
- **Tradeoff**: Requires local compute but eliminates API dependencies

**LLM Generation**
- **Ollama Integration**: Local LLM (qwen3:4b) with configurable generation parameters
- **Simple Configuration**: YAML-based configuration for model parameters
- **Local & Private**: All processing happens locally
- **Tradeoff**: Local compute requirements but complete privacy

**Vector Storage**
- **FAISS + Pickle**: Persistent storage in local files
- **No Server Required**: Perfect for development and testing
- **Easy Scaling Path**: Can migrate to distributed FAISS for production
- **Tradeoff**: Single-machine limitation but simpler deployment

**Chunking**
- **Configurable Sizes**: Default 500 chars with 50 char overlap
- **Content-Aware**: Preserves sentence boundaries when possible
- **Metadata Rich**: Preserves URLs, titles, and page structure
- **Tradeoff**: Fixed-size chunks vs semantic chunking - optimized for simplicity

**Retrieval**
- **Cosine Similarity**: FAISS-based similarity search with configurable threshold
- **Top-K Results**: Configurable number of results (default 5)
- **Relevance Filtering**: Minimum similarity threshold for quality control
- **Tradeoff**: Simple retrieval vs advanced methods - optimized for speed and reliability

## Project Structure

```
cursorkonduit/
├── src/                    # Core RAG components
│   ├── crawler.py         # Web crawler with content extraction
│   ├── chunker.py         # Text chunking and preprocessing
│   ├── indexer.py         # Content indexing and vector storage
│   ├── vector_store.py    # FAISS vector database wrapper
│   ├── qa_service.py      # Question answering pipeline
│   ├── llm_service.py     # Ollama LLM integration
│   ├── ollama_provider.py # Ollama API client
│   ├── models.py          # Pydantic data models
│   └── config.py          # Configuration management
├── data/                   # Runtime data (crawled pages, vectors)
├── tests/                  # Unit tests
├── config.yaml            # YAML configuration
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── start_rag.bat         # Windows start script
├── stop_rag.bat          # Windows stop script
└── README.md             # This file
```

## Configuration

Key settings in `config.yaml`:

```yaml
# LLM Configuration
llm:
  model_name: "qwen3:4b"
  ollama_url: "http://localhost:11434"
  generation_options:
    temperature: 0.7
    top_p: 0.9
    top_k: 40
    repeat_penalty: 1.1
    num_predict: 1024

# Embedding Configuration
embeddings:
  model_name: "all-MiniLM-L6-v2"
  dimension: 384

# Chunking Configuration
chunking:
  default_chunk_size: 500
  default_chunk_overlap: 50
  max_chunk_size: 1000
  min_chunk_size: 100

# Crawler Configuration
crawler:
  default_max_pages: 10
  default_max_depth: 2
  default_delay_ms: 1000
  user_agent: "RAG-Service/1.0"

# Q&A Configuration
qa:
  default_min_relevance: 0.5
  default_top_k: 5
  max_question_length: 1000
  min_question_length: 1

# Vector Store Configuration
vector_store:
  similarity_threshold: 0.95
  max_merge_distance: 2
  enable_deduplication: true
```

### Customizing the Models

**LLM Model:**
- Change `llm.model_name` to any Ollama model
- Popular alternatives: `llama3:8b`, `mistral:7b`, `codellama:7b`
- See available models: `ollama list`

**Embedding Model:**
- Change `embeddings.model_name` to any sentence-transformers model
- Popular alternatives: `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`
- See available models: [Hugging Face Hub](https://huggingface.co/sentence-transformers)

## API Examples

### Crawl a Website
```bash
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://docs.example.com",
    "max_pages": 10,
    "max_depth": 2,
    "crawl_delay_ms": 1000
  }'
```

### Index Crawled Content
```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "all-MiniLM-L6-v2"
  }'
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the pricing?",
    "top_k": 5,
    "min_relevance": 0.5
  }'
```

## Development

### Running Tests
```cmd
python -m pytest tests/
```

### Health Monitoring
The service provides comprehensive health checks at `/health` including:
- Service initialization status
- Vector store statistics
- LLM availability and model information
- System timestamps

### Logging
- Console and file logging (configurable in `config.yaml`)
- Log file: `data/rag_service.log`
- Configurable log levels: DEBUG, INFO, WARNING, ERROR

## Requirements

- Python 3.8+
- Ollama with qwen3:4b model (or any compatible model)
- Virtual environment with dependencies from `requirements.txt`
- Windows (batch scripts provided) or manual setup on other platforms