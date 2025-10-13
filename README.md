# RAG Service

A modular Retrieval-Augmented Generation service with web crawling and vector search capabilities.

## Quick Start

1. **Start the service:**
   ```cmd
   start_rag.bat
   ```

2. **Stop the service:**
   ```cmd
   stop_rag.bat
   ```

## API Endpoints

- **Health Check:** http://localhost:8000/health
- **API Documentation:** http://localhost:8000/docs
- **Ask Question:** http://localhost:8000/ask
- **Crawl Website:** http://localhost:8000/crawl
- **Index Content:** http://localhost:8000/index

## Manual Setup

1. **Activate virtual environment:**
   ```cmd
   .venv\Scripts\activate.bat
   ```

2. **Start Ollama:**
   ```cmd
   ollama serve
   ```

3. **Start RAG service:**
   ```cmd
   python main.py
   ```

## Requirements

- Python 3.8+
- Ollama with Qwen3:4b model
- Virtual environment with dependencies from `requirements.txt`