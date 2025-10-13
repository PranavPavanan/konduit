"""
LLM service for answer generation with Ollama integration
"""

import logging
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from src.vector_store import SearchResult
from src.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_answer(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generate answer from question and context chunks"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass

class LLMService:
    """Main LLM service that manages different providers"""
    
    def __init__(self, model_name: str = "qwen3:4b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.providers = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Try to load Ollama with Qwen3 4B
        try:
            ollama_llm = OllamaProvider(model_name=self.model_name, base_url=self.ollama_url)
            if ollama_llm.is_available():
                self.providers.append(ollama_llm)
                logger.info(f"Loaded Ollama provider with {self.model_name}")
            else:
                logger.error(f"Failed to load Ollama provider with {self.model_name}")
                raise Exception(f"Ollama model '{self.model_name}' is not available")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise Exception(f"Ollama service is not running or model '{self.model_name}' is not installed. Please run: ollama pull {self.model_name}")
    
    def generate_answer(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generate answer using the best available provider"""
        if not self.providers:
            return {
                "answer": f"LLM service is not available. Please ensure Ollama is running and the '{self.model_name}' model is installed. Run: ollama pull {self.model_name}",
                "confidence": 0.0,
                "refusal_reason": "No LLM providers available"
            }
        
        for provider in self.providers:
            if provider.is_available():
                try:
                    return provider.generate_answer(question, context_chunks)
                except Exception as e:
                    logger.error(f"Provider {type(provider).__name__} failed: {e}")
                    continue
        
        # If all providers fail, return error
        return {
            "answer": f"I apologize, but I'm currently unable to process your question. Please ensure Ollama is running and the '{self.model_name}' model is installed. Run: ollama pull {self.model_name}",
            "confidence": 0.0,
            "refusal_reason": "All providers unavailable"
        }

