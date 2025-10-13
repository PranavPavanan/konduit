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

class MockLLM(LLMProvider):
    """Mock LLM for testing purposes"""
    
    def is_available(self) -> bool:
        return True
    
    def generate_answer(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generate a mock answer"""
        if not context_chunks:
            return {
                "answer": "I cannot answer this question based on the provided information.",
                "confidence": 0.0,
                "refusal_reason": "No context provided"
            }
        
        # Simple mock response
        context_text = " ".join([chunk.text[:100] for chunk in context_chunks[:2]])
        answer = f"Based on the provided information: {context_text[:200]}..."
        
        return {
            "answer": answer,
            "confidence": 0.8,
            "generation_time": 0.1
        }

class LLMService:
    """Main LLM service that manages different providers"""
    
    def __init__(self, model_name: str = "qwen3:4b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.providers = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Try to load Ollama with Qwen2.5 4B first
        try:
            ollama_llm = OllamaProvider(model_name=self.model_name, base_url=self.ollama_url)
            if ollama_llm.is_available():
                self.providers.append(ollama_llm)
                logger.info(f"Loaded Ollama provider with {self.model_name}")
            else:
                logger.warning(f"Failed to load Ollama provider with {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")
        
        # Add mock provider as fallback
        self.providers.append(MockLLM())
        logger.info("Added mock LLM provider as fallback")
    
    def generate_answer(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generate answer using the best available provider"""
        for provider in self.providers:
            if provider.is_available():
                try:
                    return provider.generate_answer(question, context_chunks)
                except Exception as e:
                    logger.error(f"Provider {type(provider).__name__} failed: {e}")
                    continue
        
        # If all providers fail, return error
        return {
            "answer": "I apologize, but I'm currently unable to process your question due to technical difficulties.",
            "confidence": 0.0,
            "refusal_reason": "All providers unavailable"
        }

