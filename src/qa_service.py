"""
Q&A service that coordinates retrieval and generation
"""

import logging
import time
from typing import List, Dict, Any
from src.vector_store import VectorStore, SearchResult
from src.llm_service import LLMService
from src.models import Source, Timings

logger = logging.getLogger(__name__)

class QAService:
    """Service for question answering with retrieval and generation"""
    
    def __init__(self, vector_store: VectorStore = None, llm_service: LLMService = None):
        self.vector_store = vector_store or VectorStore()
        self.llm_service = llm_service or LLMService()
        
    async def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using retrieval and generation"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            search_results = await self._retrieve_relevant_chunks(question, top_k)
            retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to ms
            
            # Step 2: Generate answer
            generation_start = time.time()
            answer_result = self.llm_service.generate_answer(question, search_results)
            generation_time = (time.time() - generation_start) * 1000  # Convert to ms
            
            # Step 3: Format response
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create sources with snippets
            sources = [source.dict() for source in self._create_sources(search_results)]
            
            # Create timings
            timings = Timings(
                retrieval_ms=retrieval_time,
                generation_ms=generation_time,
                total_ms=total_time
            )
            
            response = {
                "answer": answer_result["answer"],
                "sources": sources,
                "timings": timings.dict(),
                "confidence": answer_result.get("confidence", 0.0)
            }
            
            # Add refusal reason if applicable
            if "refusal_reason" in answer_result:
                response["refusal_reason"] = answer_result["refusal_reason"]
            
            logger.info(f"Question answered in {total_time:.2f}ms (retrieval: {retrieval_time:.2f}ms, generation: {generation_time:.2f}ms)")
            
            return response
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question.",
                "sources": [],
                "timings": Timings(retrieval_ms=0, generation_ms=0, total_ms=0).dict(),
                "confidence": 0.0,
                "refusal_reason": "Processing error"
            }
    
    async def _retrieve_relevant_chunks(self, question: str, top_k: int) -> List[SearchResult]:
        """Retrieve relevant chunks for the question"""
        try:
            # Generate query embedding using the LLM service's chunker
            from src.chunker import ChunkerService
            chunker = ChunkerService()
            query_embedding = chunker.get_query_embedding(question)
            
            # Search for similar chunks
            search_results = self.vector_store.search(query_embedding, top_k)
            
            # Optionally merge consecutive chunks for better context
            if len(search_results) > 1:
                search_results = self.vector_store.merge_consecutive_chunks(search_results)
            
            logger.info(f"Retrieved {len(search_results)} relevant chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _create_sources(self, search_results: List[SearchResult]) -> List[Source]:
        """Create source citations from search results"""
        sources = []
        
        for result in search_results:
            # Create snippet (first 200 characters of the text)
            snippet = result.text[:200]
            if len(result.text) > 200:
                snippet += "..."
            
            source = Source(
                url=result.url,
                snippet=snippet,
                relevance_score=result.score
            )
            sources.append(source)
        
        return sources
    
    def set_vector_store(self, vector_store: VectorStore):
        """Set the vector store for retrieval"""
        self.vector_store = vector_store
    
    def set_llm_service(self, llm_service: LLMService):
        """Set the LLM service for generation"""
        self.llm_service = llm_service
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "llm_available": self.llm_service.providers[0].is_available() if self.llm_service.providers else False
        }
