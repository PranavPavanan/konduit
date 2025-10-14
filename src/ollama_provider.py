"""
Ollama provider for Qwen3 4B model
"""

import logging
import requests
import subprocess
from typing import List, Dict, Any, Optional
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 30

class OllamaProvider:
    """Ollama provider for Qwen3 4B model"""
    
    def __init__(self, model_name: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.is_available_flag = None
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available"""
        if self.is_available_flag is not None:
            return self.is_available_flag
        try:
            response = self.session.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            model_found = any(self.model_name in name or name in self.model_name for name in model_names)
            if not model_found:
                logger.warning("Model %s not found in Ollama registry", self.model_name)
            self.is_available_flag = model_found
            return model_found
        except requests.RequestException as exc:
            logger.error("Failed to check Ollama availability: %s", exc)
            self.is_available_flag = False
            return False

    def _pull_model(self):
        """Pull the model if it's not available"""
        try:
            logger.info("Pulling model %s from Ollama…", self.model_name)
            subprocess.run(["ollama", "pull", self.model_name], check=True)
            self.is_available_flag = None
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to pull model %s: %s", self.model_name, exc)

    def generate_answer(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generate answer from question and context chunks using Ollama"""
        if not self.is_available():
            self._pull_model()
            if not self.is_available():
                return {
                    "answer": f"Ollama model '{self.model_name}' is unavailable; run `ollama pull {self.model_name}`.",
                    "confidence": 0.0,
                    "refusal_reason": "Model unavailable"
                }
        context = self._build_context(context_chunks)
        prompt = self._create_qwen_prompt(question, context)
        try:
            logger.info(f"Context built with {len(context_chunks)} chunks, length: {len(context)}")
            logger.info(f"Prompt created, length: {len(prompt)}")
            logger.debug(f"Prompt preview: {prompt[:500]}...")
            
            response = self.session.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        "num_predict": 1024  # Increased token limit
                    }
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            
            # Extract response, use thinking field if response is empty
            raw_response = payload.get("response", "").strip()
            if not raw_response and payload.get("thinking"):
                raw_response = payload.get("thinking", "").strip()
                logger.info("Using thinking field as response")
            
            logger.info(f"Raw response from Ollama: '{raw_response}'")
            logger.info(f"Response length: {len(raw_response)}")
            
            answer = self._clean_answer(raw_response)
            logger.info(f"Cleaned answer: '{answer}'")
            
            confidence = 0.0 if self._is_refusal(answer) else self._calculate_confidence(context_chunks)
            return {"answer": answer, "confidence": confidence}
        except requests.RequestException as exc:
            logger.error("Ollama generation failed: %s", exc)
            return {
                "answer": "I encountered an error while generating the answer.",
                "confidence": 0.0,
                "refusal_reason": "Generation error"
            }

    def _build_context(self, context_chunks: List[SearchResult]) -> str:
        """Build context string from search results"""
        if not context_chunks:
            return ""
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            snippet = chunk.text.strip().replace("\n", " ")
            context_parts.append(f"[Source {idx}] {snippet}\nURL: {chunk.url}")
        return "\n\n".join(context_parts)

    def _create_qwen_prompt(self, question: str, context: str) -> str:
        """Create a simple prompt for answer generation"""
        if context:
            prompt = f"""Based on the following context, answer the question. If you cannot answer based on the context, say "I cannot answer this question based on the provided information."

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Answer the following question. If you cannot answer it, say "I cannot answer this question."

Question: {question}

Answer:"""
        return prompt

    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer"""
        answer = answer.replace("Answer:", "").replace("<|im_end|>", "").strip()
        if answer and answer[-1] not in ".!?":
            last_period = answer.rfind(".")
            if last_period != -1:
                answer = answer[: last_period + 1]
        return answer

    def _is_refusal(self, answer: str) -> bool:
        """Check if the answer is a refusal"""
        refusal_indicators = [
            "cannot answer",
            "don't have enough information",
            "not enough information",
            "cannot find",
            "unable to answer",
            "insufficient information",
            "based on the provided information"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in refusal_indicators)
    
    def _calculate_confidence(self, context_chunks: List[SearchResult]) -> float:
        """Calculate confidence based on context relevance scores"""
        if not context_chunks:
            return 0.0
        
        # Average the relevance scores
        avg_score = sum(chunk.score for chunk in context_chunks) / len(context_chunks)
        
        # Normalize to 0-1 range (assuming scores are 0-1)
        return min(avg_score, 1.0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_available():
            return {}
        try:
            response = self.session.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.error("Failed to fetch model info: %s", exc)
            return {}

    def test_connection(self) -> bool:
        """Test the connection to Ollama"""
        try:
            self.session.get(f"{self.api_url}/version", timeout=5).raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Ollama connection test failed: %s", exc)
            return False
