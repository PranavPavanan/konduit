"""
Configuration management for RAG Service
Handles loading and validation of configuration from YAML files and environment variables
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for RAG Service"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file and environment variables"""
        # Load base configuration from YAML file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file {self.config_path}: {e}")
                self._config = {}
        else:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self._config = {}
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Set defaults for missing values
        self._set_defaults()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'RAG_LOG_LEVEL': ('logging.level', str),
            'RAG_LOG_FILE': ('logging.file', str),
            'RAG_HOST': ('app.host', str),
            'RAG_PORT': ('app.port', int),
            'RAG_DEBUG': ('app.debug', lambda x: x.lower() == 'true'),
            'RAG_LLM_MODEL': ('llm.model_name', str),
            'RAG_LLM_URL': ('llm.ollama_url', str),
            'RAG_EMBEDDING_MODEL': ('embeddings.model_name', str),
            'RAG_DATA_DIR': ('data.directory', str),
            'RAG_VECTOR_INDEX_PATH': ('data.vector_index_path', str),
            'RAG_CHUNK_SIZE': ('chunking.default_chunk_size', int),
            'RAG_CHUNK_OVERLAP': ('chunking.default_chunk_overlap', int),
            'RAG_MIN_RELEVANCE': ('qa.default_min_relevance', float),
            'RAG_TOP_K': ('qa.default_top_k', int),
            'RAG_MAX_PAGES': ('crawler.default_max_pages', int),
            'RAG_MAX_DEPTH': ('crawler.default_max_depth', int),
            'RAG_DELAY_MS': ('crawler.default_delay_ms', int),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    value = converter(env_value)
                    self._set_nested_value(config_path, value)
                    logger.debug(f"Set {config_path} = {value} from {env_var}")
                except Exception as e:
                    logger.warning(f"Failed to convert {env_var}={env_value}: {e}")
    
    def _set_nested_value(self, path: str, value: Any):
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _set_defaults(self):
        """Set default values for missing configuration keys"""
        defaults = {
            'app': {
                'title': 'RAG Service',
                'description': 'A modular Retrieval-Augmented Generation service',
                'version': '1.0.0',
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'data/rag_service.log',
                'console': True
            },
            'data': {
                'directory': 'data',
                'vector_index_path': 'data/vector_index',
                'vector_dimension': 384
            },
            'llm': {
                'model_name': 'qwen3:4b',
                'ollama_url': 'http://localhost:11434',
                'request_timeout': 30,
                'generation_options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                    'num_predict': 1024
                }
            },
            'embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'dimension': 384
            },
            'chunking': {
                'default_chunk_size': 500,
                'default_chunk_overlap': 50,
                'max_chunk_size': 1000,
                'min_chunk_size': 100
            },
            'crawler': {
                'default_max_pages': 10,
                'default_max_depth': 2,
                'default_delay_ms': 1000,
                'min_delay_ms': 100,
                'max_delay_ms': 5000,
                'user_agent': 'RAG-Service/1.0',
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            },
            'qa': {
                'default_min_relevance': 0.5,
                'default_top_k': 5,
                'max_question_length': 1000,
                'min_question_length': 1
            },
            'vector_store': {
                'similarity_threshold': 0.95,
                'max_merge_distance': 2,
                'enable_deduplication': True
            },
            'cors': {
                'allow_origins': ['*'],
                'allow_credentials': True,
                'allow_methods': ['*'],
                'allow_headers': ['*']
            },
            'development': {
                'enable_hot_reload': False,
                'log_level': 'DEBUG'
            }
        }
        
        self._merge_defaults(self._config, defaults)
    
    def _merge_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]):
        """Recursively merge defaults into config"""
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._merge_defaults(config[key], value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self.get(section, {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self.get_section('app')
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get_section('logging')
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data storage configuration"""
        return self.get_section('data')
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM service configuration"""
        return self.get_section('llm')
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration"""
        return self.get_section('embeddings')
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return self.get_section('chunking')
    
    def get_crawler_config(self) -> Dict[str, Any]:
        """Get crawler configuration"""
        return self.get_section('crawler')
    
    def get_qa_config(self) -> Dict[str, Any]:
        """Get Q&A service configuration"""
        return self.get_section('qa')
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        return self.get_section('vector_store')
    
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return self.get_section('cors')
    
    def get_development_config(self) -> Dict[str, Any]:
        """Get development configuration"""
        return self.get_section('development')
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary"""
        return self._config.copy()

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
