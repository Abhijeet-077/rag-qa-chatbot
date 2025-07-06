"""
Configuration module for RAG QA Chatbot
Manages environment variables and system settings
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration settings for the RAG system"""
    
    # Required fields first
    openai_api_key: str = None
    pinecone_api_key: str = None
    pinecone_environment: str = None
    pinecone_index_name: str = None
    
    # OpenAI Configuration
    openai_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    max_tokens: int = 4000
    temperature: float = 0.3
    
    # RAG System Configuration
    top_k_results: int = 5
    confidence_threshold: float = 0.7
    max_context_length: int = 8000
    
    # Prompt Configuration
    system_role: str = "domain-aware business assistant"
    response_style: str = "professional and concise"
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        
        # Check for required environment variables but don't fail if missing
        required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'PINECONE_INDEX_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            print("Please set these in your .env file before running the system.")
        
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large'),
            embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', '3072')),
            max_tokens=int(os.getenv('MAX_TOKENS', '4000')),
            temperature=float(os.getenv('TEMPERATURE', '0.3')),
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_environment=os.getenv('PINECONE_ENVIRONMENT'),
            pinecone_index_name=os.getenv('PINECONE_INDEX_NAME'),
            top_k_results=int(os.getenv('TOP_K_RESULTS', '5')),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
            max_context_length=int(os.getenv('MAX_CONTEXT_LENGTH', '8000'))
        )

# Global configuration instance
config = RAGConfig.from_env()
