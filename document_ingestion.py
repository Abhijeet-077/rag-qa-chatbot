"""
Document Ingestion System for RAG QA Chatbot
Handles loading, processing, and indexing of business documents
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document for indexing"""
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_id: str
    embedding: Optional[List[float]] = None

class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model(config.openai_model)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _create_chunk_id(self, content: str, source: str) -> str:
        """Create unique ID for a chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{Path(source).stem}_{content_hash}"
    
    def chunk_text(self, text: str, source: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks suitable for vector indexing
        """
        if metadata is None:
            metadata = {}
        
        # Clean and normalize text
        text = text.strip().replace('\n\n', '\n').replace('\r', '')
        
        # Calculate chunk boundaries
        chunks = []
        words = text.split()
        
        start = 0
        while start < len(words):
            # Determine end of chunk
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            # Ensure we don't exceed token limits
            while self._count_tokens(chunk_content) > self.chunk_size and len(chunk_words) > 50:
                chunk_words = chunk_words[:-10]  # Remove 10 words at a time
                chunk_content = ' '.join(chunk_words)
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                'source': source,
                'chunk_index': len(chunks),
                'token_count': self._count_tokens(chunk_content),
                'word_count': len(chunk_words)
            }
            
            # Create chunk
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                source=source,
                chunk_id=self._create_chunk_id(chunk_content, source)
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.overlap
            if start >= len(words):
                break
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def process_text_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Process a text file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if metadata is None:
                metadata = {
                    'file_type': 'text',
                    'file_name': Path(file_path).name,
                    'file_size': os.path.getsize(file_path)
                }
            
            return self.chunk_text(content, file_path, metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_csv_file(self, file_path: str, text_column: str, metadata_columns: List[str] = None) -> List[DocumentChunk]:
        """Process CSV file into chunks"""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for idx, row in df.iterrows():
                content = str(row[text_column])
                
                # Build metadata from specified columns
                chunk_metadata = {
                    'file_type': 'csv',
                    'file_name': Path(file_path).name,
                    'row_index': idx
                }
                
                if metadata_columns:
                    for col in metadata_columns:
                        if col in row:
                            chunk_metadata[col] = row[col]
                
                chunk_chunks = self.chunk_text(content, f"{file_path}:row_{idx}", chunk_metadata)
                chunks.extend(chunk_chunks)
            
            logger.info(f"Processed {len(chunks)} chunks from CSV {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return []
    
    def process_json_file(self, file_path: str, text_fields: List[str]) -> List[DocumentChunk]:
        """Process JSON file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content_parts = []
                    metadata = {'file_type': 'json', 'file_name': Path(file_path).name, 'item_index': idx}
                    
                    for field in text_fields:
                        if field in item:
                            content_parts.append(f"{field}: {item[field]}")
                            metadata[field] = item[field]
                    
                    if content_parts:
                        content = '\n'.join(content_parts)
                        chunk_chunks = self.chunk_text(content, f"{file_path}:item_{idx}", metadata)
                        chunks.extend(chunk_chunks)
            
            elif isinstance(data, dict):
                content_parts = []
                metadata = {'file_type': 'json', 'file_name': Path(file_path).name}
                
                for field in text_fields:
                    if field in data:
                        content_parts.append(f"{field}: {data[field]}")
                        metadata[field] = data[field]
                
                if content_parts:
                    content = '\n'.join(content_parts)
                    chunk_chunks = self.chunk_text(content, file_path, metadata)
                    chunks.extend(chunk_chunks)
            
            logger.info(f"Processed {len(chunks)} chunks from JSON {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            return []

class DocumentIngestionManager:
    """Manages the complete document ingestion pipeline"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(config.pinecone_index_name)
        self.processor = DocumentProcessor()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.client.embeddings.create(
                model=config.embedding_model,
                input=text.replace('\n', ' ')
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _prepare_for_indexing(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Prepare chunks for Pinecone indexing"""
        vectors = []
        
        for chunk in chunks:
            # Generate embedding
            embedding = self._get_embedding(chunk.content)
            
            # Prepare metadata for Pinecone
            metadata = {
                **chunk.metadata,
                'text': chunk.content,  # Store original text in metadata
                'source': chunk.source,
                'chunk_id': chunk.chunk_id
            }
            
            # Create vector record
            vector = {
                'id': chunk.chunk_id,
                'values': embedding,
                'metadata': metadata
            }
            vectors.append(vector)
        
        return vectors
    
    def ingest_documents(self, document_paths: List[str], batch_size: int = 100) -> Dict[str, Any]:
        """
        Ingest multiple documents into the vector database
        """
        all_chunks = []
        processed_files = []
        failed_files = []
        
        for doc_path in document_paths:
            try:
                file_path = Path(doc_path)
                
                if not file_path.exists():
                    logger.warning(f"File not found: {doc_path}")
                    failed_files.append(doc_path)
                    continue
                
                # Process based on file type
                if file_path.suffix.lower() == '.txt':
                    chunks = self.processor.process_text_file(str(file_path))
                elif file_path.suffix.lower() == '.csv':
                    # You might want to customize this based on your CSV structure
                    chunks = self.processor.process_csv_file(str(file_path), 'content')
                elif file_path.suffix.lower() == '.json':
                    # You might want to customize this based on your JSON structure
                    chunks = self.processor.process_json_file(str(file_path), ['content', 'text', 'description'])
                else:
                    logger.warning(f"Unsupported file type: {file_path.suffix}")
                    failed_files.append(doc_path)
                    continue
                
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(doc_path)
                else:
                    failed_files.append(doc_path)
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                failed_files.append(doc_path)
        
        # Index chunks in batches
        total_indexed = 0
        if all_chunks:
            vectors = self._prepare_for_indexing(all_chunks)
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    total_indexed += len(batch)
                    logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} vectors")
                except Exception as e:
                    logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
        
        return {
            'total_documents': len(document_paths),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_chunks': len(all_chunks),
            'total_indexed': total_indexed,
            'index_stats': self.index.describe_index_stats()
        }
    
    def ingest_directory(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Ingest all documents from a directory
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.csv', '.json']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all files with specified extensions
        document_paths = []
        for ext in file_extensions:
            document_paths.extend(directory.glob(f"**/*{ext}"))
        
        document_paths = [str(p) for p in document_paths]
        logger.info(f"Found {len(document_paths)} documents in {directory_path}")
        
        return self.ingest_documents(document_paths)
    
    def delete_document(self, source: str) -> bool:
        """
        Delete all chunks from a specific document source
        """
        try:
            # Query for all chunks from this source
            query_response = self.index.query(
                vector=[0] * config.embedding_dimension,  # Dummy vector
                filter={"source": source},
                top_k=10000,  # Maximum to get all chunks
                include_metadata=True
            )
            
            # Extract IDs to delete
            ids_to_delete = [match['id'] for match in query_response['matches']]
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks from {source}")
                return True
            else:
                logger.info(f"No chunks found for source: {source}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {source}: {e}")
            return False
    
    def update_document(self, document_path: str) -> Dict[str, Any]:
        """
        Update a document by deleting old chunks and adding new ones
        """
        # Delete existing chunks
        self.delete_document(document_path)
        
        # Re-ingest the document
        return self.ingest_documents([document_path])
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        return self.index.describe_index_stats()

# Example usage functions
def create_sample_documents():
    """Create sample business documents for testing"""
    sample_docs = {
        'company_policy.txt': """
        Company Policy Document
        
        Remote Work Policy:
        Employees are allowed to work remotely up to 3 days per week with manager approval.
        Remote work requests must be submitted 24 hours in advance.
        
        Vacation Policy:
        All employees receive 15 days of paid vacation annually.
        Vacation requests must be submitted at least 2 weeks in advance.
        
        Meeting Guidelines:
        All meetings should have a clear agenda distributed 24 hours prior.
        Meeting duration should not exceed 60 minutes without exceptional circumstances.
        """,
        
        'product_info.txt': """
        Product Information Guide
        
        Cloud Storage Service:
        Our cloud storage service offers 1TB of secure storage with 99.9% uptime guarantee.
        Pricing: $9.99/month for individual users, $19.99/month for business users.
        
        Analytics Platform:
        Advanced analytics platform with real-time reporting and AI-powered insights.
        Supports integration with 200+ data sources.
        Pricing: Starting at $299/month for the basic plan.
        
        Customer Support:
        24/7 support available via chat, email, and phone.
        Response time: 2 hours for urgent issues, 24 hours for general inquiries.
        """,
        
        'faq.txt': """
        Frequently Asked Questions
        
        Q: How do I reset my password?
        A: Click on 'Forgot Password' on the login page and follow the instructions sent to your email.
        
        Q: What are the system requirements?
        A: Windows 10 or macOS 10.14+, 8GB RAM, 2GB free disk space, internet connection.
        
        Q: How do I contact support?
        A: You can reach support at support@company.com or call 1-800-SUPPORT.
        
        Q: What payment methods do you accept?
        A: We accept all major credit cards, PayPal, and bank transfers.
        """
    }
    
    # Create sample documents directory
    sample_dir = Path('sample_documents')
    sample_dir.mkdir(exist_ok=True)
    
    for filename, content in sample_docs.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return str(sample_dir)

if __name__ == "__main__":
    # Example usage
    ingestion_manager = DocumentIngestionManager()
    
    # Create and ingest sample documents
    sample_dir = create_sample_documents()
    print(f"Created sample documents in: {sample_dir}")
    
    # Ingest documents
    result = ingestion_manager.ingest_directory(sample_dir)
    print(f"Ingestion result: {result}")
