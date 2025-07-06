"""
Example Usage Script for RAG QA Chatbot
Demonstrates various features and use cases
"""

import json
import time
from typing import List, Dict
from rag_system import RAGSystem, RAGResponse
from document_ingestion import DocumentIngestionManager, create_sample_documents
from config import config

def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_response(response: RAGResponse, query: str):
    """Print a formatted response"""
    print(f"\n🔍 Query: {query}")
    print(f"🤖 Answer: {response.answer}")
    
    if response.sources:
        print(f"📚 Sources: {', '.join(response.sources)}")
    
    print(f"🎯 Confidence: {response.confidence:.2f}")
    
    if response.needs_clarification:
        print("❓ Clarification needed:")
        for q in response.clarification_questions:
            print(f"   • {q}")
    
    print("-" * 40)

def example_basic_queries():
    """Demonstrate basic query functionality"""
    print_separator("BASIC QUERIES")
    
    rag = RAGSystem()
    
    # Simple queries
    queries = [
        "What is the remote work policy?",
        "How much does cloud storage cost?",
        "What are the system requirements?",
        "How do I reset my password?",
        "What payment methods do you accept?"
    ]
    
    for query in queries:
        response = rag.query(query)
        print_response(response, query)
        time.sleep(1)  # Be nice to the API

def example_advanced_queries():
    """Demonstrate advanced query scenarios"""
    print_separator("ADVANCED QUERIES")
    
    rag = RAGSystem()
    
    # Complex queries
    advanced_queries = [
        "Compare pricing between cloud storage and analytics platform",
        "What are all the ways I can contact support and their response times?",
        "Summarize the meeting guidelines and vacation policies",
        "What services offer 24/7 support and what are their features?"
    ]
    
    for query in advanced_queries:
        response = rag.query(query)
        print_response(response, query)
        time.sleep(1)

def example_clarification_scenarios():
    """Demonstrate clarification handling"""
    print_separator("CLARIFICATION SCENARIOS")
    
    rag = RAGSystem()
    
    # Ambiguous queries that should trigger clarification
    ambiguous_queries = [
        "How much does it cost?",
        "What are the requirements?",
        "How do I get support?",
        "What's the policy?"
    ]
    
    for query in ambiguous_queries:
        response = rag.query(query)
        print_response(response, query)
        time.sleep(1)

def example_conversation_flow():
    """Demonstrate conversation with context"""
    print_separator("CONVERSATION WITH CONTEXT")
    
    rag = RAGSystem()
    conversation_history = []
    
    # Simulate a conversation
    conversation = [
        "What cloud services do you offer?",
        "How much does the storage service cost?",
        "What about for business users?",
        "Do you offer any guarantees?",
        "How can I sign up?"
    ]
    
    for query in conversation:
        response = rag.query(query, conversation_history)
        print_response(response, query)
        
        # Update conversation history
        if not response.needs_clarification:
            conversation_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response.answer}
            ])
        
        time.sleep(1)

def example_document_management():
    """Demonstrate document ingestion and management"""
    print_separator("DOCUMENT MANAGEMENT")
    
    ingestion_manager = DocumentIngestionManager()
    
    # Create and ingest sample documents
    print("📄 Creating sample documents...")
    sample_dir = create_sample_documents()
    print(f"✅ Sample documents created in: {sample_dir}")
    
    # Ingest documents
    print("\n📥 Ingesting documents...")
    result = ingestion_manager.ingest_directory(sample_dir)
    print(f"✅ Ingested {result['total_indexed']} chunks from {len(result['processed_files'])} files")
    
    # Show index statistics
    print("\n📊 Index Statistics:")
    stats = ingestion_manager.get_index_stats()
    print(json.dumps(stats, indent=2))
    
    return sample_dir

def example_batch_processing():
    """Demonstrate batch query processing"""
    print_separator("BATCH PROCESSING")
    
    rag = RAGSystem()
    
    # Create batch queries
    batch_queries = [
        "What is the vacation policy?",
        "How much does cloud storage cost for individuals?",
        "What are the system requirements?",
        "How do I contact support?",
        "What meeting guidelines should I follow?",
        "What's included in the analytics platform?",
        "How do I reset my password?",
        "What payment methods are accepted?"
    ]
    
    print(f"🔄 Processing {len(batch_queries)} queries in batch...")
    
    results = []
    for i, query in enumerate(batch_queries, 1):
        print(f"Processing {i}/{len(batch_queries)}: {query[:50]}...")
        response = rag.query(query)
        
        result = {
            "query": query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "needs_clarification": response.needs_clarification
        }
        results.append(result)
        time.sleep(0.5)  # Brief pause between queries
    
    # Show summary
    print(f"\n📋 Batch Processing Summary:")
    print(f"   Total queries: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if not r['needs_clarification'])}")
    print(f"   Requiring clarification: {sum(1 for r in results if r['needs_clarification'])}")
    print(f"   Average confidence: {sum(r['confidence'] for r in results) / len(results):.2f}")
    
    return results

def example_confidence_analysis():
    """Demonstrate confidence scoring analysis"""
    print_separator("CONFIDENCE ANALYSIS")
    
    rag = RAGSystem()
    
    # Queries with different expected confidence levels
    test_queries = [
        ("What is the vacation policy?", "High confidence - direct policy question"),
        ("How much does cloud storage cost?", "High confidence - specific pricing question"),
        ("What are the best practices for remote work?", "Medium confidence - interpretive question"),
        ("How much does the premium enterprise plan cost?", "Low confidence - non-existent plan"),
        ("What is the company's stance on AI ethics?", "Low confidence - topic not in documents")
    ]
    
    for query, expected in test_queries:
        response = rag.query(query)
        print(f"\n🔍 Query: {query}")
        print(f"📈 Expected: {expected}")
        print(f"🎯 Actual Confidence: {response.confidence:.2f}")
        print(f"❓ Needs Clarification: {response.needs_clarification}")
        
        if response.confidence >= config.confidence_threshold:
            print("✅ High confidence response")
        else:
            print("⚠️ Low confidence - clarification requested")
        
        time.sleep(1)

def example_system_health():
    """Demonstrate system health monitoring"""
    print_separator("SYSTEM HEALTH MONITORING")
    
    rag = RAGSystem()
    
    print("🏥 Performing system health check...")
    health = rag.health_check()
    
    print("\n📊 Health Status:")
    print(f"   OpenAI API: {health.get('openai_status', 'Unknown')}")
    print(f"   Pinecone DB: {health.get('pinecone_status', 'Unknown')}")
    
    if 'index_stats' in health:
        stats = health['index_stats']
        print(f"   Vector Count: {stats.get('total_vector_count', 0)}")
        print(f"   Index Dimension: {stats.get('dimension', 0)}")
    
    print(f"\n⚙️ Configuration:")
    print(f"   Model: {health.get('configuration', {}).get('model', 'Unknown')}")
    print(f"   Embedding Model: {health.get('configuration', {}).get('embedding_model', 'Unknown')}")
    print(f"   Index Name: {health.get('configuration', {}).get('index_name', 'Unknown')}")

def example_performance_testing():
    """Demonstrate performance characteristics"""
    print_separator("PERFORMANCE TESTING")
    
    rag = RAGSystem()
    
    # Test queries with timing
    test_queries = [
        "What is the remote work policy?",
        "How much does cloud storage cost?",
        "What are the system requirements?"
    ]
    
    print("⏱️ Testing query performance...")
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        response = rag.query(query)
        end_time = time.time()
        
        query_time = end_time - start_time
        total_time += query_time
        
        print(f"\n🔍 Query {i}: {query}")
        print(f"⏱️ Response Time: {query_time:.2f} seconds")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"📚 Sources Found: {len(response.context_used)}")
    
    avg_time = total_time / len(test_queries)
    print(f"\n📊 Performance Summary:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Time: {avg_time:.2f} seconds per query")
    print(f"   Queries per Minute: {60 / avg_time:.1f}")

def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Starting RAG QA Chatbot Examples")
    print(f"Configuration: {config.openai_model} + {config.embedding_model}")
    
    try:
        # System health first
        example_system_health()
        
        # Document management
        sample_dir = example_document_management()
        
        # Basic functionality
        example_basic_queries()
        
        # Advanced features
        example_advanced_queries()
        
        # Clarification handling
        example_clarification_scenarios()
        
        # Conversation flow
        example_conversation_flow()
        
        # Batch processing
        batch_results = example_batch_processing()
        
        # Confidence analysis
        example_confidence_analysis()
        
        # Performance testing
        example_performance_testing()
        
        print_separator("EXAMPLES COMPLETED")
        print("✅ All examples completed successfully!")
        print(f"📁 Sample documents available in: {sample_dir}")
        print("🚀 You can now run:")
        print("   - python cli.py chat")
        print("   - streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Please check your configuration and API keys.")

if __name__ == "__main__":
    run_all_examples()
