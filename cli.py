"""
Command Line Interface for RAG QA Chatbot
Provides a CLI for interacting with the RAG system
"""

import argparse
import json
import sys
from typing import List, Dict, Any
from rag_system import RAGSystem
from document_ingestion import DocumentIngestionManager, create_sample_documents
from config import config

def chat_mode(rag_system: RAGSystem):
    """Interactive chat mode"""
    print("🤖 RAG QA Chatbot - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
- quit/exit/q: Exit the chat
- help: Show this help message
- clear: Clear conversation history
- health: Check system health
- Just type your question to get an answer!
                """)
                continue
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("🗑️ Conversation history cleared")
                continue
            
            if user_input.lower() == 'health':
                health = rag_system.health_check()
                print(f"🏥 System Health: {json.dumps(health, indent=2)}")
                continue
            
            if not user_input:
                continue
            
            print("🤔 Thinking...")
            
            # Query the RAG system
            response = rag_system.query(user_input, conversation_history)
            
            print(f"\n🤖 Bot: {response.answer}")
            
            if response.sources:
                print(f"\n📚 Sources: {', '.join(response.sources)}")
            
            print(f"🎯 Confidence: {response.confidence:.2f}")
            
            if response.needs_clarification:
                print("\n❓ Clarification questions:")
                for i, question in enumerate(response.clarification_questions, 1):
                    print(f"  {i}. {question}")
            
            # Update conversation history
            if not response.needs_clarification:
                conversation_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.answer}
                ])
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def single_query(rag_system: RAGSystem, query: str, output_format: str = "text"):
    """Process a single query"""
    response = rag_system.query(query)
    
    if output_format == "json":
        result = {
            "query": query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "needs_clarification": response.needs_clarification,
            "clarification_questions": response.clarification_questions
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Query: {query}")
        print(f"Answer: {response.answer}")
        if response.sources:
            print(f"Sources: {', '.join(response.sources)}")
        print(f"Confidence: {response.confidence:.2f}")
        
        if response.needs_clarification:
            print("Clarification needed:")
            for question in response.clarification_questions:
                print(f"  - {question}")

def batch_queries(rag_system: RAGSystem, query_file: str, output_file: str = None):
    """Process multiple queries from a file"""
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            response = rag_system.query(query)
            
            result = {
                "query": query,
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": response.sources,
                "needs_clarification": response.needs_clarification,
                "clarification_questions": response.clarification_questions
            }
            results.append(result)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        else:
            print(json.dumps(results, indent=2))
    
    except FileNotFoundError:
        print(f"❌ Query file not found: {query_file}")
    except Exception as e:
        print(f"❌ Error processing batch queries: {e}")

def manage_documents(action: str, **kwargs):
    """Document management operations"""
    ingestion_manager = DocumentIngestionManager()
    
    if action == "ingest":
        if kwargs.get('directory'):
            result = ingestion_manager.ingest_directory(kwargs['directory'])
        elif kwargs.get('files'):
            result = ingestion_manager.ingest_documents(kwargs['files'])
        else:
            print("❌ Please specify either --directory or --files")
            return
        
        print(f"✅ Ingestion completed:")
        print(f"  - Processed: {len(result['processed_files'])} files")
        print(f"  - Failed: {len(result['failed_files'])} files")
        print(f"  - Total chunks: {result['total_chunks']}")
        print(f"  - Indexed: {result['total_indexed']} vectors")
    
    elif action == "delete":
        source = kwargs.get('source')
        if not source:
            print("❌ Please specify --source for deletion")
            return
        
        success = ingestion_manager.delete_document(source)
        if success:
            print(f"✅ Deleted document: {source}")
        else:
            print(f"❌ Failed to delete document: {source}")
    
    elif action == "update":
        document_path = kwargs.get('document')
        if not document_path:
            print("❌ Please specify --document for update")
            return
        
        result = ingestion_manager.update_document(document_path)
        print(f"✅ Updated document: {document_path}")
        print(f"  - Indexed: {result['total_indexed']} vectors")
    
    elif action == "stats":
        stats = ingestion_manager.get_index_stats()
        print("📊 Index Statistics:")
        print(json.dumps(stats, indent=2))
    
    elif action == "create-samples":
        sample_dir = create_sample_documents()
        print(f"✅ Sample documents created in: {sample_dir}")
        
        # Optionally ingest the sample documents
        if kwargs.get('ingest_samples'):
            result = ingestion_manager.ingest_directory(sample_dir)
            print(f"✅ Sample documents ingested: {result['total_indexed']} vectors")

def system_health():
    """Check system health"""
    try:
        rag_system = RAGSystem()
        health = rag_system.health_check()
        
        print("🏥 System Health Check:")
        print(json.dumps(health, indent=2))
        
        if health.get('openai_status') == 'healthy' and health.get('pinecone_status') == 'healthy':
            print("✅ All systems operational")
            return True
        else:
            print("❌ System issues detected")
            return False
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RAG QA Chatbot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat mode')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Process a single query')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple queries from file')
    batch_parser.add_argument('query_file', help='File containing queries (one per line)')
    batch_parser.add_argument('--output', help='Output file for results (JSON format)')
    
    # Document management commands
    doc_parser = subparsers.add_parser('docs', help='Document management')
    doc_subparsers = doc_parser.add_subparsers(dest='doc_action', help='Document actions')
    
    # Ingest documents
    ingest_parser = doc_subparsers.add_parser('ingest', help='Ingest documents')
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument('--directory', help='Directory containing documents')
    ingest_group.add_argument('--files', nargs='+', help='Specific files to ingest')
    
    # Delete document
    delete_parser = doc_subparsers.add_parser('delete', help='Delete document from index')
    delete_parser.add_argument('--source', required=True, help='Source identifier to delete')
    
    # Update document
    update_parser = doc_subparsers.add_parser('update', help='Update a document')
    update_parser.add_argument('--document', required=True, help='Document path to update')
    
    # View stats
    doc_subparsers.add_parser('stats', help='View index statistics')
    
    # Create samples
    samples_parser = doc_subparsers.add_parser('create-samples', help='Create sample documents')
    samples_parser.add_argument('--ingest', action='store_true', help='Also ingest the sample documents')
    
    # Health check
    subparsers.add_parser('health', help='Check system health')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'chat':
            rag_system = RAGSystem()
            chat_mode(rag_system)
        
        elif args.command == 'query':
            rag_system = RAGSystem()
            single_query(rag_system, args.question, args.format)
        
        elif args.command == 'batch':
            rag_system = RAGSystem()
            batch_queries(rag_system, args.query_file, args.output)
        
        elif args.command == 'docs':
            if args.doc_action == 'ingest':
                manage_documents('ingest', directory=args.directory, files=args.files)
            elif args.doc_action == 'delete':
                manage_documents('delete', source=args.source)
            elif args.doc_action == 'update':
                manage_documents('update', document=args.document)
            elif args.doc_action == 'stats':
                manage_documents('stats')
            elif args.doc_action == 'create-samples':
                manage_documents('create-samples', ingest_samples=args.ingest)
            else:
                doc_parser.print_help()
        
        elif args.command == 'health':
            system_health()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
