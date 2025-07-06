"""
Streamlit Web Interface for RAG QA Chatbot
Provides an interactive web interface for the RAG system
"""

import streamlit as st
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from rag_system import RAGSystem, RAGResponse
from document_ingestion import DocumentIngestionManager, create_sample_documents
from config import config

# Configure page
st.set_page_config(
    page_title="RAG QA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

@st.cache_resource
def initialize_ingestion_manager():
    """Initialize document ingestion manager with caching"""
    try:
        return DocumentIngestionManager()
    except Exception as e:
        st.error(f"Failed to initialize ingestion manager: {e}")
        return None

def format_response(response: RAGResponse) -> str:
    """Format RAG response for display"""
    formatted = response.answer
    
    if response.sources:
        formatted += f"\n\n**Sources:**\n"
        for source in response.sources:
            formatted += f"- {source}\n"
    
    formatted += f"\n**Confidence:** {response.confidence:.2f}"
    
    return formatted

def display_clarification_questions(questions: List[str]):
    """Display clarification questions as clickable buttons"""
    st.warning("I need more information to provide an accurate answer. Here are some clarifying questions:")
    
    for i, question in enumerate(questions):
        if st.button(question, key=f"clarification_{i}"):
            st.session_state.user_input = question
            st.rerun()

def main():
    """Main Streamlit application"""
    
    # Page header
    st.title("🤖 RAG QA Chatbot")
    st.markdown("*Intelligent business document question-answering system*")
    
    # Initialize systems
    rag_system = initialize_rag_system()
    ingestion_manager = initialize_ingestion_manager()
    
    if not rag_system or not ingestion_manager:
        st.error("Failed to initialize system components. Please check your configuration.")
        return
    
    # Sidebar for system management
    with st.sidebar:
        st.header("System Management")
        
        # System health check
        if st.button("Health Check"):
            with st.spinner("Checking system health..."):
                health = rag_system.health_check()
                
                if health.get('openai_status') == 'healthy' and health.get('pinecone_status') == 'healthy':
                    st.success("✅ All systems operational")
                else:
                    st.error("❌ System issues detected")
                
                st.json(health)
        
        # Document management
        st.subheader("Document Management")
        
        # Sample documents
        if st.button("Create Sample Documents"):
            with st.spinner("Creating sample documents..."):
                sample_dir = create_sample_documents()
                st.success(f"Sample documents created in: {sample_dir}")
        
        # Document ingestion
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=['txt', 'csv', 'json']
        )
        
        if uploaded_files and st.button("Ingest Documents"):
            with st.spinner("Ingesting documents..."):
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)
                
                # Ingest documents
                result = ingestion_manager.ingest_documents(temp_paths)
                
                # Clean up temp files
                import os
                for path in temp_paths:
                    os.remove(path)
                
                st.success(f"Ingested {result['total_indexed']} chunks from {len(result['processed_files'])} files")
                st.json(result)
        
        # Index statistics
        if st.button("View Index Stats"):
            stats = ingestion_manager.get_index_stats()
            st.json(stats)
    
    # Main chat interface
    st.header("Ask a Question")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question:",
            value=st.session_state.user_input,
            placeholder="e.g., What is the remote work policy?",
            key="query_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary")
    
    # Process query
    if (ask_button or user_query) and user_query.strip():
        with st.spinner("Thinking..."):
            # Query the RAG system
            response = rag_system.query(user_query, st.session_state.conversation_history)
            
            # Display response
            st.markdown("### Response")
            
            if response.needs_clarification:
                display_clarification_questions(response.clarification_questions)
            else:
                formatted_response = format_response(response)
                st.markdown(formatted_response)
                
                # Update conversation history
                st.session_state.conversation_history.extend([
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response.answer}
                ])
            
            # Display context information in expander
            with st.expander("View Retrieved Context", expanded=False):
                if response.context_used:
                    for i, context in enumerate(response.context_used):
                        st.markdown(f"**Context {i+1}** (Score: {context.score:.3f})")
                        st.markdown(f"*Source: {context.source}*")
                        st.text(context.content[:500] + "..." if len(context.content) > 500 else context.content)
                        st.markdown("---")
                else:
                    st.info("No relevant context found.")
        
        # Clear input
        st.session_state.user_input = ""
    
    # Conversation history
    if st.session_state.conversation_history:
        st.header("Conversation History")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Display conversation
        for i in range(0, len(st.session_state.conversation_history), 2):
            if i + 1 < len(st.session_state.conversation_history):
                user_msg = st.session_state.conversation_history[i]
                bot_msg = st.session_state.conversation_history[i + 1]
                
                st.markdown(f"**👤 You:** {user_msg['content']}")
                st.markdown(f"**🤖 Bot:** {bot_msg['content']}")
                st.markdown("---")
    
    # Example queries
    st.header("Example Queries")
    
    example_queries = [
        "What is the remote work policy?",
        "How much does the cloud storage service cost?",
        "What are the system requirements?",
        "How do I reset my password?",
        "What payment methods do you accept?",
        "What is the vacation policy?",
        "How can I contact support?",
        "Tell me about the analytics platform pricing"
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(query, key=f"example_{i}"):
                st.session_state.user_input = query
                st.rerun()

# Advanced features page
def advanced_features():
    """Advanced features and configuration"""
    st.title("Advanced Features")
    
    # Configuration display
    st.header("System Configuration")
    config_dict = {
        "OpenAI Model": config.openai_model,
        "Embedding Model": config.embedding_model,
        "Embedding Dimension": config.embedding_dimension,
        "Max Tokens": config.max_tokens,
        "Temperature": config.temperature,
        "Top K Results": config.top_k_results,
        "Confidence Threshold": config.confidence_threshold,
        "Max Context Length": config.max_context_length
    }
    
    st.json(config_dict)
    
    # Prompt engineering
    st.header("Prompt Engineering")
    rag_system = initialize_rag_system()
    if rag_system:
        st.subheader("Current System Prompt")
        st.text_area("System Prompt", rag_system.prompt_engine.base_prompt, height=300, disabled=True)
        
        st.subheader("Recursive Prompt")
        st.text_area("Recursive Prompt", rag_system.prompt_engine.recursive_prompt, height=200, disabled=True)
    
    # Analytics and metrics
    st.header("System Analytics")
    if st.button("Generate Analytics Report"):
        ingestion_manager = initialize_ingestion_manager()
        if ingestion_manager:
            stats = ingestion_manager.get_index_stats()
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Vectors", stats.get('total_vector_count', 0))
            
            with col2:
                st.metric("Index Dimension", stats.get('dimension', 0))
            
            with col3:
                st.metric("Namespaces", len(stats.get('namespaces', {})))

# Navigation
def main_navigation():
    """Main navigation setup"""
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Advanced"])
    
    with tab1:
        main()
    
    with tab2:
        advanced_features()

if __name__ == "__main__":
    main_navigation()
