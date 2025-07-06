"""
Setup Script for RAG QA Chatbot
Handles initial setup and configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🚀 RAG QA Chatbot Setup")
    print("=" * 50)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("📋 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup environment configuration"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️ .env file already exists")
        response = input("   Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("   Keeping existing .env file")
            return True
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    # Copy example to .env
    env_content = env_example.read_text()
    env_file.write_text(env_content)
    
    print("✅ Created .env file from template")
    print("📝 Please edit .env file with your API keys:")
    print("   - OPENAI_API_KEY")
    print("   - PINECONE_API_KEY")
    print("   - PINECONE_ENVIRONMENT")
    print("   - PINECONE_INDEX_NAME")
    
    return True

def validate_configuration():
    """Validate the configuration"""
    print("\n🔍 Validating configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    env_content = env_file.read_text()
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if f"{var}=your_" in env_content or var not in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️ The following variables need to be configured:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease edit the .env file before proceeding.")
        return False
    
    print("✅ Configuration appears complete")
    return True

def test_system():
    """Test system functionality"""
    print("\n🧪 Testing system functionality...")
    
    try:
        # Try importing main modules
        from config import config
        print("✅ Configuration module loaded")
        
        # Test if we can initialize the system
        from rag_system import RAGSystem
        from document_ingestion import DocumentIngestionManager
        
        print("✅ Core modules imported successfully")
        
        # Try health check
        rag = RAGSystem()
        health = rag.health_check()
        
        if health.get('openai_status') == 'healthy':
            print("✅ OpenAI connection successful")
        else:
            print("⚠️ OpenAI connection issue")
            
        if health.get('pinecone_status') == 'healthy':
            print("✅ Pinecone connection successful")
        else:
            print("⚠️ Pinecone connection issue")
            
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print("Please check your configuration and API keys.")
        return False

def create_sample_data():
    """Create sample documents and ingest them"""
    print("\n📄 Creating sample documents...")
    
    try:
        from document_ingestion import create_sample_documents, DocumentIngestionManager
        
        # Create sample documents
        sample_dir = create_sample_documents()
        print(f"✅ Sample documents created in: {sample_dir}")
        
        # Ask if user wants to ingest them
        response = input("   Do you want to ingest sample documents now? (y/n): ").lower()
        if response == 'y':
            print("📥 Ingesting sample documents...")
            
            ingestion_manager = DocumentIngestionManager()
            result = ingestion_manager.ingest_directory(sample_dir)
            
            print(f"✅ Ingested {result['total_indexed']} chunks from {len(result['processed_files'])} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def show_next_steps():
    """Show next steps to the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("\n🚀 Next Steps:")
    print()
    print("1. 📝 Edit .env file with your API keys (if not done yet)")
    print("2. 🧪 Test the system:")
    print("   python cli.py health")
    print()
    print("3. 💬 Start interactive chat:")
    print("   python cli.py chat")
    print()
    print("4. 🌐 Launch web interface:")
    print("   streamlit run streamlit_app.py")
    print()
    print("5. 📚 Run examples:")
    print("   python examples.py")
    print()
    print("6. 📖 Check README.md for detailed usage instructions")
    print()
    print("🔗 Useful commands:")
    print("   python cli.py docs create-samples --ingest  # Create and ingest sample docs")
    print("   python cli.py query \"your question\"         # Ask a single question")
    print("   python cli.py docs stats                    # View index statistics")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed at environment configuration")
        sys.exit(1)
    
    # Check if configuration is complete
    config_complete = validate_configuration()
    
    if config_complete:
        # Test system
        if test_system():
            # Create sample data
            create_sample_data()
        else:
            print("\n⚠️ System test failed, but setup is otherwise complete")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
