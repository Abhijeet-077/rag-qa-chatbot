# RAG QA Chatbot System

A complete Retrieval-Augmented Generation (RAG) system for business question-answering using OpenAI's GPT-4 and Pinecone vector database. This system implements the ROSE framework (Role, Objective, Style, Execution) with recursive prompting for optimal performance.

## 🚀 Features

- **ROSE Framework Implementation**: Structured prompt engineering for consistent, business-appropriate responses
- **Recursive Prompting**: Automatic clarification and context evaluation
- **Multi-format Document Support**: Text, CSV, and JSON document ingestion
- **Vector-based Retrieval**: Semantic search using OpenAI embeddings and Pinecone
- **Confidence Scoring**: Automatic assessment of response reliability
- **Web Interface**: Streamlit-based GUI for easy interaction
- **CLI Support**: Command-line interface for automation and batch processing
- **Health Monitoring**: System status checks and diagnostics

## 🏗️ Architecture

### Core Components

1. **RAG System** (`rag_system.py`): Main orchestrator implementing ROSE framework
2. **Document Ingestion** (`document_ingestion.py`): Document processing and vector indexing
3. **Configuration** (`config.py`): Centralized settings management
4. **Web Interface** (`streamlit_app.py`): Interactive Streamlit application
5. **CLI** (`cli.py`): Command-line interface

### ROSE Framework Implementation

**ROLE**: Domain-aware business assistant with expertise in document analysis
**OBJECTIVE**: Provide precise, business-appropriate answers while reducing manual workload
**STYLE**: Professional, concise communication with clear source attribution
**EXECUTION**: Multi-step retrieval, context evaluation, and response generation

### Recursive Prompting Features

- Automatic query clarity assessment
- Context sufficiency evaluation
- Confidence-based clarification requests
- Follow-up question generation

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and index
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   cd rag-qa-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Required Environment Variables**
   Create a `.env` file in the project root with your actual API keys:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   PINECONE_API_KEY=your_actual_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=business-knowledge-base
   ```

   ⚠️ **Security Note**: Never commit your `.env` file to version control. The `.gitignore` file is configured to exclude it.

## 🎯 Quick Start

### 1. Create Sample Documents and Setup
```bash
python cli.py docs create-samples --ingest
```

### 2. Test the System
```bash
python cli.py health
```

### 3. Start Interactive Chat
```bash
python cli.py chat
```

### 4. Launch Web Interface
```bash
streamlit run streamlit_app.py
```

## 💻 Usage Examples

### Command Line Interface

#### Interactive Chat Mode
```bash
python cli.py chat
```

#### Single Query
```bash
python cli.py query "What is the remote work policy?"
```

#### JSON Output
```bash
python cli.py query "How much does cloud storage cost?" --format json
```

#### Batch Processing
```bash
# Create a file with questions (queries.txt)
python cli.py batch queries.txt --output results.json
```

#### Document Management
```bash
# Ingest documents from directory
python cli.py docs ingest --directory ./documents

# Ingest specific files
python cli.py docs ingest --files doc1.txt doc2.csv

# View index statistics
python cli.py docs stats

# Delete a document
python cli.py docs delete --source "./documents/old_policy.txt"

# Update a document
python cli.py docs update --document "./documents/updated_policy.txt"
```

### Web Interface Features

- **Interactive Chat**: Real-time question answering
- **Document Upload**: Drag-and-drop document ingestion
- **System Health**: Real-time status monitoring
- **Context Viewing**: Inspect retrieved document chunks
- **Conversation History**: Track chat sessions

### Python API Usage

```python
from rag_system import RAGSystem
from document_ingestion import DocumentIngestionManager

# Initialize system
rag = RAGSystem()

# Process a query
response = rag.query("What are the system requirements?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Sources: {response.sources}")

# Document management
ingestion = DocumentIngestionManager()
result = ingestion.ingest_documents(["policy.txt", "faq.txt"])
print(f"Indexed {result['total_indexed']} chunks")
```

## 📝 Example Interactions

### Simple Query
```
User: "What is the vacation policy?"

Bot: "According to the company policy, all employees receive 15 days of paid vacation annually. Vacation requests must be submitted at least 2 weeks in advance.

Sources: company_policy.txt
Confidence: 0.92"
```

### Complex Query Requiring Clarification
```
User: "How much does it cost?"

Bot: "I need more information to provide an accurate answer."

Clarification Questions:
1. Are you asking about our cloud storage service or analytics platform pricing?
2. Do you need individual or business pricing information?
3. Are you interested in monthly or annual pricing options?
```

### Advanced Query with Multiple Sources
```
User: "Compare the pricing of cloud storage vs analytics platform"

Bot: "Here's a comparison of our pricing:

**Cloud Storage Service:**
- Individual: $9.99/month
- Business: $19.99/month
- Includes 1TB storage with 99.9% uptime guarantee

**Analytics Platform:**
- Basic plan: Starting at $299/month
- Includes real-time reporting and AI-powered insights
- Supports 200+ data source integrations

The analytics platform is positioned as an enterprise solution while cloud storage serves both individual and business users at different price points.

Sources: product_info.txt
Confidence: 0.95"
```

## 🔧 Configuration Options

### Core Settings
- `OPENAI_MODEL`: GPT model to use (default: gpt-4-turbo)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-large)
- `MAX_TOKENS`: Maximum response length (default: 4000)
- `TEMPERATURE`: Response creativity (default: 0.3)

### RAG Parameters
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for responses (default: 0.7)
- `MAX_CONTEXT_LENGTH`: Maximum context window (default: 8000)

### Advanced Customization
- Modify prompts in `rag_system.py` (ROSEPromptEngine class)
- Adjust chunking strategy in `document_ingestion.py`
- Customize confidence scoring algorithms

## 🏥 System Health & Monitoring

### Health Check
```bash
python cli.py health
```

### Monitoring Features
- OpenAI API connectivity
- Pinecone index status
- Vector count and statistics
- Configuration validation

### Troubleshooting Common Issues

1. **API Key Issues**
   - Verify keys in `.env` file
   - Check API key permissions

2. **Pinecone Connection Problems**
   - Confirm index exists and environment is correct
   - Check Pinecone dashboard for service status

3. **Low Response Quality**
   - Increase `TOP_K_RESULTS` for more context
   - Adjust `CONFIDENCE_THRESHOLD`
   - Review document quality and chunking

4. **Performance Issues**
   - Reduce `MAX_CONTEXT_LENGTH`
   - Optimize document chunking size
   - Monitor token usage

## 🔒 Security Best Practices

- Store API keys securely using environment variables
- Implement rate limiting for production deployments
- Validate and sanitize user inputs
- Monitor API usage and costs
- Regular security updates of dependencies

## 📊 Performance Optimization

### Document Processing
- Optimize chunk size based on content type
- Use appropriate overlap for context preservation
- Clean and preprocess documents before ingestion

### Query Performance
- Cache frequent queries
- Implement query optimization
- Monitor and optimize embedding generation

### Cost Management
- Monitor OpenAI API usage
- Optimize prompt length and token usage
- Consider using smaller models for development

## 🛣️ Roadmap

- [ ] Support for additional document formats (PDF, DOCX)
- [ ] Multi-language support
- [ ] Advanced analytics and usage tracking
- [ ] Integration with enterprise authentication systems
- [ ] Automated model fine-tuning
- [ ] Real-time document updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For questions and support:
- Check the troubleshooting section
- Review configuration settings
- Run health checks
- Create an issue for bugs or feature requests

---

**Built with ❤️ using OpenAI GPT-4, Pinecone, and Streamlit**
