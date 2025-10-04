# 🤖 Robert - RAG Powered AI Research Assistant

A fully functional Retrieval-Augmented Generation (RAG) chatbot with a modern web interface. Upload documents, provide URLs, and ask questions about your content in real-time!

## ✨ Features

- **📁 Document Upload**: Support for PDF, TXT, and Markdown files
- **🌐 URL Processing**: Load and process content from web URLs
- **💬 Interactive Chat**: Real-time question answering with context
- **🔄 Session Management**: Multiple conversation sessions
- **📊 Progress Tracking**: View loaded documents and processing status
- **🎨 Modern UI**: Beautiful and responsive Streamlit interface
- **🚀 Fast API**: RESTful backend with FastAPI
- **🔍 Context Display**: See sources and context for each answer

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit      │    │   FastAPI       │    │   RAG Service   │
│  Frontend       │◄───┤   Backend       │◄───┤                 │
│                 │    │                 │    │  • LangChain    │
└─────────────────┘    └─────────────────┘    │  • FAISS        │
        │                       │             │  • Embeddings   │
        └───────────────────────┘             └─────────────────┘
                     │
            ┌─────────────────┐
            │   Documents     │
            │  • PDFs         │
            │  • Text files   │
            │  • Web URLs     │
            └─────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (or OpenAI API key)

### 1. Clone or Setup the Project

Ensure you have all the project files in your directory:
- `backend/backend.py` - FastAPI backend server
- `backend/rag_service.py` - Core RAG service logic
- `frontend/frontend.py` - Streamlit frontend application
- `config/config.py` - Configuration settings
- `requirements.txt` - Python dependencies

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Update your `config.py` file with your API keys:

```python
# API Configuration
GOOGLE_API_KEY = "your-google-gemini-api-key-here"
OPENAI_API_KEY = "your-openai-api-key-here"  # Optional

# Model Configuration
MODEL_NAME = "gemini-2.5-flash"  # or "gpt-4"
```

### 4. Vector Storage
Existing vector storage consists of some preloaded document chunks based on harry potter related websites.
You can keep them for testing purpose or delete the contents of the DB storage directory.
All the new documents / URLs you upload will be stored in this place.

## 🚀 Running the Application

### Method 1: Using Startup Scripts

**Windows:**
```cmd
# Terminal 1 (Backend)
start_backend.bat

# Terminal 2 (Frontend)
start_frontend.bat
```

**Linux/Mac:**
```bash
# Terminal 1 (Backend)
chmod +x start_backend.sh
./start_backend.sh

# Terminal 2 (Frontend)
chmod +x start_frontend.sh
./start_frontend.sh
```

## 📱 Using the Application

1. **Access the Application**: Open your browser and go to `http://localhost:8501`

2. **Upload Documents**: 
   - Use the sidebar to upload PDF, TXT, or MD files
   - Files are processed automatically and added to your knowledge base (Vector DB Storage)

3. **Load Web Content**:
   - Enter URLs in the sidebar (one per line)
   - Web content is scraped and processed

4. **Ask Questions**:
   - Type your questions in the chat input
   - Get AI-powered answers based on your uploaded content
   - View context sources for each answer

5. **Manage Sessions**:
   - Create new sessions for different topics
   - Clear sessions to start fresh

## 🔧 API Endpoints

The backend provides several REST API endpoints:

- `GET /` - API information
- `POST /chat` - Ask questions
- `POST /upload-file` - Upload and process files
- `POST /load-urls` - Load web URLs
- `GET /session/{session_id}` - Get session information
- `DELETE /clear-session/{session_id}` - Clear session data
- `GET /health` - Health check

## 📊 Configuration Options

Edit `config.py` to customize:

- **API Keys**: Google Gemini, OpenAI
- **Model Selection**: Choose between different LLM providers
- **Embedding Model**: HuggingFace model for text embeddings
- **Chunk Settings**: Document splitting parameters
- **File Paths**: Default locations for documents

## 🎯 Example Use Cases

1. **Research Assistant**: Upload academic papers and ask questions
2. **Document Analysis**: Analyze business documents and reports  
3. **Knowledge Base**: Create a searchable knowledge base from websites
4. **Study Aid**: Upload textbooks and get explanations
5. **Content Summary**: Get insights from multiple sources

## 🔍 Troubleshooting

### Common Issues

1. **Backend Connection Error**:
   - Ensure backend is running on `http://localhost:8000`
   - Check if port 8000 is available

2. **File Upload Issues**:
   - Verify file formats (PDF, TXT, MD)
   - Check file size limits
   - Ensure proper file encoding for text files

3. **API Key Errors**:
   - Verify your API keys are correct in `config.py`
   - Check API key permissions and quotas

4. **Memory Issues**:
   - Large documents may require more RAM
   - Consider reducing chunk sizes in config

### Debug Mode

Run with debug flags for more information:

```bash
# Backend with debug
uvicorn backend.backend:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Frontend with debug
streamlit run frontend/frontend.py --logger.level debug
```

## 🔒 Security Notes

- Keep your API keys secure and never commit them to version control
- The application runs locally by default
- For production deployment, implement proper authentication
- Consider using environment variables for sensitive configuration

## 🤝 Contributing

Feel free to contribute by:
- Adding new document loaders
- Improving the UI/UX
- Adding new LLM providers
- Optimizing performance
- Adding tests

## 📄 License

This project is open source. Feel free to use and modify as needed!

---

**Enjoy your new RAG chatbot! 🎉**

For questions or issues, please check the troubleshooting section or create an issue.
This is a question answer AI bot created using RAG architecture.

Steps for setup environment for RAG:
%pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph

signup on langsmith

export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="API_KEY"

create API keys on OpenAI and Gemini if you are using their models.
