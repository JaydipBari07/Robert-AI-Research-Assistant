# config.py

# API Configuration
GOOGLE_API_KEY = ""  # Set your Google Gemini API Key
OPENAI_API_KEY = ""  # Optional for OpenAI models

# Model Configuration
MODEL_NAME = "gemini-2.5-flash"  # Choose model: e.g., 'gemini-2.5-flash' or 'gpt-4'

# Embedding Model (for text/vector embeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace model for embeddings

# FAISS Vector Store Configuration
VECTOR_STORE_PATH = "./VectorDBStorage"  # Path to store the FAISS index on disk, for persistence

# Document Loaders (URLs, PDFs, etc.)
WEB_URLS = ["https://adeepdiveintohp.wordpress.com/"]  # List of URLs to load
PDF_FILE_PATH = "./a-review-paper-on-holography-IJERTCONV4IS32002.pdf"  # Path to the PDF file for testing

# Chunking Configuration
CHUNK_SIZE = 1000  # Size of chunks for documents
CHUNK_OVERLAP = 200  # Overlap between chunks for better context retention

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"