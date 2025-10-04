"""
Modular RAG Service for dynamic document loading and question answering.
"""

import os
import uuid
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import shutil

import config.config as config
import bs4
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import TypedDict

# For better content extraction
import requests
from readability import Document as ReadabilityDocument


class RAGService:
    """
    A modular RAG service that supports dynamic document loading and question answering.
    """

    def __init__(self):
        # Set USER_AGENT environment variable early
        os.environ["USER_AGENT"] = config.USER_AGENT
        
        # Set environment variables for API keys
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
        if not config.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

        # Initialize components lazily to avoid startup delays
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.graph = None
        self._initialized = False
        
        # Persistence settings
        self.storage_dir = Path(config.VECTOR_STORE_PATH)
        self.storage_dir.mkdir(exist_ok=True)
        self.vector_store_path = self.storage_dir / "vector_store"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # Initialize text splitter (lightweight)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            add_start_index=True,
        )
        
        # Initialize prompt template (lightweight)
        self.prompt_template = PromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.
        Always thank the user for asking at the end of the answer.
        {context}

        Question: {question}

        Helpful Answer:""")
        
        # Track loaded documents - load from disk if available
        self.loaded_documents: Dict[str, List[str]] = {}
        self._load_metadata()
    
    def _ensure_initialized(self):
        """Lazy initialization of heavy components."""
        if self._initialized:
            return
        
        print("🔄 Initializing RAG service components...")
        
        # Initialize models
        self.llm = init_chat_model(config.MODEL_NAME, model_provider="google_genai")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        # Initialize vector store (load from disk if available)
        self.vector_store = self._initialize_vector_store()
        
        # Initialize LangGraph
        self.graph = self._initialize_graph()
        
        self._initialized = True
        print("✅ RAG service initialized successfully!")
    
    def _load_metadata(self):
        """Load metadata from disk if available."""
        print("Loading Metadata ... ")
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.loaded_documents = data.get('loaded_documents', {})
                self.vector_store = self._initialize_vector_store()
                print(f"📄 Loaded metadata for {len(self.loaded_documents)} sessions")
        except Exception as e:
            print(f"⚠️ Could not load metadata: {e}")
            self.loaded_documents = {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            data = {
                'loaded_documents': self.loaded_documents,
                'timestamp': str(uuid.uuid4())  # For versioning
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save metadata: {e}")
        
    def _initialize_vector_store(self) -> FAISS:
        """Initialize FAISS vector store, loading from disk if available."""
        print("Loading Vector Store...")
        try:
            # Try to load existing vector store
            if self.vector_store_path.exists():
                print("📂 Loading existing vector store from disk...")
                vector_store = FAISS.load_local(
                    str(self.vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded vector store with {vector_store.index.ntotal} documents")
                return vector_store
        except Exception as e:
            print(f"⚠️ Could not load existing vector store: {e}")
        
        # Create new vector store
        print("🆕 Creating new vector store...")
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        try:
            if self.vector_store and self._initialized:
                self.vector_store.save_local(str(self.vector_store_path))
                print(f"💾 Saved vector store to {self.vector_store_path}")
        except Exception as e:
            print(f"⚠️ Could not save vector store: {e}")
        
    def _initialize_graph(self) -> Any:
        """Initialize LangGraph for RAG pipeline."""
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt_template.invoke({
                "question": state["question"], 
                "context": docs_content
            })
            response = self.llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    
    def _extract_web_content(self, url: str) -> str:
        """Extract main content from any webpage using multiple strategies."""
        try:
            print(f"🌐 Extracting content from: {url}")
            
            # Set headers to mimic a real browser
            headers = {
                'User-Agent': config.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Fetch the webpage
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Import BeautifulSoup at the top of method to avoid scoping issues
            from bs4 import BeautifulSoup
            
            # Strategy 1: Use readability-lxml for smart content extraction
            try:
                # Convert response content to string for readability
                content_str = response.text  # Use .text instead of .content to get string
                doc = ReadabilityDocument(content_str)
                content = doc.summary()
                
                # Remove HTML tags
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text.strip()) > 100:  # If we got substantial content
                    print(f"✅ Extracted {len(text)} characters using readability")
                    return text
            except Exception as e:
                print(f"⚠️ Readability extraction failed: {e}")
            
            # Strategy 2: Fallback to BeautifulSoup with multiple selectors
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            
            # Try multiple content selectors in order of preference
            content_selectors = [
                'article',
                '.mw-parser-output',  # Wikipedia/MediaWiki content
                '.content',
                '.main-content', 
                '.post-content',
                '.entry-content',
                '.article-content',
                '.page-content',
                '[role="main"]',
                'main',
                '.container',
                '#content',
                '.wiki-content',  # Fandom wikis
                'body'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from the first matching element
                    text = elements[0].get_text(separator=' ', strip=True)
                    if len(text.strip()) > 100:  # Minimum content threshold
                        print(f"✅ Extracted {len(text)} characters using selector: {selector}")
                        return text
            
            # Strategy 3: Last resort - get all text from body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text.strip()) > 100:
                    print(f"✅ Extracted {len(text)} characters from body")
                    return text
            
            print("⚠️ No substantial content found on the webpage")
            return "No content could be extracted from this webpage."
            
        except Exception as e:
            print(f"❌ Error extracting content from {url}: {e}")
            return f"Error extracting content: {str(e)}"
    
    def load_pdf_file(self, file_path: str, session_id: str = "default") -> Dict[str, Any]:
        """Load and process a PDF file."""
        self._ensure_initialized()
        
        try:
            # Extract text from PDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
            doc.close()
            
            if not text.strip():
                return {"success": False, "error": "No text found in PDF"}
            
            # Split into chunks
            documents = [Document(page_content=text, metadata={"source": file_path, "type": "pdf"})]
            splits = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=splits)
            
            # Track loaded documents
            if session_id not in self.loaded_documents:
                self.loaded_documents[session_id] = []
            self.loaded_documents[session_id].extend(document_ids)
            
            # Save to disk
            self._save_vector_store()
            self._save_metadata()
            
            return {
                "success": True,
                "chunks_added": len(splits),
                "document_ids": document_ids,
                "message": f"Successfully loaded PDF with {len(splits)} chunks (saved to disk)"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def load_text_file(self, file_path: str, session_id: str = "default") -> Dict[str, Any]:
        """Load and process a text file."""
        self._ensure_initialized()
        
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({"type": "text", "session_id": session_id})
            
            # Split into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=splits)
            
            # Track loaded documents
            if session_id not in self.loaded_documents:
                self.loaded_documents[session_id] = []
            self.loaded_documents[session_id].extend(document_ids)
            
            # Save to disk
            self._save_vector_store()
            self._save_metadata()
            
            return {
                "success": True,
                "chunks_added": len(splits),
                "document_ids": document_ids,
                "message": f"Successfully loaded text file with {len(splits)} chunks (saved to disk)"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def load_web_urls(self, urls: List[str], session_id: str = "default") -> Dict[str, Any]:
        """Load and process web URLs with improved content extraction."""
        self._ensure_initialized()
        
        try:
            documents = []
            
            for url in urls:
                # Extract content using our improved method
                content = self._extract_web_content(url)
                
                if content and len(content.strip()) > 50:  # Minimum content check
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "type": "web",
                            "session_id": session_id
                        }
                    )
                    documents.append(doc)
                else:
                    print(f"⚠️ Minimal content extracted from {url}")
            
            if not documents:
                return {"success": False, "error": "No content could be extracted from the provided URLs"}
            
            # Split into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=splits)
            
            # Track loaded documents
            if session_id not in self.loaded_documents:
                self.loaded_documents[session_id] = []
            self.loaded_documents[session_id].extend(document_ids)
            
            # Save to disk
            self._save_vector_store()
            self._save_metadata()
            
            return {
                "success": True,
                "urls_processed": len(documents),  # URLs with content
                "chunks_added": len(splits),
                "document_ids": document_ids,
                "message": f"Successfully loaded {len(documents)}/{len(urls)} URLs with {len(splits)} chunks (saved to disk)"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ask_question(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Ask a question using the RAG pipeline."""
        self._ensure_initialized()
        
        try:
            result = self.graph.invoke({"question": question})
            
            # Extract context information
            context_info = []
            for doc in result.get("context", []):
                context_info.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown")
                })
            
            return {
                "success": True,
                "question": question,
                "answer": result["answer"],
                "context": context_info,
                "session_id": session_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_session_info(self, session_id: str = "default") -> Dict[str, Any]:
        """Get information about loaded documents in a session."""
        # Only initialize if we have a vector store to check
        if self._initialized and self.vector_store:
            total_docs = self.vector_store.index.ntotal
        else:
            total_docs = 0
        return {
            "session_id": session_id,
            "documents_loaded": len(self.loaded_documents.get(session_id, [])),
            "total_documents_in_store": total_docs
        }
    
    def clear_session(self, session_id: str = "default") -> Dict[str, Any]:
        """Clear documents for a specific session (simplified - clears all for now)."""
        try:
            # For simplicity, we'll reinitialize the vector store
            # In a production system, you'd want more sophisticated session management
            if self._initialized:
                self.vector_store = self._initialize_vector_store()
            
            # Clear the storage directory
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(exist_ok=True)
            
            self.loaded_documents.clear()
            
            # Save empty metadata
            self._save_metadata()
            
            return {
                "success": True,
                "message": f"Session {session_id} cleared successfully (storage cleared from disk)"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to temporary location and return path."""
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(temp_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors


# Global RAG service instance
rag_service = RAGService()
