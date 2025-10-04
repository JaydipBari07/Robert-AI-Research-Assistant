"""
Streamlit Frontend for RAG Chatbot Application
"""

import streamlit as st
import requests
import uuid
from typing import List, Dict
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="🤖 Robert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stChat {
        background-color: #f8f9fa;
    }
    .upload-section {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .session-info {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = 0

def make_api_request(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to the backend server. Please make sure it's running on http://localhost:8000"
    except Exception as e:
        return False, str(e)

def display_chat_message(message: dict, is_user: bool = False):
    """Display a chat message with proper styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show context if available
        if message.get('context'):
            with st.expander("📄 View Context Sources"):
                for i, ctx in enumerate(message['context']):
                    st.markdown(f"""
                    **Source {i+1}:** {ctx.get('source', 'Unknown')} ({ctx.get('type', 'Unknown')})
                    
                    {ctx.get('content', 'No content available')}
                    """)

def upload_file_section():
    """File upload section."""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=['pdf', 'txt', 'md'],
        help="Upload PDF, TXT, or Markdown files"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload and Process", type="primary"):
            with st.spinner("Processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"session_id": st.session_state.session_id}
                
                success, result = make_api_request("/upload-file", "POST", data=data, files=files)
                
                if success:
                    st.success(f"✅ {result['message']}")
                    st.session_state.documents_loaded += 1
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def url_input_section():
    """URL input section."""
    st.subheader("🌐 Load Web Content")
    st.caption("✨ Works with any website - no special configuration needed!")
    
    url_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com\nhttps://another-site.com\nhttps://news.ycombinator.com",
        height=100
    )
    
    if st.button("🔗 Load URLs", type="primary"):
        if url_input.strip():
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            with st.spinner("Loading web content..."):
                data = {
                    "urls": urls,
                    "session_id": st.session_state.session_id
                }
                
                success, result = make_api_request("/load-urls", "POST", data=data)
                
                if success:
                    st.success(f"✅ {result['message']}")
                    st.session_state.documents_loaded += len(urls)
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
        else:
            st.warning("Please enter at least one URL")

def get_session_info():
    """Get and display session information."""
    success, result = make_api_request(f"/session/{st.session_state.session_id}")
    
    if success:
        st.markdown('<div class="session-info">', unsafe_allow_html=True)
        st.subheader("📊 Session Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Loaded", result['documents_loaded'])
        with col2:
            st.metric("Total Chunks", result['total_documents_in_store'])
        
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        return result
    else:
        st.error(f"❌ Error getting session info: {result}")
        return None

def main():
    """Main application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 Robert")
    st.markdown("Ask questions about your uploaded documents and web content!")
    
    # Show persistence info
    # st.info("📚 **Persistent Storage Enabled**: Your documents are automatically saved and will be available after restart!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Document Management")
        
        # Session management
        if st.button("🔄 New Session", help="Start a new chat session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.documents_loaded = 0
            st.rerun()
        
        if st.button("🗑️ Clear Session", help="Clear all documents and chat history"):
            success, result = make_api_request(f"/clear-session/{st.session_state.session_id}", "DELETE")
            if success:
                st.session_state.chat_history = []
                st.session_state.documents_loaded = 0
                st.success("Session cleared!")
                st.rerun()
            else:
                st.error(f"Error: {result}")
        
        st.divider()
        
        # File upload
        upload_file_section()
        
        st.divider()
        
        # URL input
        url_input_section()
        
        st.divider()
        
        # Session info
        get_session_info()
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                display_chat_message({'content': message['content']}, is_user=True)
            else:
                display_chat_message(message)
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Display user message immediately
        display_chat_message({'content': user_question}, is_user=True)
        
        # Get response from API
        with st.spinner("🤔 Thinking..."):
            data = {
                "question": user_question,
                "session_id": st.session_state.session_id
            }
            
            success, result = make_api_request("/chat", "POST", data=data)
            
            if success:
                # Add bot response to chat history
                bot_message = {
                    'role': 'assistant',
                    'content': result['answer'],
                    'context': result.get('context', [])
                }
                st.session_state.chat_history.append(bot_message)
                
                # Display bot response
                display_chat_message(bot_message)
            else:
                error_message = {
                    'role': 'assistant',
                    'content': f"❌ Sorry, I encountered an error: {result}",
                    'context': []
                }
                st.session_state.chat_history.append(error_message)
                display_chat_message(error_message)
        
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
            Upload documents or provide URLs to get started!
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
