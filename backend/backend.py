"""
FastAPI Backend for RAG Chatbot Application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid

from .rag_service import rag_service

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    success: bool
    question: str
    answer: str
    context: List[dict]
    session_id: str
    error: Optional[str] = None

class URLRequest(BaseModel):
    urls: List[str]
    session_id: Optional[str] = "default"

class LoadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: Optional[int] = None
    error: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    documents_loaded: int
    total_documents_in_store: int

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "upload_file": "/upload-file",
            "load_urls": "/load-urls",
            "session_info": "/session/{session_id}",
            "clear_session": "/clear-session/{session_id}"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for asking questions."""
    try:
        result = rag_service.ask_question(request.question, request.session_id)
        
        if result["success"]:
            return ChatResponse(
                success=True,
                question=result["question"],
                answer=result["answer"],
                context=result["context"],
                session_id=result["session_id"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file", response_model=LoadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form("default")
):
    """Upload and process a file (PDF or text)."""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.md']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Save file temporarily
        temp_file_path = rag_service.save_uploaded_file(file_content, file.filename)
        
        try:
            # Process based on file type
            if file_extension == '.pdf':
                result = rag_service.load_pdf_file(temp_file_path, session_id)
            else:  # .txt, .md
                result = rag_service.load_text_file(temp_file_path, session_id)
            
            if result["success"]:
                return LoadResponse(
                    success=True,
                    message=result["message"],
                    chunks_added=result["chunks_added"]
                )
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        finally:
            # Clean up temporary file
            rag_service.cleanup_temp_file(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-urls", response_model=LoadResponse)
async def load_urls(request: URLRequest):
    """Load and process URLs."""
    try:
        if not request.urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        # Basic URL validation
        for url in request.urls:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
        
        result = rag_service.load_web_urls(request.urls, request.session_id)
        
        if result["success"]:
            return LoadResponse(
                success=True,
                message=result["message"],
                chunks_added=result["chunks_added"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a session."""
    try:
        info = rag_service.get_session_info(session_id)
        return SessionInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear all documents for a session."""
    try:
        result = rag_service.clear_session(session_id)
        
        if result["success"]:
            return {"message": result["message"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
