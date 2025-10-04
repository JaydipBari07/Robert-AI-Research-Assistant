@echo off
echo Starting RAG Chatbot Backend...
echo.

echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI backend server...
echo Backend will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

uvicorn backend.backend:app --host 0.0.0.0 --port 8000 --reload
