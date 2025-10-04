#!/bin/bash
echo "Starting RAG Chatbot Frontend..."
echo

echo "Make sure the backend is running on http://localhost:8000"
echo

echo "Starting Streamlit frontend..."
streamlit run frontend/frontend.py --server.port 8501
