#!/bin/bash

# Start the FastAPI backend server in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend server
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0