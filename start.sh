#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Use $PORT if defined, else default 8000
PORT=${PORT:-8000}

# Run FastAPI
uvicorn app.main:app --host 0.0.0.0 --port $PORT
