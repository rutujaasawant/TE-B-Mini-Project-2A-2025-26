#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
