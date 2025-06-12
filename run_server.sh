#!/bin/bash

# Background Removal API Server Startup Script

echo "🚀 Starting Background Removal API Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set environment variable to disable albumentations update check
export NO_ALBUMENTATIONS_UPDATE=1

# Start the server
echo "🌟 Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation available at: http://localhost:8000/docs"
echo "🌐 Web Interface: Open index.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python server_app.py 