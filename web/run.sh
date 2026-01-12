#!/bin/bash

echo "==================================="
echo "  Nahawi Web Editor Startup"
echo "==================================="
echo

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org"
    exit 1
fi

# Check for model files
MODEL_PATH="$PROJECT_ROOT/models/base/fasih_v15_model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at $MODEL_PATH"
    echo "The demo will fail until model files are available."
    echo "See README.md for download instructions."
    echo
fi

echo "Installing backend dependencies..."
cd "$SCRIPT_DIR/backend"
pip install -r requirements.txt -q

echo "Starting Backend Server..."
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to initialize (10 seconds)..."
sleep 10

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend failed to start"
    exit 1
fi

echo "Installing frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi

echo "Starting Frontend Dev Server..."
npm run dev &
FRONTEND_PID=$!

echo
echo "==================================="
echo "  Servers are running..."
echo "==================================="
echo
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/api/health"
echo "Frontend:     http://localhost:5173"
echo
echo "Press Ctrl+C to stop all servers"

# Cleanup on exit
cleanup() {
    echo "Shutting down servers..."
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM
wait
