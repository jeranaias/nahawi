@echo off
echo ===================================
echo   Nahawi Web Editor Startup
echo ===================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Check if Node is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

:: Check for model files
set MODEL_PATH=%~dp0..\models\base\fasih_v15_model.pt
if not exist "%MODEL_PATH%" (
    echo WARNING: Model file not found at %MODEL_PATH%
    echo The demo will fail until model files are available.
    echo See README.md for download instructions.
    echo.
)

echo Starting Backend Server...
start "Nahawi Backend" cmd /c "cd /d %~dp0backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000"

echo Waiting for backend to initialize (15 seconds)...
timeout /t 15 /nobreak >nul

echo Starting Frontend Dev Server...
start "Nahawi Frontend" cmd /c "cd /d %~dp0frontend && npm install && npm run dev"

echo.
echo ===================================
echo   Servers are starting...
echo ===================================
echo.
echo Backend API:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo Health Check: http://localhost:8000/api/health
echo Frontend:     http://localhost:5173
echo.
echo Press any key to open the web editor...
pause >nul

start http://localhost:5173
