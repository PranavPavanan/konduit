@echo off
echo Starting RAG Service...
echo.

REM Navigate to project directory
cd /d "C:\Pranav\stuff\cursorkonduit"

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Start Ollama in background
echo Starting Ollama...
start "Ollama" /min ollama serve

REM Wait for Ollama to start
echo Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

REM Check if Ollama is running
echo Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama failed to start
    pause
    exit /b 1
)

echo Ollama is running!

REM Start the RAG service
echo Starting RAG service...
echo.
echo Service will be available at:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo   - Health: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the service
echo.

python main.py

pause
