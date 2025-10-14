@echo off
echo Starting RAG Service...
echo.

REM Navigate to project directory
cd /d "C:\Pranav\stuff\cursorkonduit"

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if Ollama is already running
echo Checking if Ollama is already running...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama is already running on port 11434!
    goto :ollama_ready
)

REM Start Ollama in background
echo Starting Ollama...
start "Ollama" /min ollama serve

REM Wait for Ollama to start
echo Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

REM Check if Ollama started successfully
echo Checking Ollama status...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama failed to start
    pause
    exit /b 1
)

echo Ollama started successfully!

:ollama_ready

REM Check if qwen3:4b model is available
echo Checking for qwen3:4b model...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 -UseBasicParsing; if ($response.Content -match 'qwen3') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: qwen3:4b model not found!
    echo Please run: ollama pull qwen3:4b
    echo.
    echo Do you want to download it now? (y/n)
    set /p choice=
    if /i "%choice%"=="y" (
        echo Downloading qwen3:4b model...
        ollama pull qwen3:4b
        if %errorlevel% neq 0 (
            echo ERROR: Failed to download qwen3:4b model
            pause
            exit /b 1
        )
        echo Model downloaded successfully!
    ) else (
        echo Please download the model manually and restart the service
        pause
        exit /b 1
    )
)

echo qwen3:4b model is available!

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
