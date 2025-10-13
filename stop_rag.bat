@echo off
echo Stopping RAG Service...

REM Kill Python processes
echo Stopping Python processes...
taskkill /f /im python.exe >nul 2>&1

REM Kill Ollama processes
echo Stopping Ollama...
taskkill /f /im ollama.exe >nul 2>&1

REM Kill any processes using port 8000
echo Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Kill any processes using port 11434
echo Checking port 11434...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :11434') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo All services stopped!
pause
