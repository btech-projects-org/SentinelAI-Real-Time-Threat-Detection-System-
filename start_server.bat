@echo off
setlocal
title SentinelAI - Threat Detection System

echo ========================================================
echo   SENTINELAI AUTO-STARTUP SCRIPT
echo ========================================================

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

:: 2. Check/Create Virtual Environment
if not exist "venv" (
    echo [SETUP] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [SETUP] Virtual environment created.
) else (
    echo [INFO] Virtual environment found.
)

:: 3. Activate Virtual Environment
call venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [SETUP] Checking and installing dependencies...
echo         (This may take a while for the first time)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    echo Please check your internet connection or requirements.txt.
    pause
    exit /b
)

:: 5. Launch Application
echo.
echo [INFO] Dependencies installed. Launching application...
python validate_and_run.py

pause
