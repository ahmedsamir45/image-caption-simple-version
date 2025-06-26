@echo off
echo === AI Image Captioning Environment Setup ===
echo Optimized for weak devices

echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo Removing existing virtual environment if it exists...
if exist venv rmdir /s /q venv

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install -r ai_requirements.txt

echo.
echo Installing additional optimizations for weak devices...
pip install tensorflow-cpu==2.15.0
pip install psutil
pip install memory-profiler

echo.
echo === Setup Complete! ===
echo.
echo Next steps:
echo 1. Test environment: python test_environment.py
echo 2. Check memory: python memory_check.py
echo 3. Start training: python optimized_train.py
echo.
echo To activate environment later: venv\Scripts\activate.bat
pause 