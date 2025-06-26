@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r ai_requirements.txt

echo Virtual environment setup complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
pause 