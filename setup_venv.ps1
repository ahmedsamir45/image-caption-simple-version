Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r ai_requirements.txt

Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host "To activate the virtual environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Read-Host "Press Enter to continue" 