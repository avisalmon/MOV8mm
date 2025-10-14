# 8mm Film Enhancement - Windows Setup Script
# Run this script in PowerShell to set up your environment

Write-Host "=== 8mm Film Enhancement Setup ===" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    exit 1
}

# Check if we're in the correct directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "✗ requirements.txt not found. Please run this script from the project directory." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Setting up virtual environment..." -ForegroundColor Yellow

# Create virtual environment
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting execution policy for current user..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host ""
Write-Host "Installing Python packages (this may take several minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "⚠ Some packages may have failed to install. Check the output above." -ForegroundColor Yellow
}

# Run Python setup script
Write-Host ""
Write-Host "Running setup script to download models..." -ForegroundColor Yellow
python setup.py

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Place your 8mm MP4 files in the 'input_videos' folder" -ForegroundColor White
Write-Host "2. Review settings in config.py" -ForegroundColor White  
Write-Host "3. Run enhancement scripts (to be created next)" -ForegroundColor White
Write-Host ""
Write-Host "To reactivate the virtual environment later:" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Hardware detected:" -ForegroundColor White
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"} | Select-Object -First 1
    if ($gpu) {
        Write-Host "✓ NVIDIA GPU: $($gpu.Name)" -ForegroundColor Green
    } else {
        Write-Host "⚠ No NVIDIA GPU detected. CPU processing will be much slower." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not detect GPU information" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")