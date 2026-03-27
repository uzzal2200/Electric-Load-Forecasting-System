# PowerShell script to set up directories and prepare for data preprocessing

# Create directories if they don't exist
Write-Host "Creating directories..." -ForegroundColor Green
if (-not (Test-Path -Path "processed_data")) {
    New-Item -Path "processed_data" -ItemType Directory | Out-Null
    Write-Host "Created processed_data directory" -ForegroundColor Cyan
} else {
    Write-Host "processed_data directory already exists" -ForegroundColor Cyan
}

# Check if data files exist
Write-Host "`nChecking data files..." -ForegroundColor Green
if (Test-Path -Path "archive") {
    $texasFiles = Get-ChildItem -Path "archive" -Filter "texas*.csv"
    $jsonFiles = Get-ChildItem -Path "archive" -Filter "*.json"
    $eiaFiles = Get-ChildItem -Path "archive" -Filter "EIA*.csv"
    
    Write-Host "Found $($texasFiles.Count) Texas electricity data files" -ForegroundColor Cyan
    Write-Host "Found $($jsonFiles.Count) weather data JSON files" -ForegroundColor Cyan
    Write-Host "Found $($eiaFiles.Count) EIA data files" -ForegroundColor Cyan
} else {
    Write-Host "Warning: archive directory not found. Make sure your data files are in the correct location." -ForegroundColor Yellow
}

# Check Python installation
Write-Host "`nChecking Python installation..." -ForegroundColor Green
try {
    $pythonVersion = python --version
    Write-Host "Python is installed: $pythonVersion" -ForegroundColor Cyan
    
    # Check required Python modules
    Write-Host "`nChecking required Python modules..." -ForegroundColor Green
    $modules = @("pandas", "numpy", "matplotlib", "seaborn", "sklearn", "scipy")
    $missingModules = @()
    
    foreach ($module in $modules) {
        $moduleCheck = python -c "import $module" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Module $module is installed" -ForegroundColor Cyan
        } else {
            Write-Host "Module $module is NOT installed" -ForegroundColor Red
            $missingModules += $module
        }
    }
    
    if ($missingModules.Count -gt 0) {
        Write-Host "`nMissing modules: $($missingModules -join ', ')" -ForegroundColor Yellow
        Write-Host "Please install missing modules using: pip install $($missingModules -join ' ')" -ForegroundColor Yellow
    } else {
        Write-Host "`nAll required Python modules are installed!" -ForegroundColor Green
    }
} catch {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.7 or higher." -ForegroundColor Red
}

# Instructions for running the data preprocessing pipeline
Write-Host "`nTo run the data preprocessing pipeline:" -ForegroundColor Green
Write-Host "1. Make sure all required Python modules are installed" -ForegroundColor Cyan
Write-Host "2. Run: python scripts/data_preprocessing.py" -ForegroundColor Cyan
Write-Host "3. Run: python scripts/merge_data.py" -ForegroundColor Cyan
Write-Host "4. Run: python scripts/anomaly_detection.py" -ForegroundColor Cyan
Write-Host "`nOr run the full pipeline with: python scripts/run_preprocessing.py" -ForegroundColor Cyan

Write-Host "`nSetup completed!" -ForegroundColor Green 