#!/usr/bin/env pwsh
# Clang Format Manager - PowerShell Launcher
# Simply run this script to format modified C/C++ files interactively

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "clang_format_manager.py"

Write-Host "Starting Clang Format Manager..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python or ensure it's in your PATH" -ForegroundColor Yellow
    exit 1
}

# Run the Python script
python $pythonScript

# Preserve exit code
exit $LASTEXITCODE
