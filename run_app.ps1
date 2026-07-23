$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create one with: python -m venv .venv" -ForegroundColor Yellow
}
.
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
