# AutoML-Agent — Ollama setup for Windows 11 (no OpenAI API required)

param(
    [string]$Model = "llama3.1:8b",
    [ValidateSet("minimal", "full")]
    [string]$Profile = "minimal"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== AutoML-Agent Ollama Setup ===" -ForegroundColor Cyan

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed." -ForegroundColor Red
    Write-Host "Download from: https://ollama.com/download/windows"
    Write-Host "After install, re-run: .\setup_ollama.ps1"
    exit 1
}

Write-Host "Pulling model: $Model ..." -ForegroundColor Cyan
ollama pull $Model

Write-Host "Starting Ollama service check..." -ForegroundColor Cyan
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "Ollama is running. Models: $($tags.models.name -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

& "$RepoRoot\run_project.ps1" -Setup -Profile $Profile

Write-Host ""
Write-Host "Verifying local LLM connection..." -ForegroundColor Cyan
& ".\venv\Scripts\python.exe" -c @"
from configs import Configs, AVAILABLE_LLMs
from local_llm import is_local_llm_running, list_local_models
print('USE_LOCAL_LLM=', Configs.USE_LOCAL_LLM)
print('Default model=', AVAILABLE_LLMs['local']['model'])
print('Ollama running=', is_local_llm_running())
print('Models=', list_local_models())
"@

Write-Host ""
Write-Host "Setup complete. Run the project with:" -ForegroundColor Green
Write-Host "  .\run_local.ps1" -ForegroundColor Green
