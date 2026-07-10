# AutoML-Agent — run with local Ollama (no OpenAI API key)

param(
    [string]$Task = "tabular_classification",
    [string]$DataPath = "agent_workspace\datasets\sample_banana.csv",
    [string]$Llm = "local",
    [string]$Model = "",
    [switch]$Setup,
    [switch]$Notebook
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"

$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"

if ($Setup) {
    & "$RepoRoot\setup_ollama.ps1"
    exit $LASTEXITCODE
}

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run: .\setup_ollama.ps1"
}

if ($Model) {
    $env:DEFAULT_LOCAL_MODEL = $Model
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "Ollama not found. Install from https://ollama.com/download/windows"
}

try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

foreach ($d in @("agent_workspace\datasets", "agent_workspace\exp", "agent_workspace\trained_models")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

if (-not (Test-Path $DataPath)) {
    Write-Host "Dataset not found at $DataPath - check agent_workspace/datasets/" -ForegroundColor Yellow
}

if ($Notebook) {
    & $VenvPython -m jupyter notebook AutoMLAgent.ipynb
    exit 0
}

Write-Host "Running AutoML-Agent with local LLM ($Llm)..." -ForegroundColor Cyan
& $VenvPython run_automl.py --task $Task --llm $Llm --data-path $DataPath --n-plans 1 --n-revise 1
