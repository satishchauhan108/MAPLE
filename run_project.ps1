# AutoML-Agent — Windows 11 setup and launch script
# Usage (from repo root):
#   .\run_project.ps1 -Setup
#   .\run_project.ps1 -RunNotebook
#   .\run_project.ps1 -RunExample -Task tabular_classification -DataPath "agent_workspace\datasets\banana_quality.csv"

param(
    [switch]$Setup,
    [switch]$RunNotebook,
    [switch]$RunExample,
    [string]$Task = "tabular_classification",
    [string]$DataPath = "",
    [string]$Llm = "gpt-4",
    [ValidateSet("minimal", "full")]
    [string]$Profile = "full"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"

$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$VenvPip = Join-Path $RepoRoot "venv\Scripts\pip.exe"

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        python -m venv venv
    }
}

function Install-Dependencies {
    param([string]$RequirementsFile)
    Write-Host "Installing from $RequirementsFile ..." -ForegroundColor Cyan
    & $VenvPip install --upgrade pip setuptools wheel
    & $VenvPip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    & $VenvPip install -r $RequirementsFile
    if ($RequirementsFile -eq "windows_requirements.txt") {
        & $VenvPip install torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
        & $VenvPip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
    }
}

function Ensure-Workspace {
    $dirs = @(
        "agent_workspace\datasets",
        "agent_workspace\exp",
        "agent_workspace\trained_models"
    )
    foreach ($d in $dirs) {
        $full = Join-Path $RepoRoot $d
        if (-not (Test-Path $full)) {
            New-Item -ItemType Directory -Path $full -Force | Out-Null
        }
    }
}

function Show-ConfigReminder {
    Write-Host ""
    Write-Host "=== Configuration required ===" -ForegroundColor Yellow
    Write-Host "1. Edit configs.py and set Configs.OPENAI_KEY (and SEARCHAPI_API_KEY for web search)."
    Write-Host "2. USE_OPENAI_PARSER is True by default on Windows (no vLLM required)."
    Write-Host "3. Place kaggle.json in %USERPROFILE%\.kaggle\ if using Kaggle retrieval."
    Write-Host "4. PapersWithCode data (_data/paperswithcode/) is optional; arXiv/web/Kaggle still work."
    Write-Host ""
}

if ($Setup) {
    Ensure-Venv
    $req = if ($Profile -eq "minimal") { "minimal_requirements.txt" } else { "windows_requirements.txt" }
    Install-Dependencies -RequirementsFile $req
    Ensure-Workspace
    & $VenvPython -m ipykernel install --user --name automl-agent --display-name "AutoML-Agent"
    Show-ConfigReminder
    Write-Host "Setup complete." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run: .\run_project.ps1 -Setup"
}

Ensure-Workspace

if ($RunNotebook) {
    Show-ConfigReminder
    & $VenvPython -m jupyter notebook AutoMLAgent.ipynb
    exit 0
}

if ($RunExample) {
    if (-not $DataPath) {
        Write-Error "Provide -DataPath to your dataset file or folder."
    }
    if (-not $env:OPENAI_API_KEY -and -not $env:OPENAI_KEY) {
        Write-Host "NOTE: Running in local LLM mode (Ollama). No OpenAI key required." -ForegroundColor Cyan
        Write-Host "Use .\run_local.ps1 instead for Ollama setup." -ForegroundColor Cyan
    }
    & $VenvPython run_automl.py --task $Task --llm $Llm --data-path $DataPath --n-plans 1 --n-revise 1
    exit $LASTEXITCODE
}

Write-Host @"
AutoML-Agent launcher

  .\run_project.ps1 -Setup                          Install deps (full profile)
  .\run_project.ps1 -Setup -Profile minimal           Install core deps only
  .\run_project.ps1 -RunNotebook                      Open AutoMLAgent.ipynb
  .\run_project.ps1 -RunExample -Task tabular_classification -DataPath "agent_workspace\datasets\data.csv"

"@
