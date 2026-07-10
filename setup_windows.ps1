# AutoML-Agent — Windows setup wrapper
# Installs dependencies, creates workspace dirs, registers Jupyter kernel.

param(
    [ValidateSet("minimal", "full")]
    [string]$Profile = "minimal"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

& "$RepoRoot\run_project.ps1" -Setup -Profile $Profile
