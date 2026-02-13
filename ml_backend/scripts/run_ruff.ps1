# ml_backend/scripts/run_ruff.ps1
[CmdletBinding()]
param(
    [switch]$Fix,
    [string]$Select = "F401,F841"
)

$ErrorActionPreference = "Stop"

# Script lives in ml_backend/scripts/; ml_backend is parent
$mlBackend = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $mlBackend

Write-Host "RepoRoot: $repoRoot"
Write-Host "MLBackend: $mlBackend"

Push-Location $repoRoot
try {
    py -V | Out-Host

    # Ensure ruff exists
    py -m ruff --version 2>$null | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing ruff..."
        py -m pip install -U ruff | Out-Host
    }

    $args = @("check", "ml_backend", "--select", $Select)
    if ($Fix) { $args += "--fix" }

    Write-Host ("Running: py -m ruff " + ($args -join " "))
    py -m ruff @args
}
finally {
    Pop-Location
}
