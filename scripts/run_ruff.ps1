# PowerShell-safe ruff check. Run from project root.
# Usage: .\scripts\run_ruff.ps1
# Usage with auto-fix: .\scripts\run_ruff.ps1 -Fix

param([switch]$Fix)

# 1) Check Python
py -V
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found. Install from python.org and add to PATH."
    exit 1
}

# 2) Install ruff
py -m pip install -U ruff 2>$null | Out-Null

# 3) Run from repo root
Set-Location $PSScriptRoot\..

$targets = @("ml_backend")
if ($Fix) {
    py -m ruff check $targets --select F401,F841 --fix
} else {
    py -m ruff check $targets --select F401,F811,F821,F841
}
