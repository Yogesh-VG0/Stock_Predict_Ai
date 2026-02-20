# Run the same steps as .github/workflows/daily-predictions.yml locally.
# Usage: from repo root (Stock_Predict_Ai):
#   .\.github\scripts\run-daily-pipeline-local.ps1
# For quick test without MongoDB (yfinance only, 3 tickers):
#   .\.github\scripts\run-daily-pipeline-local.ps1 -Quick
#
# Ensure .env has MONGODB_URI and API keys for full run. With -Quick, only Python deps are needed.

param(
    [switch]$Quick  # Use --no-mongo, train/predict 3 tickers only (no verify step)
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot -replace '[\\/]\.github[\\/]scripts$', ''
if (-not (Test-Path "$RepoRoot\ml_backend")) {
    $RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
}
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
Write-Host "Repo root: $RepoRoot"
Write-Host "PYTHONPATH: $env:PYTHONPATH"

# Step 1: Sentiment cron (non-fatal in CI)
Write-Host ''; Write-Host '===== Running sentiment cron =====' -ForegroundColor Cyan
try {
    python -m ml_backend.sentiment_cron
} catch {
    Write-Host ('Sentiment cron failed (non-fatal): ' + $_.Exception.Message)
}

# Step 2: Train
Write-Host ''; Write-Host '===== Training =====' -ForegroundColor Cyan
if ($Quick) {
    python -m ml_backend.scripts.run_pipeline --tickers AAPL MSFT NVDA --no-mongo
} else {
    python -m ml_backend.scripts.run_pipeline --all-tickers --no-predict
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 3: Generate predictions (batches)
if (-not $Quick) {
    $Batches = @(
        'AAPL MSFT NVDA GOOGL META AVGO ORCL CRM AMD INTC',
        'CSCO ADBE QCOM TXN NOW INTU AMZN TSLA HD NFLX',
        'LOW SBUX NKE MCD DIS BKNG TGT JPM V MA',
        'BAC WFC GS MS AXP BLK SCHW C COF BK',
        'MET AIG USB XOM CVX COP JNJ UNH LLY PFE',
        'ABBV ABT TMO DHR MRK AMGN GILD ISRG MDT BMY',
        'CVS WMT COST PG KO PEP MDLZ CL MO CAT',
        'HON UNP BA RTX LMT DE GE GD EMR FDX',
        'UPS MMM CMCSA VZ T CHTR BRK-B ACN IBM PYPL',
        'LIN NEE SO DUK AMT SPG PLTR TMUS PM AMAT'
    )
    $FailedBatches = @()
    for ($i = 0; $i -lt $Batches.Count; $i++) {
        $batch = $Batches[$i]
        $num = $i + 1
        Write-Host ''; Write-Host ("===== Batch $num/$($Batches.Count): $batch =====") -ForegroundColor Cyan
        $success = $false
        foreach ($attempt in 1..3) {
            Write-Host "  Attempt $attempt..."
            python -m ml_backend.scripts.run_pipeline --predict-only --tickers $batch.Split()
            if ($LASTEXITCODE -eq 0) { $success = $true; break }
            Write-Host "  Attempt $attempt failed."
            Start-Sleep -Seconds ($attempt * 20)
        }
        if (-not $success) {
            Write-Host "  *** Batch $num FAILED after 3 attempts ***" -ForegroundColor Red
            $FailedBatches += $num
        }
        if ($num -lt $Batches.Count) { Start-Sleep -Seconds 5 }
    }
    Write-Host ''; Write-Host '===== Pipeline Summary =====' -ForegroundColor Cyan
    Write-Host ('Failed batches: ' + $FailedBatches.Count)
    if ($FailedBatches.Count -gt 3) {
        Write-Host 'Too many failures - exiting.'
        exit 1
    }
} else {
    Write-Host ''; Write-Host '(Quick mode: skipping prediction batches)'
}

# Step 4: Verify predictions (requires MongoDB; skip in Quick)
if (-not $Quick -and $env:MONGODB_URI) {
    Write-Host ''; Write-Host '===== Verify predictions =====' -ForegroundColor Cyan
    $verifyScript = Join-Path $PSScriptRoot 'verify_predictions_fresh.py'
    python $verifyScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Step 5: SHAP (one batch in Quick, all in full)
Write-Host ''; Write-Host '===== SHAP analysis =====' -ForegroundColor Cyan
if ($Quick) {
    python -m ml_backend.explain.shap_analysis --tickers AAPL MSFT NVDA 2>$null; if ($LASTEXITCODE -ne 0) { Write-Host 'SHAP failed (non-fatal)' }
} else {
    $ShapBatches = @(
        'AAPL MSFT NVDA GOOGL META AVGO ORCL CRM AMD INTC',
        'CSCO ADBE QCOM TXN NOW INTU AMZN TSLA HD NFLX',
        'LOW SBUX NKE MCD DIS BKNG TGT JPM V MA',
        'BAC WFC GS MS AXP BLK SCHW C COF BK',
        'MET AIG USB XOM CVX COP JNJ UNH LLY PFE',
        'ABBV ABT TMO DHR MRK AMGN GILD ISRG MDT BMY',
        'CVS WMT COST PG KO PEP MDLZ CL MO CAT',
        'HON UNP BA RTX LMT DE GE GD EMR FDX',
        'UPS MMM CMCSA VZ T CHTR BRK-B ACN IBM PYPL',
        'LIN NEE SO DUK AMT SPG PLTR TMUS PM AMAT'
    )
    $shapFailed = 0
    foreach ($b in $ShapBatches) {
        python -m ml_backend.explain.shap_analysis --tickers $b.Split() 2>$null
        if ($LASTEXITCODE -ne 0) { $shapFailed++; Write-Host ('SHAP batch failed: ' + $b) }
    }
    Write-Host ('SHAP batches failed: ' + $shapFailed + ' / ' + $ShapBatches.Count)
    if ($shapFailed -gt 5) { exit 1 }
}

# Step 6: Generate AI explanations
Write-Host ''; Write-Host '===== Generating AI explanations =====' -ForegroundColor Cyan
try {
    python -m ml_backend.scripts.generate_explanations
} catch {
    Write-Host ('Explanation generation failed (non-fatal): ' + $_.Exception.Message)
}

# Step 7: Evaluate stored predictions
Write-Host ''; Write-Host '===== Evaluate stored predictions =====' -ForegroundColor Cyan
python -m ml_backend.scripts.evaluate_models --stored --days 60 2>$null; if ($LASTEXITCODE -ne 0) { Write-Host 'Evaluation failed (non-fatal)' }

# Step 8: Drift monitor
Write-Host ''; Write-Host '===== Drift monitor =====' -ForegroundColor Cyan
python -m ml_backend.scripts.drift_monitor 2>$null; if ($LASTEXITCODE -ne 0) { Write-Host 'Drift monitor failed (non-fatal)' }

# Step 9: Data retention
Write-Host ''; Write-Host '===== Data retention cleanup =====' -ForegroundColor Cyan
try {
    python -m ml_backend.scripts.data_retention
} catch {
    Write-Host ('Retention cleanup failed (non-fatal): ' + $_.Exception.Message)
}

Write-Host ''
Write-Host '===== Local pipeline run complete =====' -ForegroundColor Green
