$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       PREMIER LEAGUE PREDICTOR - WEEKEND UPDATE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/5] Downloading latest 2026/27 match data..." -ForegroundColor Yellow
uv run python -m ingest.football_data_client --season 2026

Write-Host ""
Write-Host "[2/5] Cleaning match data..." -ForegroundColor Yellow
uv run python -m clean.clean_matches --season 2026

Write-Host ""
Write-Host "[3/5] Rebuilding features..." -ForegroundColor Yellow
uv run python -m features.feature_engineering --start-season 2024 --end-season 2026

Write-Host ""
Write-Host "[4/5] Retraining Dixon-Coles model..." -ForegroundColor Yellow
uv run python -m models.poisson_dixon_coles train --start-season 2024 --end-season 2026 --model-name 2024_2026

Write-Host ""
Write-Host "[5/5] Generating predictions for the next Gameweek..." -ForegroundColor Yellow
uv run python -m models.poisson_dixon_coles predict-gameweek --model-name 2024_2026 --season 2026

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "              WEEKEND PREDICTIONS COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to close"