param(
    [string]$ImagesDir = "bounding_box_test",
    [int]$Limit = 10,
    [string]$Model = "qwen/qwen3-vl-32b-instruct",
    [double]$SleepSeconds = 0.25
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment python not found at $Python"
}

$OutputDir = Join-Path $Root "sample_outputs"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

& $Python (Join-Path $Root "eval_qwen3_vl_openrouter.py") `
    --dataset-root $Root `
    --images-dir (Join-Path $Root $ImagesDir) `
    --limit $Limit `
    --model $Model `
    --sleep-seconds $SleepSeconds `
    --output-mat (Join-Path $OutputDir "gallery_market_sample.mat") `
    --metrics-json (Join-Path $OutputDir "market_attribute_metrics_sample.json") `
    --cache-dir (Join-Path $Root ".openrouter_cache")
