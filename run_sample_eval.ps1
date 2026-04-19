param(
    [string]$ImagesDir = "",
    [int]$Limit = 10,
    [int]$SampleSize = 0,
    [int]$SampleSeed = 1501,
    [string]$SampleManifest = "",
    [string]$WriteSampleManifest = "",
    [string]$StratifyBy = "",
    [string]$Model = "qwen/qwen3.5-9b",
    [int]$MaxTokens = 8192,
    [double]$SleepSeconds = 0.25
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment python not found at $Python"
}

if ([string]::IsNullOrWhiteSpace($ImagesDir)) {
    if ($env:MARKET1501_IMAGES_DIR) {
        $ImagesDir = $env:MARKET1501_IMAGES_DIR
    } else {
        $ImagesDir = Join-Path $Root "data\Market-1501-v15.09.15\bounding_box_test"
    }
}

$OutputDir = Join-Path $Root "sample_outputs"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Command = @(
    (Join-Path $Root "eval_qwen3_vl_openrouter.py"),
    "--dataset-root", $Root,
    "--images-dir", $ImagesDir,
    "--limit", $Limit,
    "--model", $Model,
    "--max-tokens", $MaxTokens,
    "--sleep-seconds", $SleepSeconds,
    "--output-mat", (Join-Path $OutputDir "gallery_market_sample.mat"),
    "--metrics-json", (Join-Path $OutputDir "market_attribute_metrics_sample.json")
)

if ($SampleSize -gt 0) {
    $Command += @("--sample-size", $SampleSize, "--sample-seed", $SampleSeed)
}

if (-not [string]::IsNullOrWhiteSpace($SampleManifest)) {
    $Command += @("--sample-manifest", $SampleManifest)
}

if (-not [string]::IsNullOrWhiteSpace($WriteSampleManifest)) {
    $Command += @("--write-sample-manifest", $WriteSampleManifest)
}

if (-not [string]::IsNullOrWhiteSpace($StratifyBy)) {
    $Command += @("--stratify-by", $StratifyBy)
}

& $Python @Command
