param(
    [int]$MaxMinutes = 15,
    [string]$DatasetUrl = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-103/train.txt",
    [string]$CorpusPath = "data/benchmarks/wikitext103_train.txt",
    [string]$LongCorpusPath = "data/benchmarks/wikitext103_train_long.txt",
    [int]$MaxChars = 250000
)

$ErrorActionPreference = "Stop"

if ($MaxMinutes -lt 1) {
    throw "MaxMinutes must be >= 1"
}

$pythonCmd = "C:/Python313/python.exe"
if (-not (Test-Path $pythonCmd)) {
    $pythonCmd = "python"
}

$overall = [System.Diagnostics.Stopwatch]::StartNew()
$benchmarkDir = Split-Path -Parent $CorpusPath
if (-not (Test-Path $benchmarkDir)) {
    New-Item -ItemType Directory -Path $benchmarkDir | Out-Null
}
if (-not (Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts" | Out-Null
}

if (-not (Test-Path $CorpusPath)) {
    Write-Host "Downloading benchmark corpus from: $DatasetUrl"
    Invoke-WebRequest -Uri $DatasetUrl -OutFile $CorpusPath
}

& $pythonCmd -c "from pathlib import Path; src=Path(r'$CorpusPath'); dst=Path(r'$LongCorpusPath'); text=src.read_text(encoding='utf-8'); dst.write_text(text[:$MaxChars], encoding='utf-8'); print('prepared', dst, 'bytes', dst.stat().st_size)"

$size = (Get-Item $LongCorpusPath).Length
Write-Host "Using corpus: $LongCorpusPath ($size bytes)"

$remaining = [Math]::Max(1, ($MaxMinutes * 60) - [int]$overall.Elapsed.TotalSeconds)
$repoRoot = (Get-Location).Path
$env:PYTHONPATH = (Join-Path $repoRoot "src")

$args = @(
    "-m",
    "word2vec",
    "--corpus",
    $LongCorpusPath,
    "--benchmark-profile",
    "custom",
    "--embedding-dim",
    "32",
    "--num-negatives",
    "3",
    "--window-size",
    "3",
    "--stream-pairs",
    "--epochs",
    "1",
    "--benchmark-repeats",
    "1",
    "--benchmark-json",
    "artifacts/benchmark_long_real.json",
    "--benchmark-markdown",
    "artifacts/benchmark_long_real.md",
    "--queries",
    "word,vectors,language"
)

Write-Host "Running long benchmark with timeout: $remaining seconds"
$proc = Start-Process -FilePath $pythonCmd -ArgumentList $args -WorkingDirectory $repoRoot -NoNewWindow -PassThru

try {
    Wait-Process -Id $proc.Id -Timeout $remaining
} catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Long benchmark exceeded timeout budget (${MaxMinutes} minutes)."
}

$proc.Refresh()
if ($null -ne $proc.ExitCode -and $proc.ExitCode -ne 0) {
    throw "Long benchmark process failed with exit code $($proc.ExitCode)."
}

$totalSeconds = [math]::Round($overall.Elapsed.TotalSeconds, 2)
Write-Host "Completed in ${totalSeconds}s"
Write-Host "Artifacts: artifacts/benchmark_long_real.json, artifacts/benchmark_long_real.md"