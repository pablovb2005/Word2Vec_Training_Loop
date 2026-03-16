param(
    [int]$MaxMinutes = 5,
    [string]$DatasetUrl = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    [string]$CorpusPath = "data/benchmarks/wikitext2_train.txt",
    [string]$MediumCorpusPath = "data/benchmarks/wikitext2_train_medium.txt",
    [int]$MaxChars = 50000
)

$ErrorActionPreference = "Stop"
$overall = [System.Diagnostics.Stopwatch]::StartNew()

if ($MaxMinutes -lt 1) {
    throw "MaxMinutes must be >= 1"
}

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

C:/Python313/python.exe -c "from pathlib import Path; src=Path(r'$CorpusPath'); dst=Path(r'$MediumCorpusPath'); text=src.read_text(encoding='utf-8'); dst.write_text(text[:$MaxChars], encoding='utf-8'); print('prepared', dst, 'bytes', dst.stat().st_size)"

$size = (Get-Item $MediumCorpusPath).Length
Write-Host "Using corpus: $MediumCorpusPath ($size bytes)"

$remaining = [Math]::Max(1, ($MaxMinutes * 60) - [int]$overall.Elapsed.TotalSeconds)
$repoRoot = (Get-Location).Path
$env:PYTHONPATH = (Join-Path $repoRoot "src")
$args = @(
    "-m",
    "word2vec",
    "--corpus",
    $MediumCorpusPath,
    "--benchmark-profile",
    "custom",
    "--embedding-dim",
    "24",
    "--num-negatives",
    "2",
    "--window-size",
    "2",
    "--stream-pairs",
    "--epochs",
    "1",
    "--benchmark-repeats",
    "1",
    "--benchmark-json",
    "artifacts/benchmark_medium_real.json",
    "--benchmark-markdown",
    "artifacts/benchmark_medium_real.md",
    "--queries",
    "word,vectors"
)

Write-Host "Running benchmark with timeout: $remaining seconds"
$proc = Start-Process -FilePath "C:/Python313/python.exe" -ArgumentList $args -WorkingDirectory $repoRoot -NoNewWindow -PassThru

try {
    Wait-Process -Id $proc.Id -Timeout $remaining
} catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Benchmark exceeded timeout budget (${MaxMinutes} minutes)."
}

$proc.Refresh()
if ($null -ne $proc.ExitCode -and $proc.ExitCode -ne 0) {
    throw "Benchmark process failed with exit code $($proc.ExitCode)."
}

$totalSeconds = [math]::Round($overall.Elapsed.TotalSeconds, 2)
Write-Host "Completed in ${totalSeconds}s"
Write-Host "Artifacts: artifacts/benchmark_medium_real.json, artifacts/benchmark_medium_real.md"
Write-Host "Model: artifacts/models/custom_wikitext2_train_medium.npz"
