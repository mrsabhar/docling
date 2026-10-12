# Run the Intel CPU benchmark + correctness check (Windows PowerShell 5.1+ / pwsh 7+).
#
#   .\experiments\intel-cpu\run_bench.ps1 [-Experiment A-baseline] [-Label NAME] `
#       [-Ocr off,on] [-Docs id1,id2] [-WorkerArgs @('--num-threads','8')]
#
# Thread variables ($env:OMP_NUM_THREADS, $env:DOCLING_NUM_THREADS, ...) are
# inherited from the session and recorded in every result file.
[CmdletBinding()]
param(
    [string]$Experiment = "A-baseline",
    [string]$Label = "",
    [string[]]$Ocr = @("off", "on"),
    [string[]]$Docs = @(),
    [int]$Warmup = 1,
    [int]$Runs = 5,
    [string[]]$WorkerArgs = @()
)
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $Here "..\..")
$Project = Join-Path $Here "uvproject"
Set-Location $Repo

uv sync --frozen --project $Project
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

$benchArgs = @("--experiment", $Experiment, "--ocr") + $Ocr + @("--warmup", $Warmup, "--runs", $Runs)
if ($Label) { $benchArgs += @("--label", $Label) }
if ($Docs.Count -gt 0) { $benchArgs += @("--docs") + $Docs }
if ($WorkerArgs.Count -gt 0) { $benchArgs += @("--") + $WorkerArgs }

uv run --frozen --no-sync --project $Project python (Join-Path $Here "scripts\bench.py") @benchArgs
$benchExit = $LASTEXITCODE

if (-not $Label) {
    $Label = uv run --frozen --no-sync --project $Project python -c "import sys; sys.path.insert(0, r'$Here\scripts'); from common import machine_tag; print(machine_tag())"
}
$Results = Join-Path $Here "results\$Experiment\$Label"
uv run --frozen --no-sync --project $Project python (Join-Path $Here "scripts\correctness.py") groundtruth $Results --repo-check
Write-Host "results: $Results"
exit $benchExit
