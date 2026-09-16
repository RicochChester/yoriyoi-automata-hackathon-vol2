param(
    [int]$SeedStart = 1,
    [int]$SeedEnd = 100
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
Push-Location $PSScriptRoot
try {
    & $pythonPath -X utf8 -m poc.ab_poc.bridge_experiment --seed-start $SeedStart --seed-end $SeedEnd
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed with exit code $LASTEXITCODE. Read the error above." }
} finally {
    Pop-Location
}
