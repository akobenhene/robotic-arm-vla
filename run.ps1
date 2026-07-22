# Run the LeRobot Aloha demo without activating the venv (avoids PS execution-policy issues).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Missing .venv. Create with: py -3.11 -m venv .venv && .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

if ($args.Count -eq 0) {
    & $Python main.py --steps 400 --seed 0
} else {
    & $Python @args
}
