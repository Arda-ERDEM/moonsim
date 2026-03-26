# Moon Generator GUI Launcher (PowerShell)
# Run this file or type: .\run.ps1

Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
python -m moon_gen
