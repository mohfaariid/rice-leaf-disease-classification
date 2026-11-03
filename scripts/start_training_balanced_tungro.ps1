$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MainPath = Join-Path $ProjectRoot "main_balanced_tungro.py"
$LogPath = Join-Path $ProjectRoot "reports\training_balanced_tungro.log"
$ErrPath = Join-Path $ProjectRoot "reports\training_balanced_tungro.err.log"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "reports") | Out-Null

$env:RICE_EPOCHS = "20"
$env:RICE_IMAGE_SIZE = "256"
$env:RICE_BATCH_SIZE = "32"

$process = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList "-u `"$MainPath`"" `
    -WorkingDirectory $WorkspaceRoot `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $ErrPath `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Training dataset balanced Tungro dimulai."
Write-Host "PID      : $($process.Id)"
Write-Host "Log      : $LogPath"
Write-Host "Error log: $ErrPath"
