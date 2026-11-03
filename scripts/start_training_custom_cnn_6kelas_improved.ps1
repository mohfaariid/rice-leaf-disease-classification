$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MainPath = Join-Path $ProjectRoot "main_custom_cnn_6_classes_improved.py"
$LogPath = Join-Path $ProjectRoot "reports\training_custom_cnn_6kelas_improved.log"
$ErrPath = Join-Path $ProjectRoot "reports\training_custom_cnn_6kelas_improved.err.log"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "reports") | Out-Null

$env:RICE_EPOCHS = "30"
$env:RICE_IMAGE_SIZE = "128"
$env:RICE_BATCH_SIZE = "32"
$env:RICE_LEARNING_RATE = "0.0005"

$process = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList "-u `"$MainPath`"" `
    -WorkingDirectory $WorkspaceRoot `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $ErrPath `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Training CNN improved dimulai."
Write-Host "PID      : $($process.Id)"
Write-Host "Log      : $LogPath"
Write-Host "Error log: $ErrPath"
