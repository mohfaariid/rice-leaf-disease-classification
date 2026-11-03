$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "reports\training_20_epoch_rerun.log"
$err = Join-Path $root "reports\training_20_epoch_rerun.err.log"

$env:RICE_EPOCHS = "20"
$env:RICE_IMAGE_SIZE = "256"
$env:RICE_BATCH_SIZE = "32"

$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-u", "main.py") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

Write-Output "PID=$($process.Id)"
Write-Output "LOG=$log"
Write-Output "ERR=$err"
