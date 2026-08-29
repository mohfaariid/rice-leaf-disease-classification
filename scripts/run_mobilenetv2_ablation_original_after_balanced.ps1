param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\run_mobilenetv2_finetune_ablation.py"
$ReportDir = Join-Path $ProjectRoot "reports"
$DatasetPath = "C:\Users\farid\Downloads\archive (5)\Rice_Leaf_AUG"
$FeatureModelPath = Join-Path $ProjectRoot "outputs\mobilenetv2_feature_extraction_best_val_loss.keras"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutPath = Join-Path $ReportDir "ablation_mobilenetv2_original_7kelas_$Timestamp.log"
$StderrPath = Join-Path $ReportDir "ablation_mobilenetv2_original_7kelas_$Timestamp.err.log"
$StatusPath = Join-Path $ReportDir "ablation_mobilenetv2_original_7kelas_$Timestamp.status.txt"

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

function Write-Status {
    param([string]$Message)
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message" | Tee-Object -FilePath $StatusPath -Append
}

Write-Status "START queued original ablation"
Write-Status "WAIT_PID=$WaitPid"
Write-Status "STDOUT=$StdoutPath"
Write-Status "STDERR=$StderrPath"

if ($WaitPid -gt 0) {
    $ExistingProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($ExistingProcess) {
        Write-Status "Waiting for process $WaitPid to finish"
        Wait-Process -Id $WaitPid
    } else {
        Write-Status "Process $WaitPid already finished"
    }
}

$env:RICE_ABLATION_LAYERS = if ($env:RICE_ABLATION_LAYERS) { $env:RICE_ABLATION_LAYERS } else { "0,10,20,30,50" }
$env:RICE_ABLATION_EPOCHS = if ($env:RICE_ABLATION_EPOCHS) { $env:RICE_ABLATION_EPOCHS } else { "30" }
$env:RICE_FINE_TUNE_LR = if ($env:RICE_FINE_TUNE_LR) { $env:RICE_FINE_TUNE_LR } else { "0.00001" }
$env:RICE_IMAGE_SIZE = if ($env:RICE_IMAGE_SIZE) { $env:RICE_IMAGE_SIZE } else { "256" }
$env:RICE_BATCH_SIZE = if ($env:RICE_BATCH_SIZE) { $env:RICE_BATCH_SIZE } else { "32" }

$ArgumentList = @(
    "-u",
    "`"$ScriptPath`"",
    "--experiment-name",
    "original_7kelas",
    "--dataset-path",
    "`"$DatasetPath`"",
    "--feature-model-path",
    "`"$FeatureModelPath`""
)

Write-Status "RUN original ablation"
$Process = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru

Write-Status "PID=$($Process.Id)"
$Process.WaitForExit()
Write-Status "FINISH exit_code=$($Process.ExitCode)"

if ($Process.ExitCode -ne 0) {
    throw "Original ablation failed with exit code $($Process.ExitCode). See $StderrPath"
}
