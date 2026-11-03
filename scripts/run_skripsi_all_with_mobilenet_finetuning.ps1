param(
    [ValidateSet("both", "original", "balanced")]
    [string]$Scenario = "both"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReportDir = Join-Path $ProjectRoot "reports"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StatusPath = Join-Path $ReportDir "training_skripsi_finetuning_$Timestamp.status.txt"

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$env:RICE_EPOCHS = if ($env:RICE_EPOCHS) { $env:RICE_EPOCHS } else { "100" }
$env:RICE_IMAGE_SIZE = if ($env:RICE_IMAGE_SIZE) { $env:RICE_IMAGE_SIZE } else { "256" }
$env:RICE_BATCH_SIZE = if ($env:RICE_BATCH_SIZE) { $env:RICE_BATCH_SIZE } else { "32" }
$env:RICE_RESUME_TRAINING = if ($env:RICE_RESUME_TRAINING) { $env:RICE_RESUME_TRAINING } else { "0" }
$env:RICE_FEATURE_EPOCHS = if ($env:RICE_FEATURE_EPOCHS) { $env:RICE_FEATURE_EPOCHS } else { "100" }
$env:RICE_FINE_TUNE_EPOCHS = if ($env:RICE_FINE_TUNE_EPOCHS) { $env:RICE_FINE_TUNE_EPOCHS } else { "30" }
$env:RICE_FINE_TUNE_LAYERS = if ($env:RICE_FINE_TUNE_LAYERS) { $env:RICE_FINE_TUNE_LAYERS } else { "30" }
$env:RICE_FINE_TUNE_LR = if ($env:RICE_FINE_TUNE_LR) { $env:RICE_FINE_TUNE_LR } else { "0.00001" }

function Write-Status {
    param([string]$Message)
    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    $Line | Tee-Object -FilePath $StatusPath -Append
}

function Invoke-TrainingScenario {
    param(
        [string]$Name,
        [string]$EntryPoint
    )

    $StdoutPath = Join-Path $ReportDir "training_${Name}_finetuning_$Timestamp.log"
    $StderrPath = Join-Path $ReportDir "training_${Name}_finetuning_$Timestamp.err.log"

    Write-Status "START $Name"
    Write-Status "ENTRYPOINT $EntryPoint"
    Write-Status "STDOUT $StdoutPath"
    Write-Status "STDERR $StderrPath"

    $Process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList @("-u", $EntryPoint) `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru

    Write-Status "PID $Name $($Process.Id)"
    $Process.WaitForExit()
    Write-Status "FINISH $Name exit_code=$($Process.ExitCode)"

    if ($Process.ExitCode -ne 0) {
        throw "Training $Name gagal dengan exit code $($Process.ExitCode). Cek log error: $StderrPath"
    }
}

Write-Status "RUNNER START scenario=$Scenario"
Write-Status "PROJECT_ROOT $ProjectRoot"
Write-Status "PYTHON $PythonPath"
Write-Status "RICE_EPOCHS=$env:RICE_EPOCHS"
Write-Status "RICE_FEATURE_EPOCHS=$env:RICE_FEATURE_EPOCHS"
Write-Status "RICE_FINE_TUNE_EPOCHS=$env:RICE_FINE_TUNE_EPOCHS"
Write-Status "RICE_FINE_TUNE_LAYERS=$env:RICE_FINE_TUNE_LAYERS"
Write-Status "RICE_FINE_TUNE_LR=$env:RICE_FINE_TUNE_LR"
Write-Status "RICE_RESUME_TRAINING=$env:RICE_RESUME_TRAINING"

if ($Scenario -eq "both" -or $Scenario -eq "original") {
    Invoke-TrainingScenario -Name "original" -EntryPoint "main.py"
}

if ($Scenario -eq "both" -or $Scenario -eq "balanced") {
    Invoke-TrainingScenario -Name "balanced_tungro" -EntryPoint "main_balanced_tungro.py"
}

Write-Status "ALL_DONE"
