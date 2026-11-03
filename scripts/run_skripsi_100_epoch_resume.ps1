param(
    [ValidateSet("both", "original", "balanced")]
    [string]$Scenario = "both"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReportDir = Join-Path $ProjectRoot "reports"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StatusPath = Join-Path $ReportDir "training_skripsi_100epoch_$Timestamp.status.txt"

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$env:RICE_EPOCHS = "100"
$env:RICE_IMAGE_SIZE = "256"
$env:RICE_BATCH_SIZE = "32"

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

    $StdoutPath = Join-Path $ReportDir "training_${Name}_100epoch_$Timestamp.log"
    $StderrPath = Join-Path $ReportDir "training_${Name}_100epoch_$Timestamp.err.log"

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

Write-Status "RUNNER START scenario=$Scenario epochs=$env:RICE_EPOCHS image_size=$env:RICE_IMAGE_SIZE batch_size=$env:RICE_BATCH_SIZE"
Write-Status "PROJECT_ROOT $ProjectRoot"
Write-Status "PYTHON $PythonPath"
Write-Status "Resume aktif melalui latest weights di outputs/latest_checkpoints dan history CSV di reports"

if ($Scenario -eq "both" -or $Scenario -eq "original") {
    Invoke-TrainingScenario -Name "original" -EntryPoint "main.py"
}

if ($Scenario -eq "both" -or $Scenario -eq "balanced") {
    Invoke-TrainingScenario -Name "balanced_tungro" -EntryPoint "main_balanced_tungro.py"
}

Write-Status "ALL_DONE"
