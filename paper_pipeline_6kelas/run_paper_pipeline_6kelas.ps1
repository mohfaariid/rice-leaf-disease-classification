$ErrorActionPreference = "Stop"

$PipelineDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PipelineDir
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EntryPoint = Join-Path $PipelineDir "run_paper_pipeline_6kelas.py"

$env:PAPER_EPOCHS = if ($env:PAPER_EPOCHS) { $env:PAPER_EPOCHS } else { "100" }
$env:PAPER_IMAGE_SIZE = if ($env:PAPER_IMAGE_SIZE) { $env:PAPER_IMAGE_SIZE } else { "256" }
$env:PAPER_BATCH_SIZE = if ($env:PAPER_BATCH_SIZE) { $env:PAPER_BATCH_SIZE } else { "32" }
$env:PAPER_LEARNING_RATE = if ($env:PAPER_LEARNING_RATE) { $env:PAPER_LEARNING_RATE } else { "0.001" }

Set-Location -LiteralPath $ProjectRoot
& $PythonPath -u $EntryPoint
