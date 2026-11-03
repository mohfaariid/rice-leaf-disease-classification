$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$log = Join-Path $root "reports\training_20_epoch_rerun.log"
$err = Join-Path $root "reports\training_20_epoch_rerun.err.log"

Write-Output "--- STDOUT TAIL ---"
if (Test-Path -LiteralPath $log) {
    Get-Content -LiteralPath $log -Tail 80
}
else {
    Write-Output "Log file not found: $log"
}

Write-Output "--- STDERR TAIL ---"
if (Test-Path -LiteralPath $err) {
    Get-Content -LiteralPath $err -Tail 60
}
else {
    Write-Output "Error log file not found: $err"
}

Write-Output "--- PYTHON PROCESSES ---"
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Select-Object ProcessId, CommandLine |
    Format-Table -AutoSize
