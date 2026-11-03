$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogPath = Join-Path $ProjectRoot "reports\training_balanced_tungro.log"
$ErrPath = Join-Path $ProjectRoot "reports\training_balanced_tungro.err.log"

Write-Host "=== Python processes ==="
tasklist /FI "IMAGENAME eq python.exe"

Write-Host ""
Write-Host "=== Training log tail ==="
if (Test-Path $LogPath) {
    Get-Content -LiteralPath $LogPath -Tail 120
} else {
    Write-Host "Log belum ada: $LogPath"
}

Write-Host ""
Write-Host "=== Error log tail ==="
if (Test-Path $ErrPath) {
    Get-Content -LiteralPath $ErrPath -Tail 50
} else {
    Write-Host "Error log belum ada: $ErrPath"
}
