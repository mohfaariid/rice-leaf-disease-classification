Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Select-Object ProcessId, CreationDate, CommandLine |
    Sort-Object CreationDate |
    Format-Table -AutoSize
