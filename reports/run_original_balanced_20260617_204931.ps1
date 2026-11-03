$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project'
$env:RICE_EPOCHS = '20'
$env:RICE_IMAGE_SIZE = '256'
$env:RICE_BATCH_SIZE = '32'

"STARTED original $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204931.status.txt' -Encoding utf8
$p1 = Start-Process -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe' -ArgumentList @('-u', 'main.py') -WorkingDirectory 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project' -RedirectStandardOutput 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204931.log' -RedirectStandardError 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204931.err.log' -WindowStyle Hidden -Wait -PassThru
"FINISHED original exit=$($p1.ExitCode) $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204931.status.txt' -Append -Encoding utf8

"STARTED balanced $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204931.status.txt' -Append -Encoding utf8
$p2 = Start-Process -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe' -ArgumentList @('-u', 'main_balanced_tungro.py') -WorkingDirectory 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project' -RedirectStandardOutput 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204931.log' -RedirectStandardError 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204931.err.log' -WindowStyle Hidden -Wait -PassThru
"FINISHED balanced exit=$($p2.ExitCode) $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204931.status.txt' -Append -Encoding utf8
"ALL_DONE $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204931.status.txt' -Append -Encoding utf8
