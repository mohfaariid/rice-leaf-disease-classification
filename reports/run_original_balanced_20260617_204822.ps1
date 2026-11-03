$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project'
$env:RICE_EPOCHS = '20'
$env:RICE_IMAGE_SIZE = '256'
$env:RICE_BATCH_SIZE = '32'

"STARTED original $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204822.status.txt' -Encoding utf8
cmd /c ""C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe" -u main.py 1> "C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204822.log" 2> "C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204822.err.log""
"FINISHED original exit=$LASTEXITCODE $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204822.status.txt' -Append -Encoding utf8

"STARTED balanced $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204822.status.txt' -Append -Encoding utf8
cmd /c ""C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe" -u main_balanced_tungro.py 1> "C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204822.log" 2> "C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204822.err.log""
"FINISHED balanced exit=$LASTEXITCODE $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204822.status.txt' -Append -Encoding utf8
"ALL_DONE $((Get-Date).ToString('o'))" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204822.status.txt' -Append -Encoding utf8
