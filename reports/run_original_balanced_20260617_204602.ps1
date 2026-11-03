$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project'
$env:RICE_EPOCHS = '20'
$env:RICE_IMAGE_SIZE = '256'
$env:RICE_BATCH_SIZE = '32'

"STARTED original 2026-06-17T20:46:02.0927806+07:00" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204602.status.txt' -Encoding utf8
& 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe' -u main.py 1> 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204602.log' 2> 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_20260617_204602.err.log'
"FINISHED original 2026-06-17T20:46:02.0927806+07:00" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204602.status.txt' -Append -Encoding utf8

"STARTED balanced 2026-06-17T20:46:02.0927806+07:00" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204602.status.txt' -Append -Encoding utf8
& 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\.venv\Scripts\python.exe' -u main_balanced_tungro.py 1> 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204602.log' 2> 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_balanced_tungro_20260617_204602.err.log'
"FINISHED balanced 2026-06-17T20:46:02.0937804+07:00" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204602.status.txt' -Append -Encoding utf8
"ALL_DONE 2026-06-17T20:46:02.0937804+07:00" | Out-File -FilePath 'C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\training_original_balanced_20260617_204602.status.txt' -Append -Encoding utf8
