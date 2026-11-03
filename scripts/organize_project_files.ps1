$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")

$directories = @(
    "Draft Skripsi",
    "Draft Paper",
    "notebooks\visual_reports",
    "references\notes",
    "references\temp_extraction"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $dir) | Out-Null
}

$moves = @(
    @("draft_bab_1_3_mobilenetv2.md", "Draft Skripsi\draft_bab_1_3_mobilenetv2.md"),
    @("draft_skripsi_bab_1_5.md", "Draft Skripsi\draft_skripsi_bab_1_5.md"),
    @("draft_paper_malcom.md", "Draft Paper\draft_paper_malcom.md"),
    @("_template_extract", "Draft Paper\_template_extract"),
    @("visual_experiment_report.ipynb", "notebooks\visual_reports\visual_experiment_report.ipynb"),
    @("project_completion_roadmap.md", "Panduan Penulisan\project_completion_roadmap.md"),
    @("RUN_20_EPOCHS.md", "Panduan Penulisan\RUN_20_EPOCHS.md"),
    @("INDEX_DOKUMENTASI.md", "Panduan Penulisan\INDEX_DOKUMENTASI.md"),
    @("PROJECT_STRUCTURE.md", "Panduan Penulisan\PROJECT_STRUCTURE.md"),
    @("CHECKLIST_REFERENSI.md", "references\notes\CHECKLIST_REFERENSI.md"),
    @("DAFTAR_PUSTAKA_IEEE.md", "references\notes\DAFTAR_PUSTAKA_IEEE.md"),
    @("PANDUAN_REFERENSI.md", "references\notes\PANDUAN_REFERENSI.md"),
    @("PEMAHAMAN_STRUKTUR_SCREENING.md", "references\notes\PEMAHAMAN_STRUKTUR_SCREENING.md"),
    @("REFERENSI_MAPPING.md", "references\notes\REFERENSI_MAPPING.md"),
    @("REFERENSI_QUICK_REFERENCE.md", "references\notes\REFERENSI_QUICK_REFERENCE.md"),
    @("RINGKASAN_PENYESUAIAN_REFERENSI.txt", "references\notes\RINGKASAN_PENYESUAIAN_REFERENSI.txt"),
    @("temp_pdf_content.txt", "references\temp_extraction\temp_pdf_content.txt"),
    @("temp_pdf_content_02.txt", "references\temp_extraction\temp_pdf_content_02.txt")
)

foreach ($move in $moves) {
    $source = Join-Path $root $move[0]
    $destination = Join-Path $root $move[1]

    if (Test-Path -LiteralPath $source) {
        Move-Item -LiteralPath $source -Destination $destination -Force
        Write-Output "Moved: $($move[0]) -> $($move[1])"
    }
}

$pycache = Join-Path $root "__pycache__"
if (Test-Path -LiteralPath $pycache) {
    Remove-Item -LiteralPath $pycache -Recurse -Force
    Write-Output "Removed: __pycache__"
}

Write-Output "Project files organized."
