$env:COLMAP_BIN = "$PSScriptRoot\colmap_bin\COLMAP.bat"
$env:SEMANTISPLAT_DEVICE = "cuda"
.\.venv\Scripts\python.exe -m semantisplat.api.app

