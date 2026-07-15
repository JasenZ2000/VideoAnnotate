[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Name = "AnnotationDatasetSplitter",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
if (-not $OutputDir) { $OutputDir = Join-Path $projectRoot "dist" }

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", $Name,
    "--distpath", $OutputDir,
    "--hidden-import", "PySide6.QtWidgets",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtCore",
    (Join-Path $projectRoot "dataset_split_tool\__main__.py")
)

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE. Install requirements\locany-tool-windows.txt first."
}

Write-Host "Built: $(Join-Path $OutputDir "$Name.exe")"
