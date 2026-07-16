[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Name = "PartDirectoryScannerTool",
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
    (Join-Path $projectRoot "workflow_platform\part_scanner_gui.py")
)

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE. Install requirements\part-scanner-windows.txt first."
}

Write-Host "Built: $(Join-Path $OutputDir "$Name.exe")"
