[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Name = "LocateAnythingBatchTool",
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
    "--collect-submodules", "paramiko",
    (Join-Path $projectRoot "locany_batch_tool\__main__.py")
)

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE. Install requirements\locany-tool-windows.txt first."
}

Write-Host "Built: $(Join-Path $OutputDir "$Name.exe")"
