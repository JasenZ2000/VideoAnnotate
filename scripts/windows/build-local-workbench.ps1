[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Name = "VideoAnnotationWorkbench",
    [string]$OutputDir = "",
    [switch]$Windowed
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "dist"
}

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", $Name,
    "--distpath", $OutputDir,
    "--add-data", "$projectRoot\utils\annotator\static;utils\annotator\static",
    "--add-data", "$projectRoot\utils\frame_sampler\static;utils\frame_sampler\static",
    "--collect-submodules", "utils.annotator",
    "--collect-submodules", "utils.frame_sampler",
    (Join-Path $projectRoot "local_workbench\__main__.py")
)
if ($Windowed) {
    $arguments += "--windowed"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE. Install it first with: pip install -r requirements\windows-build.txt"
}

Write-Host "Built: $(Join-Path $OutputDir "$Name.exe")"
