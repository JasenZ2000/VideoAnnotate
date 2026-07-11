[CmdletBinding()]
param(
    [string]$HostName = "",
    [int]$Port = 0,
    [string]$TasksDir = "",
    [string]$Database = "",
    [string]$Config = "",
    [string]$Python = "",
    [switch]$Restart,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs
)

$ErrorActionPreference = "Stop"

# Defaults for double-click/edit-and-run usage. These can also be overridden by
# parameters or ANNOTATION_PLATFORM_* environment variables.
$DefaultHostName = "0.0.0.0"
$DefaultPort = 8088

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $projectRoot "scripts\windows\run-platform.ps1"

if (-not $HostName) {
    $HostName = if ($env:ANNOTATION_PLATFORM_HOST) { $env:ANNOTATION_PLATFORM_HOST } else { $DefaultHostName }
}
if ($Port -le 0) {
    $Port = if ($env:ANNOTATION_PLATFORM_PORT) { [int]$env:ANNOTATION_PLATFORM_PORT } else { $DefaultPort }
}
if (-not $Python) {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path $venvPython) { $venvPython } else { "python" }
}

$arguments = @{
    HostName = $HostName
    Port = $Port
    Python = $Python
}
if ($TasksDir) { $arguments.TasksDir = $TasksDir }
if ($Database) { $arguments.Database = $Database }
if ($Config) { $arguments.Config = $Config }
if ($Restart) { $arguments.Restart = $true }
if ($ServerArgs) { $arguments.ServerArgs = $ServerArgs }

Write-Host "Workflow platform will listen on ${HostName}:$Port"
& $launcher @arguments
