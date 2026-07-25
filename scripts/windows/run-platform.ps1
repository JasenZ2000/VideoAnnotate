[CmdletBinding()]
param(
    [string]$HostName = "",
    [int]$Port = 0,
    [string]$TasksDir = "",
    [string]$Database = "",
    [string]$SslCertFile = "",
    [string]$SslKeyFile = "",
    [switch]$AutoHttps,
    [string]$TlsHosts = "",
    [string]$TlsCertDir = "",
    [string]$Python = "python",
    [switch]$Restart,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not $HostName) {
    $HostName = if ($env:ANNOTATION_PLATFORM_HOST) { $env:ANNOTATION_PLATFORM_HOST } else { "0.0.0.0" }
}
if ($Port -le 0) {
    $Port = if ($env:ANNOTATION_PLATFORM_PORT) { [int]$env:ANNOTATION_PLATFORM_PORT } else { 8088 }
}
if (-not $TasksDir) {
    $TasksDir = if ($env:ANNOTATION_PLATFORM_TASKS_DIR) { $env:ANNOTATION_PLATFORM_TASKS_DIR } else { "D:\annotation_tasks" }
}
if (-not $Database) {
    $Database = if ($env:ANNOTATION_PLATFORM_DB) { $env:ANNOTATION_PLATFORM_DB } else { Join-Path $TasksDir "platform.sqlite3" }
}
if (-not $SslCertFile) { $SslCertFile = $env:ANNOTATION_PLATFORM_SSL_CERTFILE }
if (-not $SslKeyFile) { $SslKeyFile = $env:ANNOTATION_PLATFORM_SSL_KEYFILE }
if ($env:ANNOTATION_PLATFORM_AUTO_HTTPS -eq "1") { $AutoHttps = $true }
if (-not $TlsHosts) { $TlsHosts = $env:ANNOTATION_PLATFORM_TLS_HOSTS }
if (-not $TlsCertDir) { $TlsCertDir = $env:ANNOTATION_PLATFORM_TLS_CERT_DIR }
if ([bool]$SslCertFile -ne [bool]$SslKeyFile) {
    throw "SslCertFile and SslKeyFile must be provided together."
}
$scheme = if ($SslCertFile -or $AutoHttps) { "https" } else { "http" }
function Get-PlatformListenerPids {
    $listenerPattern = "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
    return @(
        netstat -ano |
            Select-String $listenerPattern |
            ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
            Sort-Object -Unique
    )
}

$listenerPids = @(Get-PlatformListenerPids)
if ($listenerPids.Count -gt 0) {
    if (-not $Restart) {
        throw "Port $Port is already used by PID(s): $($listenerPids -join ', '). Use -Restart to replace the running annotation platform."
    }

    $healthIdentifiesPlatform = $false
    try {
        $health = Invoke-RestMethod -Uri "${scheme}://127.0.0.1:$Port/api/health" -TimeoutSec 3
        $healthIdentifiesPlatform = $health.service -eq "annotation-collaboration-platform"
    }
    catch {
        Write-Warning "The existing service did not answer the platform health check. Process names will still be verified before restart."
    }

    foreach ($listenerPid in $listenerPids) {
        $process = Get-Process -Id $listenerPid -ErrorAction Stop
        if ($process.ProcessName -notin @("python", "pythonw")) {
            throw "Refusing to stop non-Python process $listenerPid ($($process.ProcessName)) on port $Port."
        }
        if (-not $healthIdentifiesPlatform) {
            Write-Warning "Stopping Python process $listenerPid on the configured platform port $Port."
        }
        Stop-Process -Id $listenerPid -Force
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        Start-Sleep -Milliseconds 250
        $remainingPids = @(Get-PlatformListenerPids)
    } while ($remainingPids.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline)

    if ($remainingPids.Count -gt 0) {
        throw "Port $Port was not released by PID(s): $($remainingPids -join ', ')."
    }
}

New-Item -ItemType Directory -Force -Path $TasksDir | Out-Null

Write-Host "Starting annotation platform at ${scheme}://${HostName}:$Port"
Write-Host "Tasks: $TasksDir"
Write-Host "Database: $Database"

$tlsArgs = @()
if ($SslCertFile) {
    $tlsArgs = @("--ssl-certfile", $SslCertFile, "--ssl-keyfile", $SslKeyFile)
}
elseif ($AutoHttps) {
    $tlsArgs = @("--auto-https")
    if ($TlsHosts) { $tlsArgs += @("--tls-hosts", $TlsHosts) }
    if ($TlsCertDir) { $tlsArgs += @("--tls-cert-dir", $TlsCertDir) }
}

& $Python -m workflow_platform.server `
    --host $HostName `
    --port $Port `
    --tasks-dir $TasksDir `
    --database $Database `
    @tlsArgs `
    @ServerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Annotation platform exited with code $LASTEXITCODE."
}
