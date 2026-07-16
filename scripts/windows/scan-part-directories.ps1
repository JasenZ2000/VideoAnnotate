[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [ValidateRange(0, 32)]
    [int]$MaxDepth = 4,
    [string[]]$MarkerDirectories = @("images", "labels", "annotations"),
    [ValidateRange(1, 32)]
    [int]$MinimumMarkerCount = 1,
    [string]$OutputFile = "",
    [switch]$CopyToClipboard
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path.TrimEnd("\", "/")
$markers = @($MarkerDirectories | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
if (-not $markers.Count) {
    throw "MarkerDirectories cannot be empty."
}
if ($MinimumMarkerCount -gt $markers.Count) {
    throw "MinimumMarkerCount cannot exceed the number of MarkerDirectories."
}

$results = [System.Collections.Generic.List[string]]::new()

function Visit-Directory {
    param([string]$Directory, [int]$Depth)

    $children = @(Get-ChildItem -LiteralPath $Directory -Directory -Force -ErrorAction Stop)
    $childNames = @{}
    foreach ($child in $children) {
        $childNames[$child.Name.ToLowerInvariant()] = $true
    }
    $markerCount = @($markers | Where-Object { $childNames.ContainsKey($_) }).Count
    if ($markerCount -ge $MinimumMarkerCount) {
        $relative = if ($Directory.Equals($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            "."
        } else {
            $Directory.Substring($rootPath.Length).TrimStart("\", "/")
        }
        $results.Add(($relative -replace "\\", "/"))
        return
    }

    if ($Depth -ge $MaxDepth) { return }
    foreach ($child in $children) {
        if (($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
        Visit-Directory -Directory $child.FullName -Depth ($Depth + 1)
    }
}

Visit-Directory -Directory $rootPath -Depth 0
$lines = @($results | Sort-Object -Unique)
if (-not $lines.Count) {
    throw "No work directories found under '$rootPath' within depth $MaxDepth. Markers: $($markers -join ', ')."
}

if ($OutputFile) {
    $target = [System.IO.Path]::GetFullPath($OutputFile)
    $parent = Split-Path -Parent $target
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Set-Content -LiteralPath $target -Value $lines -Encoding UTF8
    Write-Host "Saved $($lines.Count) Part directories to: $target"
}
if ($CopyToClipboard) {
    $lines -join [Environment]::NewLine | Set-Clipboard
    Write-Host "Copied $($lines.Count) Part directories to the clipboard."
}
if (-not $OutputFile -and -not $CopyToClipboard) {
    $lines
}
