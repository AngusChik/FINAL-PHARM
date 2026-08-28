param(
    [string]$ServerUrl = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"
$certificate = Join-Path $projectRoot "Pharmacy-Root-Certificate.crt"
$setupBatch = Join-Path $projectRoot "setup-workstation.bat"
$setupScript = Join-Path $PSScriptRoot "setup-workstation.ps1"

if (-not $ServerUrl) {
    if (-not (Test-Path -LiteralPath $envFile)) { throw ".env is missing." }
    $hostLine = Get-Content -LiteralPath $envFile |
        Where-Object { $_.TrimStart().StartsWith("PHARMACY_HOST=") } |
        Select-Object -First 1
    if (-not $hostLine) { throw ".env is missing PHARMACY_HOST." }
    $hostName = $hostLine.Split("=", 2)[1].Trim().Trim('"').Trim("'")
    $ServerUrl = "https://$hostName"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "workstation-kit"
}
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot $OutputDirectory
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ([IO.Path]::GetPathRoot($OutputDirectory) -eq $OutputDirectory) {
    throw "A drive root cannot be used as the workstation kit directory."
}

foreach ($required in @($certificate, $setupBatch, $setupScript)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Workstation kit source is missing: $required"
    }
}

$scriptsDirectory = Join-Path $OutputDirectory "scripts"
New-Item -ItemType Directory -Force -Path $scriptsDirectory | Out-Null
Copy-Item -LiteralPath $certificate -Destination `
    (Join-Path $OutputDirectory "Pharmacy-Root-Certificate.crt") -Force
Copy-Item -LiteralPath $setupBatch -Destination `
    (Join-Path $OutputDirectory "setup-workstation.bat") -Force
Copy-Item -LiteralPath $setupScript -Destination `
    (Join-Path $scriptsDirectory "setup-workstation.ps1") -Force

Set-Content -LiteralPath (Join-Path $OutputDirectory "server-url.txt") `
    -Value $ServerUrl -Encoding ASCII
$instructions = @(
    "PHARMACY WORKSTATION SETUP",
    "",
    "1. Double-click setup-workstation.bat.",
    "2. Enter this server URL when asked: $ServerUrl",
    "3. Restart Chrome or Edge if it was already open.",
    "4. Open the new Pharmacy shortcut.",
    "",
    "This kit contains only the public HTTPS certificate, server address,",
    "shortcut installer, and these instructions. It contains no database",
    "passwords, application source code, Python, PostgreSQL, Caddy, or Git."
)
Set-Content -LiteralPath (Join-Path $OutputDirectory "README.txt") `
    -Value $instructions -Encoding UTF8

Write-Host "Workstation kit created at $OutputDirectory" -ForegroundColor Green
