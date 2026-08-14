$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$setupScript = $PSCommandPath
$productionScript = Join-Path $PSScriptRoot "production.ps1"
$envFile = Join-Path $projectRoot ".env"
$envExample = Join-Path $projectRoot ".env.example"
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$caddy = Join-Path $projectRoot "caddy.exe"
$rootCertificate = Join-Path $projectRoot "caddy_data\caddy\pki\authorities\local\root.crt"
$sharedCertificate = Join-Path $projectRoot "Pharmacy-Root-Certificate.crt"
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Host "Administrator access is required for firewall and certificate setup." -ForegroundColor Yellow
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$setupScript`""
    $elevated = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
    exit $elevated.ExitCode
}

function Invoke-Native([string]$FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Read-DotEnv {
    $values = @{}
    if (-not (Test-Path -LiteralPath $envFile)) { return $values }
    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $values[$parts[0].Trim()] = $parts[1].Trim()
    }
    return $values
}

function Set-DotEnvValue([string]$Key, [string]$Value) {
    $lines = if (Test-Path -LiteralPath $envFile) { @(Get-Content -LiteralPath $envFile) } else { @() }
    $prefix = "$Key="
    $found = $false
    $updated = foreach ($line in $lines) {
        if ($line.TrimStart().StartsWith($prefix)) {
            $found = $true
            "$Key=$Value"
        }
        else { $line }
    }
    if (-not $found) { $updated += "$Key=$Value" }
    Set-Content -LiteralPath $envFile -Value $updated -Encoding UTF8
}

function New-DjangoSecret {
    $bytes = New-Object byte[] 48
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) }
    finally { $generator.Dispose() }
    return [Convert]::ToBase64String($bytes).Replace("+", "-").Replace("/", "_").TrimEnd("=")
}

function ConvertTo-DotEnvQuotedValue([string]$Value) {
    $escaped = $Value.Replace("\", "\\").Replace('"', '\"')
    $escaped = $escaped.Replace("`r", "\r").Replace("`n", "\n")
    return '"' + $escaped + '"'
}

function Get-RecommendedLanAddress {
    $addresses = @(
        Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
            Where-Object {
                $_.IPAddress -ne "127.0.0.1" -and
                -not $_.IPAddress.StartsWith("169.254.") -and
                $_.InterfaceAlias -notmatch "Loopback|vEthernet|Virtual|VPN"
            } |
            Sort-Object InterfaceMetric
    )
    if ($addresses.Count -eq 0) { return "" }
    return $addresses[0].IPAddress
}

function Test-IPv4Address([string]$Address) {
    $parsed = $null
    return [Net.IPAddress]::TryParse($Address, [ref]$parsed) -and
        $parsed.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork
}

function Install-Caddy {
    if (Test-Path -LiteralPath $caddy) {
        Write-Host "Caddy already exists in the project." -ForegroundColor DarkGreen
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Caddy is missing and winget is unavailable. Install App Installer, then rerun setup."
    }

    Write-Host "Installing the official Caddy package..."
    Invoke-Native $winget.Source @(
        "install", "--id", "CaddyServer.Caddy", "--exact",
        "--accept-package-agreements", "--accept-source-agreements"
    )

    $installedCaddy = Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Recurse -Filter "caddy.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*CaddyServer.Caddy*" } |
        Select-Object -First 1
    if (-not $installedCaddy) {
        $command = Get-Command caddy -ErrorAction SilentlyContinue
        if ($command) { $installedCaddy = Get-Item -LiteralPath $command.Source }
    }
    if (-not $installedCaddy) {
        throw "Caddy installed, but caddy.exe could not be located. Restart Windows and rerun setup."
    }
    Copy-Item -LiteralPath $installedCaddy.FullName -Destination $caddy -Force
}

function Install-DatabaseBackupTask {
    $taskName = "Pharmacy Database Backup"
    $taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$backupScript`" -Reason scheduled"
    Invoke-Native "schtasks.exe" @(
        "/Create", "/TN", $taskName, "/TR", $taskCommand,
        "/SC", "DAILY", "/ST", "02:00", "/RU", "SYSTEM", "/RL", "HIGHEST", "/F"
    )
}

try {
    Set-Location $projectRoot
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host " Pharmacy Main Computer - One-Time Setup" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""

    if (Test-Path -LiteralPath (Join-Path $projectRoot ".runtime\production.json")) {
        Write-Host "Stopping the existing pharmacy server before setup..." -ForegroundColor Yellow
        Invoke-Native "powershell.exe" @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $productionScript,
            "-Action", "stop"
        )
    }

    Write-Host "[1/10] Creating the Python environment..." -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $python)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw "Python is not installed or not available on PATH." }
        Invoke-Native $pythonCommand.Source @("-m", "venv", "env")
    }
    Invoke-Native $python @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Native $python @("-m", "pip", "install", "-r", "requirements.txt")

    Write-Host "[2/10] Installing the supplier-ordering browser..." -ForegroundColor Cyan
    Invoke-Native $python @("-m", "playwright", "install", "chromium")

    Write-Host "[3/10] Creating secure application configuration..." -ForegroundColor Cyan
    $createdEnv = $false
    if (-not (Test-Path -LiteralPath $envFile)) {
        if (-not (Test-Path -LiteralPath $envExample)) { throw ".env.example is missing." }
        Copy-Item -LiteralPath $envExample -Destination $envFile
        $createdEnv = $true
    }
    $configuration = Read-DotEnv
    $secret = if ($configuration.ContainsKey("DJANGO_SECRET_KEY")) { $configuration["DJANGO_SECRET_KEY"] } else { "" }
    if (-not $secret -or $secret -in @("replace-with-a-real-secret-key", "django-insecure-fallback-for-dev-only")) {
        Set-DotEnvValue "DJANGO_SECRET_KEY" (New-DjangoSecret)
    }
    Set-DotEnvValue "DJANGO_DEBUG" "false"

    if ($createdEnv) {
        Write-Host "Enter the PostgreSQL password used by the pharmacy database."
        Write-Host "Press Enter only if PostgreSQL accepts local connections without a password."
        $securePassword = Read-Host "Database password" -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
        try { $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        Set-DotEnvValue "DB_PASSWORD" (ConvertTo-DotEnvQuotedValue $plainPassword)
    }

    Write-Host "[4/10] Configuring the pharmacy LAN address..." -ForegroundColor Cyan
    $recommendedAddress = Get-RecommendedLanAddress
    $prompt = if ($recommendedAddress) { "Server LAN IP [$recommendedAddress]" } else { "Server LAN IP" }
    $serverAddress = (Read-Host $prompt).Trim()
    if (-not $serverAddress) { $serverAddress = $recommendedAddress }
    if (-not (Test-IPv4Address $serverAddress)) { throw "'$serverAddress' is not a valid IPv4 address." }
    Invoke-Native $python @("configure_ip.py", $serverAddress)

    Write-Host "[5/10] Installing Caddy HTTPS..." -ForegroundColor Cyan
    Install-Caddy

    Write-Host "[6/10] Configuring the Windows network and firewall..." -ForegroundColor Cyan
    $publicProfiles = @(
        Get-NetConnectionProfile -ErrorAction SilentlyContinue |
            Where-Object {
                $_.NetworkCategory -eq "Public" -and
                $_.IPv4Connectivity -ne "Disconnected"
            }
    )
    foreach ($profile in $publicProfiles) {
        Write-Host "Changing network '$($profile.Name)' from Public to Private for pharmacy LAN access."
        Set-NetConnectionProfile -InterfaceIndex $profile.InterfaceIndex -NetworkCategory Private
    }
    Get-NetFirewallRule -DisplayName "Pharmacy HTTPS" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName "Pharmacy HTTPS" -Direction Inbound -Protocol TCP `
        -LocalPort 80,443 -Action Allow -Profile Private,Domain -RemoteAddress LocalSubnet | Out-Null
    Get-NetFirewallRule -DisplayName "Pharmacy App" -ErrorAction SilentlyContinue | Disable-NetFirewallRule

    Write-Host "[7/10] Backing up and preparing the database..." -ForegroundColor Cyan
    $env:DJANGO_SETTINGS_MODULE = "inventory.settings_production"
    Invoke-Native $python @("manage.py", "check", "--deploy")
    Invoke-Native "powershell.exe" @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $backupScript,
        "-Reason", "setup-pre-migration"
    )
    Invoke-Native $python @("manage.py", "migrate", "--noinput")
    Invoke-Native $python @("manage.py", "collectstatic", "--noinput")

    Write-Host "[8/10] Scheduling verified daily database backups..." -ForegroundColor Cyan
    Install-DatabaseBackupTask

    Write-Host "[9/10] Initializing Caddy HTTPS certificates..." -ForegroundColor Cyan
    $env:PHARMACY_HOST = $serverAddress
    $env:XDG_DATA_HOME = Join-Path $projectRoot "caddy_data"
    New-Item -ItemType Directory -Force -Path $env:XDG_DATA_HOME | Out-Null
    Invoke-Native $caddy @("validate", "--config", (Join-Path $projectRoot "Caddyfile"))

    Write-Host "[10/10] Trusting the server certificate..." -ForegroundColor Cyan
    for ($attempt = 0; $attempt -lt 20 -and -not (Test-Path -LiteralPath $rootCertificate); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path -LiteralPath $rootCertificate)) {
        throw "Caddy started but its root certificate was not created."
    }
    Invoke-Native "certutil.exe" @("-user", "-addstore", "Root", $rootCertificate)
    Copy-Item -LiteralPath $rootCertificate -Destination $sharedCertificate -Force

    Write-Host ""
    Write-Host "SETUP COMPLETE" -ForegroundColor Green
    Write-Host "Server: https://$serverAddress" -ForegroundColor Green
    Write-Host "Other computers: install Pharmacy-Root-Certificate.crt, then open the server URL."
    Write-Host "Production will start after this administrator setup window closes."
    Write-Host "Start later: production.bat"
    Write-Host "Stop:        production.bat stop"
    Write-Host "Backup now:  production.bat backup"
    exit 0
}
catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Correct the reported problem, then run setup-main-computer.bat again."
    exit 1
}
