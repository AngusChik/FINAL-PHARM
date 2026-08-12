param(
    [ValidateSet("start", "stop", "status", "restart", "logs", "open", "menu")]
    [string]$Action = "start",
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$Lan,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".runtime"
$pidFile = Join-Path $runtimeDir "development.json"
$logDir = Join-Path $projectRoot "logs"

function Assert-DevelopmentConfiguration {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Virtual environment not found. Run setup_env.bat first."
    }
}

function Repair-DuplicatePathEnvironment {
    $processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $processPath) {
        $processPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

function Test-TcpPort([string]$ComputerName, [int]$PortNumber, [int]$TimeoutMs = 400) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect($ComputerName, $PortNumber, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne($TimeoutMs)) { return $false }
        $client.EndConnect($result)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Read-DevelopmentState {
    if (-not (Test-Path -LiteralPath $pidFile)) { return $null }
    try { return Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Test-DevelopmentProcess([object]$state) {
    if (-not $state -or -not ($state.PSObject.Properties.Name -contains "pid")) {
        return $false
    }
    return [bool](Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue)
}

function Stop-DevelopmentProcessTree([int]$ProcessId) {
    & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Test-DevelopmentHealth([int]$PortNumber) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "http://127.0.0.1:$PortNumber/login/"
        return $response.StatusCode -eq 200
    }
    catch { return $false }
}

function Show-DevelopmentStatus {
    $state = Read-DevelopmentState
    $running = Test-DevelopmentProcess $state
    Write-Host "Development: $(if ($running) { 'running' } else { 'stopped' })" `
        -ForegroundColor $(if ($running) { "Green" } else { "Yellow" })

    if ($running) {
        $statePort = [int]$state.port
        Write-Host "URL:         http://127.0.0.1:$statePort"
        Write-Host "LAN access:  $(if ($state.lan) { 'enabled' } else { 'disabled' })"
        Write-Host "Auto-reload: enabled"
        if (Test-DevelopmentHealth $statePort) {
            Write-Host "Django/DB:   healthy (HTTP 200)" -ForegroundColor Green
        }
        else {
            Write-Host "Django/DB:   starting or unhealthy" -ForegroundColor Red
        }
    }
}

function Start-Development([int]$PortNumber, [bool]$AllowLan) {
    $existing = Read-DevelopmentState
    if (Test-DevelopmentProcess $existing) {
        Write-Host "Development is already running; nothing needs to be started." -ForegroundColor Yellow
        Show-DevelopmentStatus
        return
    }
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force
    }
    if (Test-TcpPort "127.0.0.1" $PortNumber) {
        throw "Port $PortNumber is already in use. Development defaults to 8001 so production can remain on 8000."
    }

    Assert-DevelopmentConfiguration
    New-Item -ItemType Directory -Force -Path $runtimeDir, $logDir | Out-Null
    Set-Location $projectRoot
    $env:DJANGO_SETTINGS_MODULE = "inventory.settings_development"

    Write-Host "Running Django checks and migrations..." -ForegroundColor Cyan
    & $python manage.py check
    if ($LASTEXITCODE -ne 0) { throw "Django configuration check failed." }
    & $python manage.py migrate --noinput
    if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }

    $bindHost = if ($AllowLan) { "0.0.0.0" } else { "127.0.0.1" }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputLog = Join-Path $logDir "development-$stamp.log"
    $errorLog = Join-Path $logDir "development-$stamp.error.log"

    Repair-DuplicatePathEnvironment
    $serverProcess = Start-Process -FilePath $python `
        -ArgumentList @("manage.py", "runserver", "${bindHost}:$PortNumber") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outputLog -RedirectStandardError $errorLog

    $healthy = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($serverProcess.HasExited) { break }
        if (Test-DevelopmentHealth $PortNumber) {
            $healthy = $true
            break
        }
    }
    if (-not $healthy) {
        if (-not $serverProcess.HasExited) { Stop-DevelopmentProcessTree $serverProcess.Id }
        throw "Development did not become healthy. Check $errorLog"
    }

    [ordered]@{
        pid = $serverProcess.Id
        port = $PortNumber
        lan = $AllowLan
        output_log = $outputLog
        error_log = $errorLog
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8

    $url = "http://127.0.0.1:$PortNumber"
    Write-Host "Development is healthy at $url" -ForegroundColor Green
    Write-Host "Auto-reload is enabled. The server remains running if this console is closed."
    Write-Host "Stop it from this console or with: development.bat stop"
    if (-not $NoBrowser) { Start-Process $url }
}

function Stop-Development {
    $state = Read-DevelopmentState
    if (-not (Test-DevelopmentProcess $state)) {
        if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
        Write-Host "Development is already stopped." -ForegroundColor Yellow
        return
    }

    $processId = [int]$state.pid
    Stop-DevelopmentProcessTree $processId
    if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
    Write-Host "Development stopped." -ForegroundColor Green
}

function Open-DevelopmentSite([int]$FallbackPort) {
    $state = Read-DevelopmentState
    $sitePort = if (Test-DevelopmentProcess $state) { [int]$state.port } else { $FallbackPort }
    Start-Process "http://127.0.0.1:$sitePort"
}

function Open-DevelopmentLogs {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $recentLogs = Get-ChildItem -LiteralPath $logDir -File -Filter "development-*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 6
    if ($recentLogs) {
        Write-Host ""
        Write-Host "Most recent development logs:" -ForegroundColor Cyan
        $recentLogs | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-Host
    }
    else {
        Write-Host "No development logs have been created yet."
    }
    Start-Process -FilePath explorer.exe -ArgumentList @($logDir)
}

function Wait-ForMenu {
    Write-Host ""
    Read-Host "Press Enter to return to the control console" | Out-Null
}

function Show-DevelopmentMenu([int]$PortNumber, [bool]$AllowLan) {
    while ($true) {
        Clear-Host
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host "             PHARMACY DEVELOPMENT CONTROL" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host ""
        Show-DevelopmentStatus
        Write-Host ""
        Write-Host "  [1] Start development (port $PortNumber)"
        Write-Host "  [2] Stop development"
        Write-Host "  [3] Restart development"
        Write-Host "  [4] Open development website"
        Write-Host "  [5] Open development logs"
        Write-Host "  [0] Exit this console"
        Write-Host ""

        $selection = Read-Host "Choose an option"
        if ($selection -eq "0") { return }

        try {
            switch ($selection) {
                "1" { Start-Development $PortNumber $AllowLan }
                "2" { Stop-Development }
                "3" { Stop-Development; Start-Development $PortNumber $AllowLan }
                "4" { Open-DevelopmentSite $PortNumber }
                "5" { Open-DevelopmentLogs }
                default { Write-Host "Please choose a number from 0 to 5." -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host "Development command failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        Wait-ForMenu
    }
}

Set-Location $projectRoot

try {
    switch ($Action) {
        "start" { Start-Development $Port $Lan.IsPresent }
        "stop" { Stop-Development }
        "status" { Show-DevelopmentStatus }
        "restart" { Stop-Development; Start-Development $Port $Lan.IsPresent }
        "logs" { Open-DevelopmentLogs }
        "open" { Open-DevelopmentSite $Port }
        "menu" { Show-DevelopmentMenu $Port $Lan.IsPresent }
    }
}
catch {
    Write-Host "Development command failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
