param(
    [ValidateSet(
        "start", "stop", "status", "restart", "logs", "open", "menu",
        "setup", "refresh-data", "check", "publish", "production-status",
        "production-open", "production-logs"
    )]
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
$developmentEnvFile = Join-Path $projectRoot ".env.development"
$developmentDataScript = Join-Path $PSScriptRoot "development-data.ps1"
$releaseScript = Join-Path $PSScriptRoot "publish-release.ps1"
$workflowConfigFile = Join-Path $runtimeDir "development-workflow.json"
$developmentOperationLock = Join-Path $runtimeDir "development-operation.lock"

function Invoke-WithDevelopmentOperationLock(
    [scriptblock]$Operation,
    [int]$TimeoutMilliseconds = 30000
) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    $stream = $null
    while ($null -eq $stream -and [DateTime]::UtcNow -lt $deadline) {
        try {
            $stream = [IO.File]::Open(
                $developmentOperationLock,
                [IO.FileMode]::OpenOrCreate,
                [IO.FileAccess]::ReadWrite,
                [IO.FileShare]::None
            )
        }
        catch [IO.IOException] { Start-Sleep -Milliseconds 100 }
    }
    if ($null -eq $stream) {
        throw "Another development start, stop, or data refresh is already running."
    }
    try { & $Operation }
    finally { $stream.Dispose() }
}

function Assert-DevelopmentConfiguration {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Virtual environment not found. Run setup_env.bat first."
    }
    if (-not (Test-Path -LiteralPath $developmentEnvFile)) {
        throw (
            ".env.development is missing. Choose Setup isolated development " +
            "from this control panel first."
        )
    }
    if (-not (Test-Path -LiteralPath $developmentDataScript)) {
        throw "Development data controller is missing: $developmentDataScript"
    }
    & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
        -File $developmentDataScript -Action status
    if ($LASTEXITCODE -ne 0) {
        throw "Development database isolation validation failed."
    }
}

function Invoke-ControllerScript(
    [string]$Path,
    [string[]]$Arguments,
    [string]$FailureMessage
) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required controller is missing: $Path"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)."
    }
}

function Read-WorkflowConfiguration {
    if (-not (Test-Path -LiteralPath $workflowConfigFile)) { return $null }
    try {
        $config = Get-Content -LiteralPath $workflowConfigFile -Raw | ConvertFrom-Json
        if (-not ($config.PSObject.Properties.Name -contains "production_worktree") -or
            -not $config.production_worktree) {
            throw "production_worktree is missing"
        }
        return $config
    }
    catch {
        throw "Development workflow configuration is invalid: $($_.Exception.Message)"
    }
}

function Get-ProductionController {
    $workflow = Read-WorkflowConfiguration
    if (-not $workflow) {
        throw (
            "The isolated production worktree is not configured. Run the " +
            "development workflow setup before publishing."
        )
    }
    $productionRoot = [string]$workflow.production_worktree
    $controller = Join-Path $productionRoot "scripts\production.ps1"
    if (-not (Test-Path -LiteralPath $controller)) {
        throw "Production controller was not found in the configured worktree: $controller"
    }
    return $controller
}

function Invoke-ProductionControllerAction(
    [string]$ProductionAction,
    [switch]$NoBrowser
) {
    $controller = Get-ProductionController
    $source = Get-Content -LiteralPath $controller -Raw
    if ($ProductionAction -eq "ensure" -and
        $source -notmatch '(?m)\[ValidateSet\([^\]]*"ensure"') {
        throw (
            "The production worktree does not yet contain the guarded ensure action. " +
            "Publish this tested release before installing or opening the staff shortcut."
        )
    }
    $arguments = @(
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", $controller, "-Action", $ProductionAction
    )
    if ($NoBrowser) { $arguments += "-NoBrowser" }
    if ($source -match '(?m)\[switch\]\s*\$NonInteractive\b') {
        $arguments += "-NonInteractive"
    }
    & powershell.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Production action '$ProductionAction' failed (exit code $LASTEXITCODE)."
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
    $requiredState = @(
        "pid", "port", "project_root", "python_path", "process_start_utc"
    )
    if (-not $state) {
        return $false
    }
    foreach ($property in $requiredState) {
        if (-not ($state.PSObject.Properties.Name -contains $property)) {
            return $false
        }
    }
    try {
        if ([int]$state.port -ne 8001 -or
            [IO.Path]::GetFullPath([string]$state.project_root) -ine $projectRoot -or
            [IO.Path]::GetFullPath([string]$state.python_path) -ine $python) {
            return $false
        }
    }
    catch { return $false }
    $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    try {
        $recordedStart = [DateTimeOffset]::Parse(
            [string]$state.process_start_utc
        ).UtcDateTime
        $actualStart = $process.StartTime.ToUniversalTime()
        if ([Math]::Abs(($actualStart - $recordedStart).TotalSeconds) -gt 2) {
            return $false
        }
        if ($process.Path -and
            [IO.Path]::GetFullPath($process.Path) -ine $python) {
            return $false
        }
    }
    catch { return $false }
    return $true
}

function Wait-TcpPortClosed([int]$PortNumber, [int]$TimeoutMs = 5000) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Test-TcpPort "127.0.0.1" $PortNumber 150)) { return $true }
        Start-Sleep -Milliseconds 100
    }
    return -not (Test-TcpPort "127.0.0.1" $PortNumber 150)
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Invoke-ElevatedDevelopmentStop([int]$ProcessId) {
    Write-Host (
        "Development was started with Administrator privileges. " +
        "Approve the Windows prompt once so it can be stopped safely."
    ) -ForegroundColor Yellow
    $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
    try {
        $elevated = Start-Process $taskkill -Verb RunAs `
            -ArgumentList @("/PID", "$ProcessId", "/T", "/F") `
            -Wait -PassThru -WindowStyle Hidden
    }
    catch {
        throw "Administrator approval was cancelled; development is still running."
    }
    if ($elevated.ExitCode -ne 0) {
        throw "The elevated development stop failed with exit code $($elevated.ExitCode)."
    }
}

function Stop-DevelopmentProcessTree([int]$ProcessId) {
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return
    }
    $previousErrorPreference = $ErrorActionPreference
    $taskKillOutput = @()
    try {
        $ErrorActionPreference = "Continue"
        $taskKillOutput = @(& taskkill.exe /PID $ProcessId /T /F 2>&1)
    }
    finally { $ErrorActionPreference = $previousErrorPreference }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    while ((Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) -and
        [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 200
    }
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        $detail = ($taskKillOutput | ForEach-Object { [string]$_ }) -join " "
        if ($detail -match "Access is denied" -and
            -not (Test-IsAdministrator)) {
            Invoke-ElevatedDevelopmentStop $ProcessId
            if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
                return
            }
        }
        throw (
            "Could not stop tracked development process $ProcessId." +
            $(if ($detail) { " Windows reported: $detail" } else { "" })
        )
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
        Write-Host "Auto-reload: disabled (use Restart development after code changes)"
        if (Test-DevelopmentHealth $statePort) {
            Write-Host "Django/DB:   healthy (HTTP 200)" -ForegroundColor Green
        }
        else {
            Write-Host "Django/DB:   starting or unhealthy" -ForegroundColor Red
        }
    }
}

function Start-Development([int]$PortNumber, [bool]$AllowLan) {
    if ($PortNumber -ne 8001) {
        throw "Development is fixed to localhost port 8001."
    }
    if ($AllowLan) {
        throw (
            "Development is localhost-only because it contains test data. " +
            "Use production for pharmacy workstations."
        )
    }
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

    $bindHost = "127.0.0.1"
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputLog = Join-Path $logDir "development-$stamp.log"
    $errorLog = Join-Path $logDir "development-$stamp.error.log"

    Repair-DuplicatePathEnvironment
    $serverProcess = Start-Process -FilePath $python `
        -ArgumentList @(
            "manage.py", "runserver", "${bindHost}:$PortNumber", "--noreload"
        ) `
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
        project_root = $projectRoot
        python_path = $python
        process_start_utc = $serverProcess.StartTime.ToUniversalTime().ToString("o")
        output_log = $outputLog
        error_log = $errorLog
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8

    $url = "http://127.0.0.1:$PortNumber"
    Write-Host "Development is healthy at $url" -ForegroundColor Green
    Write-Host (
        "Auto-reload is disabled for reliable Windows process control; " +
        "use Restart development after code changes."
    )
    Write-Host "The server remains running if this console is closed."
    Write-Host "Stop it from this console or with: development.bat stop"
    if (-not $NoBrowser) { Start-Process $url }
}

function Stop-Development {
    $state = Read-DevelopmentState
    if (-not (Test-DevelopmentProcess $state)) {
        if (Test-TcpPort "127.0.0.1" 8001) {
            throw (
                "Port 8001 is active but its process identity is not safely " +
                "tracked. Close that process before starting or refreshing development."
            )
        }
        if (Test-Path -LiteralPath $pidFile) {
            Remove-Item -LiteralPath $pidFile -Force
        }
        Write-Host "Development is already stopped." -ForegroundColor Yellow
        return
    }

    $processId = [int]$state.pid
    Stop-DevelopmentProcessTree $processId
    if (-not (Wait-TcpPortClosed 8001)) {
        throw "Development did not stop cleanly; port 8001 is still active."
    }
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

function Setup-DevelopmentEnvironment {
    Invoke-ControllerScript $developmentDataScript @("-Action", "setup") `
        "Development environment setup failed"
    Write-Host "Choose Refresh development data to copy a safe production snapshot." -ForegroundColor Cyan
}

function Refresh-DevelopmentData {
    $state = Read-DevelopmentState
    if (Test-DevelopmentProcess $state) {
        $confirmation = Read-Host (
            "Development must stop before its data is refreshed. Stop it now? [y/N]"
        )
        if ($confirmation -notmatch '^(?i)y(es)?$') {
            throw "Development data refresh was cancelled."
        }
        Invoke-WithDevelopmentOperationLock { Stop-Development }
    }
    Invoke-ControllerScript $developmentDataScript @("-Action", "refresh") `
        "Development data refresh failed"
}

function Invoke-ReleaseController([string]$ReleaseAction) {
    if (-not (Test-Path -LiteralPath $releaseScript)) {
        throw "Release controller is missing: $releaseScript"
    }
    Invoke-ControllerScript $releaseScript @("-Action", $ReleaseAction) `
        "Release $ReleaseAction failed"
}

function Show-ProductionStatus {
    try {
        Invoke-ProductionControllerAction "status" -NoBrowser
    }
    catch {
        Write-Host "Production: not provisioned ($($_.Exception.Message))" -ForegroundColor Yellow
    }
}

function Open-ProductionSite {
    Invoke-ProductionControllerAction "ensure"
}

function Open-ProductionLogs {
    Invoke-ProductionControllerAction "logs"
}

function Wait-ForMenu {
    Write-Host ""
    Read-Host "Press Enter to return to the control console" | Out-Null
}

function Show-DevelopmentMenu([int]$PortNumber, [bool]$AllowLan) {
    while ($true) {
        Clear-Host
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host "         PHARMACY DEVELOPMENT & RELEASE CONTROL" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host ""
        Show-DevelopmentStatus
        Show-ProductionStatus
        Write-Host ""
        Write-Host "  [1] Start development (port $PortNumber)"
        Write-Host "  [2] Stop development"
        Write-Host "  [3] Restart development"
        Write-Host "  [4] Open development website"
        Write-Host "  [5] Set up isolated development database"
        Write-Host "  [6] Refresh development data from production snapshot"
        Write-Host "  [7] Run release checks"
        Write-Host "  [8] Publish tested release (production, then GitHub)"
        Write-Host "  [9] Open production"
        Write-Host " [10] Open development logs"
        Write-Host " [11] Open production logs"
        Write-Host "  [0] Exit this console"
        Write-Host ""

        $selection = Read-Host "Choose an option"
        if ($selection -eq "0") { return }

        try {
            switch ($selection) {
                "1" {
                    Invoke-WithDevelopmentOperationLock {
                        Start-Development $PortNumber $AllowLan
                    }
                }
                "2" { Invoke-WithDevelopmentOperationLock { Stop-Development } }
                "3" {
                    Invoke-WithDevelopmentOperationLock {
                        Stop-Development
                        Start-Development $PortNumber $AllowLan
                    }
                }
                "4" { Open-DevelopmentSite $PortNumber }
                "5" { Setup-DevelopmentEnvironment }
                "6" { Refresh-DevelopmentData }
                "7" { Invoke-ReleaseController "check" }
                "8" { Invoke-ReleaseController "publish" }
                "9" { Open-ProductionSite }
                "10" { Open-DevelopmentLogs }
                "11" { Open-ProductionLogs }
                default { Write-Host "Please choose a number from 0 to 11." -ForegroundColor Yellow }
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
        "start" {
            Invoke-WithDevelopmentOperationLock {
                Start-Development $Port $Lan.IsPresent
            }
        }
        "stop" { Invoke-WithDevelopmentOperationLock { Stop-Development } }
        "status" { Show-DevelopmentStatus }
        "restart" {
            Invoke-WithDevelopmentOperationLock {
                Stop-Development
                Start-Development $Port $Lan.IsPresent
            }
        }
        "logs" { Open-DevelopmentLogs }
        "open" { Open-DevelopmentSite $Port }
        "setup" { Setup-DevelopmentEnvironment }
        "refresh-data" { Refresh-DevelopmentData }
        "check" { Invoke-ReleaseController "check" }
        "publish" { Invoke-ReleaseController "publish" }
        "production-status" { Show-ProductionStatus }
        "production-open" { Open-ProductionSite }
        "production-logs" { Open-ProductionLogs }
        "menu" { Show-DevelopmentMenu $Port $Lan.IsPresent }
    }
}
catch {
    Write-Host "Development command failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
