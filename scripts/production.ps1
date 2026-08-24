param(
    [ValidateSet("menu", "start", "stop", "status", "update", "restart", "logs", "open", "backup")]
    [string]$Action = "start",
    [switch]$NoBrowser,
    [switch]$ElevatedRetry
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$pythonw = Join-Path $projectRoot "env\Scripts\pythonw.exe"
$waitress = Join-Path $projectRoot "env\Scripts\waitress-serve.exe"
$envFile = Join-Path $projectRoot ".env"
$runtimeDir = Join-Path $projectRoot ".runtime"
$pidFile = Join-Path $runtimeDir "production.json"
$logDir = Join-Path $projectRoot "logs"
$caddyDataDir = Join-Path $projectRoot "caddy_data"
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Invoke-ElevatedProductionStop {
    Write-Host (
        "The running server was started with Administrator privileges. " +
        "Approve the Windows prompt once so it can be stopped safely."
    ) -ForegroundColor Yellow
    $arguments = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" " +
        "-Action stop -NoBrowser -ElevatedRetry"
    )
    try {
        $elevated = Start-Process powershell.exe -Verb RunAs `
            -ArgumentList $arguments -Wait -PassThru
    }
    catch {
        throw "Administrator approval was cancelled; production is still running."
    }
    if ($elevated.ExitCode -ne 0) {
        throw "The elevated production stop failed with exit code $($elevated.ExitCode)."
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
        $values[$parts[0].Trim()] = $parts[1].Trim().Trim('"').Trim("'")
    }
    return $values
}

function Get-PharmacyHost([hashtable]$config) {
    if ($config.ContainsKey("PHARMACY_HOST") -and $config["PHARMACY_HOST"]) {
        return $config["PHARMACY_HOST"]
    }
    if ($config.ContainsKey("DJANGO_ALLOWED_HOSTS")) {
        foreach ($candidate in $config["DJANGO_ALLOWED_HOSTS"].Split(",")) {
            $hostName = $candidate.Trim()
            if ($hostName -and $hostName -notin @("localhost", "127.0.0.1", "0.0.0.0")) {
                return $hostName
            }
        }
    }
    throw "Set PHARMACY_HOST or a LAN address in DJANGO_ALLOWED_HOSTS in .env."
}

function Get-CaddyExecutable {
    $localCaddy = Join-Path $projectRoot "caddy.exe"
    if (Test-Path -LiteralPath $localCaddy) { return $localCaddy }
    $command = Get-Command caddy -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "Caddy was not found. Put caddy.exe in the project folder or on PATH."
}

function Test-TcpPort([string]$ComputerName, [int]$Port, [int]$TimeoutMs = 400) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect($ComputerName, $Port, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne($TimeoutMs)) { return $false }
        $client.EndConnect($result)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Test-HttpsHealth([string]$HostName) {
    # Windows PowerShell 5's HTTPS stack cannot reliably negotiate with newer
    # Caddy certificates. Use the project's Python/OpenSSL runtime and bypass
    # certificate verification only for this localhost-controlled readiness test.
    $probe = @"
import ssl, sys, urllib.request
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
)
try:
    for path in ('/healthz/', '/login/'):
        response = opener.open(sys.argv[1] + path, timeout=3)
        if response.status != 200:
            raise SystemExit(1)
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
"@
    & $python -c $probe "https://$HostName" 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Repair-DuplicatePathEnvironment {
    # Some Windows launch contexts expose both `Path` and `PATH`. PowerShell's
    # Start-Process treats environment names case-insensitively and otherwise
    # fails while building the child environment dictionary.
    $processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $processPath) {
        $processPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

function Test-TrackedProcess([object]$data, [string]$propertyName) {
    if (-not $data -or -not ($data.PSObject.Properties.Name -contains $propertyName)) {
        return $false
    }
    $processId = [int]$data.$propertyName
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) { return $false }

    # Never kill an unrelated process that happens to have reused a stale PID.
    $allowedNames = if ($propertyName -eq "caddy_pid") {
        @("caddy")
    }
    else {
        @("python", "pythonw", "waitress-serve")
    }
    if ($process.ProcessName -notin $allowedNames) { return $false }

    $startedProperty = $propertyName -replace "_pid$", "_started_at"
    if ($data.PSObject.Properties.Name -contains $startedProperty) {
        try {
            $expectedStart = [DateTimeOffset]::Parse([string]$data.$startedProperty)
            $actualStart = [DateTimeOffset]$process.StartTime
            if ([Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -gt 2) {
                return $false
            }
        }
        catch { return $false }
    }
    return $true
}

function Read-ProcessState {
    if (-not (Test-Path -LiteralPath $pidFile)) { return $null }
    try { return Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Invoke-Django([string[]]$Arguments) {
    & $python manage.py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Django command failed: manage.py $($Arguments -join ' ')"
    }
}

function Invoke-DatabaseBackup([string]$Reason) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $backupScript -Reason $Reason
    if ($LASTEXITCODE -ne 0) {
        throw "Database backup failed. No migration or application start was attempted."
    }
}

function Assert-ProductionConfiguration([hashtable]$config) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Virtual environment not found. Run setup_env.bat first."
    }
    if (-not (Test-Path -LiteralPath $waitress)) {
        throw "Waitress is not installed. Run setup_env.bat again."
    }
    if (-not (Test-Path -LiteralPath $pythonw)) {
        throw "The windowless Python launcher is missing. Recreate the virtual environment."
    }
    if (-not (Test-Path -LiteralPath $envFile)) {
        throw ".env is missing. Copy .env.example to .env and configure it."
    }
    $secret = if ($config.ContainsKey("DJANGO_SECRET_KEY")) { $config["DJANGO_SECRET_KEY"] } else { "" }
    if (-not $secret -or $secret -in @("replace-with-a-real-secret-key", "django-insecure-fallback-for-dev-only")) {
        throw "Set a real DJANGO_SECRET_KEY in .env before production startup."
    }
}

function Stop-TrackedProcessTree([int]$ProcessId) {
    # Waitress's Windows launcher owns a Python child process. Stop the exact
    # tracked tree so port 8000 cannot be left behind after a restart. Native
    # taskkill writes benign child-race messages to stderr (for example, a
    # child exits between enumeration and termination). With the script-wide
    # ErrorActionPreference=Stop, allowing that stderr through aborts restart
    # before we can verify whether the tracked parent actually stopped.
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return
    }

    $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $treeIds = @($ProcessId)
    do {
        $children = @(
            $allProcesses |
                Where-Object {
                    $_.ParentProcessId -in $treeIds -and
                    $_.ProcessId -notin $treeIds
                } |
                Select-Object -ExpandProperty ProcessId
        )
        if ($children.Count -gt 0) { $treeIds += $children }
    } while ($children.Count -gt 0)

    $previousErrorPreference = $ErrorActionPreference
    $taskKillOutput = @()
    try {
        $ErrorActionPreference = "Continue"
        $taskKillOutput = @(
            & taskkill.exe /PID $ProcessId /T /F 2>&1
        )
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }

    # Retry the exact, pre-captured tree from leaves to root. Stop-Process is a
    # fallback for taskkill races, not a broad process-name kill.
    [array]::Reverse($treeIds)
    foreach ($treeProcessId in $treeIds) {
        Stop-Process -Id $treeProcessId -Force -ErrorAction SilentlyContinue
    }

    $deadline = (Get-Date).AddSeconds(5)
    while (
        (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) -and
        (Get-Date) -lt $deadline
    ) {
        Start-Sleep -Milliseconds 200
    }
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        $detail = ($taskKillOutput | ForEach-Object { "$_" }) -join " "
        if (
            $detail -match "Access is denied" -and
            -not (Test-IsAdministrator) -and
            -not $ElevatedRetry
        ) {
            Invoke-ElevatedProductionStop
            if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
                return
            }
        }
        throw (
            "Could not stop tracked production process $ProcessId." +
            $(if ($detail) { " Windows reported: $detail" } else { "" })
        )
    }
}

function Wait-TcpPortClosed([int]$Port, [int]$TimeoutMs = 5000) {
    $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-TcpPort "127.0.0.1" $Port 150)) { return $true }
        Start-Sleep -Milliseconds 200
    }
    return -not (Test-TcpPort "127.0.0.1" $Port 150)
}

function Stop-Production {
    $state = Read-ProcessState
    if (-not $state) {
        Write-Host "No tracked production processes are running."
        return
    }

    foreach ($name in @("caddy_pid", "waitress_pid")) {
        if (Test-TrackedProcess $state $name) {
            $processId = [int]$state.$name
            Stop-TrackedProcessTree $processId
            Write-Host "Stopped process $processId ($name)."
        }
    }
    if (-not (Wait-TcpPortClosed 8000)) {
        throw "Waitress stopped incompletely: port 8000 is still in use."
    }
    if (-not (Wait-TcpPortClosed 443)) {
        throw "Caddy stopped incompletely: port 443 is still in use."
    }
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force
    }
}

function Show-Status {
    $state = Read-ProcessState
    $waitressRunning = Test-TrackedProcess $state "waitress_pid"
    $caddyRunning = Test-TrackedProcess $state "caddy_pid"

    Write-Host "Waitress: $(if ($waitressRunning) { 'running' } else { 'stopped' })"
    Write-Host "Caddy:    $(if ($caddyRunning) { 'running' } else { 'stopped' })"

    if ($waitressRunning) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "http://127.0.0.1:8000/healthz/"
            Write-Host "Django/DB: healthy (HTTP $($response.StatusCode))" -ForegroundColor Green
        }
        catch {
            Write-Host "Django/DB: unhealthy" -ForegroundColor Red
        }
    }
    if ($caddyRunning -and $state.host) {
        if (Test-HttpsHealth $state.host) {
            Write-Host "HTTPS:     healthy" -ForegroundColor Green
        }
        else {
            Write-Host "HTTPS:     unhealthy" -ForegroundColor Red
        }
    }
}

function Open-ProductionSite([hashtable]$config) {
    $state = Read-ProcessState
    $hostName = if ($state -and $state.PSObject.Properties.Name -contains "host" -and $state.host) {
        [string]$state.host
    }
    else {
        Get-PharmacyHost $config
    }
    Start-Process "https://$hostName"
}

function Open-ProductionLogs {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $recentLogs = Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 6

    if ($recentLogs) {
        Write-Host ""
        Write-Host "Most recent production logs:" -ForegroundColor Cyan
        $recentLogs | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-Host
    }
    else {
        Write-Host "No production logs have been created yet."
    }
    Start-Process -FilePath explorer.exe -ArgumentList @($logDir)
}

function Wait-ForMenu {
    Write-Host ""
    Read-Host "Press Enter to return to the control console" | Out-Null
}

function Start-Production([hashtable]$config) {
    $existing = Read-ProcessState
    if ((Test-TrackedProcess $existing "waitress_pid") -or (Test-TrackedProcess $existing "caddy_pid")) {
        Write-Host "Production is already running; nothing needs to be started." -ForegroundColor Yellow
        Show-Status
        return
    }
    if (Test-TcpPort "127.0.0.1" 8000) {
        throw "Port 8000 is already in use by an untracked process."
    }
    if (Test-TcpPort "127.0.0.1" 443) {
        throw "Port 443 is already in use by an untracked process."
    }

    $caddy = Get-CaddyExecutable
    $pharmacyHost = Get-PharmacyHost $config
    $env:PHARMACY_HOST = $pharmacyHost
    $env:DJANGO_SETTINGS_MODULE = "inventory.settings_production"
    # Keep Caddy's CA, certificates, and state with this deployment. This also
    # avoids locked-down Windows profiles where AppData is not writable.
    $env:XDG_DATA_HOME = $caddyDataDir

    New-Item -ItemType Directory -Force -Path $runtimeDir, $logDir, $caddyDataDir | Out-Null
    Set-Location $projectRoot

    Write-Host "Validating production configuration..." -ForegroundColor Cyan
    Invoke-Django @("check", "--deploy")
    Write-Host "Creating a verified pre-start database backup..." -ForegroundColor Cyan
    Invoke-DatabaseBackup "pre-start"
    Invoke-Django @("migrate", "--noinput")
    Invoke-Django @("collectstatic", "--noinput")

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $waitressOut = Join-Path $logDir "waitress-$stamp.log"
    $waitressErr = Join-Path $logDir "waitress-$stamp.error.log"
    $caddyOut = Join-Path $logDir "caddy-$stamp.log"
    $caddyErr = Join-Path $logDir "caddy-$stamp.error.log"

    $waitressProcess = $null
    $caddyProcess = $null
    try {
        Repair-DuplicatePathEnvironment
        # pythonw keeps the long-lived application server independent of the
        # console that ran production.bat. This is also important for the
        # interactive supplier-ordering workers that Waitress launches later.
        $waitressProcess = Start-Process -FilePath $pythonw `
            -ArgumentList @(
                "-m",
                "waitress",
                "--host=127.0.0.1",
                "--port=8000",
                "--threads=4",
                "--trusted-proxy=127.0.0.1",
                "--trusted-proxy-headers=x-forwarded-proto",
                "inventory.wsgi:application"
            ) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $waitressOut -RedirectStandardError $waitressErr

        $healthy = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 500
            if ($waitressProcess.HasExited) { break }
            try {
                $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://127.0.0.1:8000/healthz/"
                if ($response.StatusCode -eq 200) { $healthy = $true; break }
            }
            catch { }
        }
        if (-not $healthy) {
            throw "Waitress did not become healthy. Check $waitressErr"
        }

        $caddyProcess = Start-Process -FilePath $caddy `
            -ArgumentList @("run", "--config", (Join-Path $projectRoot "Caddyfile")) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $caddyOut -RedirectStandardError $caddyErr

        $caddyReady = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 500
            if ($caddyProcess.HasExited) { break }
            if ((Test-TcpPort "127.0.0.1" 443) -and (Test-HttpsHealth $pharmacyHost)) {
                $caddyReady = $true
                break
            }
        }
        if (-not $caddyReady) {
            throw "Caddy did not produce a healthy HTTPS response. Check $caddyErr"
        }

        [ordered]@{
            waitress_pid = $waitressProcess.Id
            waitress_started_at = $waitressProcess.StartTime.ToString("o")
            caddy_pid = $caddyProcess.Id
            caddy_started_at = $caddyProcess.StartTime.ToString("o")
            host = $pharmacyHost
            started_at = (Get-Date).ToString("o")
        } | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8

        $url = "https://$pharmacyHost"
        Write-Host "Production is healthy at $url" -ForegroundColor Green
        Write-Host "Logs: $logDir"
        Write-Host "Stop with: production.bat stop"
        if (-not $NoBrowser) { Start-Process $url }
    }
    catch {
        if ($caddyProcess -and -not $caddyProcess.HasExited) { Stop-TrackedProcessTree $caddyProcess.Id }
        if ($waitressProcess -and -not $waitressProcess.HasExited) { Stop-TrackedProcessTree $waitressProcess.Id }
        throw
    }
}

function Show-ProductionMenu([hashtable]$config) {
    while ($true) {
        Clear-Host
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host "              PHARMACY PRODUCTION CONTROL" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host ""
        try {
            Show-Status
        }
        catch {
            Write-Host "Status check failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "  [1] Start production"
        Write-Host "  [2] Stop production"
        Write-Host "  [3] Restart / apply updates"
        Write-Host "  [4] Open pharmacy website"
        Write-Host "  [5] Open production logs"
        Write-Host "  [6] Back up the database now"
        Write-Host "  [0] Exit this console"
        Write-Host ""

        $selection = Read-Host "Choose an option"
        if ($selection -eq "0") { return }

        try {
            switch ($selection) {
                "1" { Assert-ProductionConfiguration $config; Start-Production $config }
                "2" { Stop-Production }
                "3" {
                    Assert-ProductionConfiguration $config
                    Stop-Production
                    Start-Production $config
                }
                "4" { Open-ProductionSite $config }
                "5" { Open-ProductionLogs }
                "6" { Invoke-DatabaseBackup "manual" }
                default { Write-Host "Please choose a number from 0 to 5." -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host "Production command failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        Wait-ForMenu
    }
}

$configuration = Read-DotEnv
Set-Location $projectRoot

try {
    switch ($Action) {
        "menu" { Show-ProductionMenu $configuration }
        "start" {
            Assert-ProductionConfiguration $configuration
            Start-Production $configuration
        }
        "stop" { Stop-Production }
        "status" { Show-Status }
        "update" {
            Assert-ProductionConfiguration $configuration
            Stop-Production
            Start-Production $configuration
        }
        "restart" {
            Assert-ProductionConfiguration $configuration
            Stop-Production
            Start-Production $configuration
        }
        "logs" { Open-ProductionLogs }
        "open" { Open-ProductionSite $configuration }
        "backup" {
            Assert-ProductionConfiguration $configuration
            Invoke-DatabaseBackup "manual"
        }
    }
}
catch {
    Write-Host "Production command failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
