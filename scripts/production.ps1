param(
    [ValidateSet("menu", "ensure", "start", "stop", "status", "update", "restart", "logs", "open", "backup", "clear-recovery-block")]
    [string]$Action = "start",
    [switch]$NoBrowser,
    [switch]$NonInteractive,
    [switch]$UserRequested,
    [string]$ReleaseToken = ""
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
$controlLockFile = Join-Path $runtimeDir "production-control.lock"
$releaseLockFile = Join-Path $runtimeDir "production-release.lock"
$releaseOwnerFile = Join-Path $runtimeDir "production-release.owner.json"
$productionRoleFile = Join-Path $runtimeDir "production-role.json"
$recoveryRequiredFile = Join-Path $runtimeDir "production-recovery-required.json"
$operatorStoppedFile = Join-Path $runtimeDir "production-operator-stopped.json"
$logDir = Join-Path $projectRoot "logs"
$controlLog = Join-Path $logDir "production-control.log"
$caddyDataDir = Join-Path $projectRoot "caddy_data"
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"

function Write-ProductionControlLog([string]$Message, [string]$Level = "INFO") {
    try {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        Add-Content -LiteralPath $controlLog -Encoding UTF8 -Value (
            "$(Get-Date -Format o) [$Level] action=$Action $Message"
        )
    }
    catch {
        # A logging problem must not mask the real production result.
    }
}

function Invoke-WithProductionControlLock(
    [scriptblock]$Operation,
    [int]$TimeoutMilliseconds = 1500
) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $lockStream = $null
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        while ($null -eq $lockStream) {
            try {
                $lockStream = [IO.File]::Open(
                    $controlLockFile,
                    [IO.FileMode]::OpenOrCreate,
                    [IO.FileAccess]::ReadWrite,
                    [IO.FileShare]::None
                )
            }
            catch [IO.IOException] {
                if ($timer.ElapsedMilliseconds -ge $TimeoutMilliseconds) {
                    throw (
                        "Another production control operation is already in progress. " +
                        "Wait for it to finish, then try again."
                    )
                }
                Start-Sleep -Milliseconds 100
            }
        }
        & $Operation
    }
    finally {
        $timer.Stop()
        if ($null -ne $lockStream) { $lockStream.Dispose() }
    }
}

function Test-ReleaseGateAuthorization([string]$Token) {
    $tokenGuid = [Guid]::Empty
    if (
        [string]::IsNullOrWhiteSpace($Token) -or
        -not [Guid]::TryParse($Token, [ref]$tokenGuid)
    ) {
        return $false
    }
    if (
        -not (Test-Path -LiteralPath $releaseOwnerFile -PathType Leaf) -or
        -not (Test-Path -LiteralPath $releaseLockFile -PathType Leaf)
    ) {
        return $false
    }

    try {
        $owner = Get-Content -LiteralPath $releaseOwnerFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $tokenProperty = $owner.PSObject.Properties["release_token"]
        if ($null -eq $tokenProperty) { return $false }

        $ownerGuid = [Guid]::Empty
        if (-not [Guid]::TryParse([string]$tokenProperty.Value, [ref]$ownerGuid)) {
            return $false
        }
        if ($ownerGuid -ne $tokenGuid) { return $false }
    }
    catch {
        return $false
    }

    # Metadata alone is not authority. The publisher must still own the
    # operating-system file lock for the complete release transaction.
    $probeStream = $null
    try {
        $probeStream = [IO.File]::Open(
            $releaseLockFile,
            [IO.FileMode]::Open,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
        return $false
    }
    catch [IO.IOException] {
        $nativeError = $_.Exception.HResult -band 0xFFFF
        return @(32, 33) -contains $nativeError
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $probeStream) { $probeStream.Dispose() }
    }
}

function Invoke-WithProductionReleaseGate(
    [scriptblock]$Operation,
    [int]$TimeoutMilliseconds = 1500
) {
    # publish-release.ps1 owns this outer lock across stop, backup, code
    # promotion, start, and health verification. Its child controller calls
    # bypass only this gate and still take the production-control lock below.
    if ($ReleaseToken) {
        if (-not (Test-ReleaseGateAuthorization $ReleaseToken)) {
            throw "The production release token is invalid or its release lock is not active."
        }
        & $Operation
        return
    }

    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $lockStream = $null
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        while ($null -eq $lockStream) {
            try {
                $lockStream = [IO.File]::Open(
                    $releaseLockFile,
                    [IO.FileMode]::OpenOrCreate,
                    [IO.FileAccess]::ReadWrite,
                    [IO.FileShare]::None
                )
            }
            catch [IO.IOException] {
                if ($timer.ElapsedMilliseconds -ge $TimeoutMilliseconds) {
                    throw (
                        "A production release is currently updating the live system. " +
                        "Wait for it to finish, then try again."
                    )
                }
                Start-Sleep -Milliseconds 100
            }
        }
        & $Operation
    }
    finally {
        $timer.Stop()
        if ($null -ne $lockStream) { $lockStream.Dispose() }
    }
}

function ConvertTo-NormalizedProductionPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
}

function Assert-ProductionRole {
    if (-not (Test-Path -LiteralPath $productionRoleFile -PathType Leaf)) {
        throw (
            "This checkout is not authorized to run production. " +
            "The production role marker is missing: $productionRoleFile"
        )
    }

    try {
        $roleMarker = Get-Content -LiteralPath $productionRoleFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        throw "The production role marker is unreadable or invalid JSON."
    }

    foreach ($property in @(
        "schema_version", "role", "worktree", "branch", "remote", "created_at"
    )) {
        if (-not ($roleMarker.PSObject.Properties.Name -contains $property)) {
            throw "The production role marker is missing '$property'."
        }
    }

    $schemaVersion = 0
    if (-not [int]::TryParse(
        [string]$roleMarker.schema_version,
        [ref]$schemaVersion
    ) -or $schemaVersion -ne 1) {
        throw "The production role marker must use schema_version 1."
    }
    if ([string]$roleMarker.role -cne "production") {
        throw "The production role marker does not identify a production checkout."
    }
    if ([string]$roleMarker.branch -cne "main") {
        throw "The production role marker must identify branch 'main'."
    }
    if ([string]$roleMarker.remote -cne "origin") {
        throw "The production role marker must identify remote 'origin'."
    }
    if ([string]::IsNullOrWhiteSpace([string]$roleMarker.worktree) -or
        -not [IO.Path]::IsPathRooted([string]$roleMarker.worktree)) {
        throw "The production role marker must identify its absolute worktree path."
    }

    $createdAt = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse(
        [string]$roleMarker.created_at,
        [ref]$createdAt
    )) {
        throw "The production role marker has an invalid created_at timestamp."
    }

    try {
        $markedRoot = ConvertTo-NormalizedProductionPath ([string]$roleMarker.worktree)
        $actualRoot = ConvertTo-NormalizedProductionPath $projectRoot
    }
    catch {
        throw "The production role marker contains an invalid worktree path."
    }
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals($markedRoot, $actualRoot)) {
        throw "The production role marker belongs to a different checkout."
    }

    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $git) {
        throw "Git is required to verify the production checkout branch."
    }
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $branchOutput = @(
            & $git.Source -C $projectRoot symbolic-ref --quiet --short HEAD 2>&1
        )
        $branchExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    $actualBranch = (($branchOutput | ForEach-Object { [string]$_ }) -join "").Trim()
    if ($branchExitCode -ne 0 -or $actualBranch -cne "main") {
        throw "Production controls require the checkout to be on branch 'main'."
    }

    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $statusOutput = @(
            & $git.Source -C $projectRoot status --porcelain=v1 `
                --untracked-files=all 2>&1
        )
        $statusExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($statusExitCode -ne 0 -or
        (($statusOutput | ForEach-Object { [string]$_ }) -join "").Trim()) {
        throw "Production controls require a clean authorized main worktree."
    }
}

function Invoke-WithProductionMutationLocks(
    [scriptblock]$Operation,
    [int]$ReleaseGateTimeoutMilliseconds = 1500,
    [int]$ControlTimeoutMilliseconds = 1500
) {
    # Keep the caller's operation under a distinct name. Both lock helpers
    # have their own Operation parameter, so reusing that name here would make
    # the inner control lock recursively invoke the release-gate scriptblock.
    $operationToRun = $Operation
    Invoke-WithProductionReleaseGate `
        -TimeoutMilliseconds $ReleaseGateTimeoutMilliseconds `
        -Operation {
            Invoke-WithProductionControlLock `
                -TimeoutMilliseconds $ControlTimeoutMilliseconds `
                -Operation {
                    Assert-ProductionRole
                    & $operationToRun
                }
        }
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Invoke-ElevatedProductionStop([int]$ProcessId) {
    Write-Host (
        "The running server was started with Administrator privileges. " +
        "Approve the Windows prompt once so it can be stopped safely."
    ) -ForegroundColor Yellow
    $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
    try {
        $elevated = Start-Process $taskkill -Verb RunAs `
            -ArgumentList @("/PID", "$ProcessId", "/T", "/F") `
            -Wait -PassThru -WindowStyle Hidden
    }
    catch {
        throw "Administrator approval was cancelled; production is still running."
    }
    if ($elevated.ExitCode -ne 0) {
        throw "The elevated production stop failed with exit code $($elevated.ExitCode)."
    }
}

function Read-DotEnv([string]$Path = $envFile) {
    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $values }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $values[$parts[0].Trim()] = $parts[1].Trim().Trim('"').Trim("'")
    }
    return $values
}

function ConvertTo-DotEnvQuotedValue([string]$Value) {
    $escaped = $Value.Replace("\", "\\").Replace('"', '\"')
    $escaped = $escaped.Replace("`r", "\r").Replace("`n", "\n")
    return '"' + $escaped + '"'
}

function Set-DotEnvValue([string]$Key, [string]$Value) {
    $lines = if (Test-Path -LiteralPath $envFile) {
        @(Get-Content -LiteralPath $envFile)
    }
    else { @() }
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

function Read-DatabasePassword([string]$DatabaseUser) {
    $securePassword = Read-Host (
        "Enter the PostgreSQL password for '$DatabaseUser' (input is hidden)"
    ) -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
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

function Test-DatabaseLogin([string]$Password) {
    $previousPassword = [Environment]::GetEnvironmentVariable("DB_PASSWORD", "Process")
    $previousSettings = [Environment]::GetEnvironmentVariable("DJANGO_SETTINGS_MODULE", "Process")
    $previousErrorPreference = $ErrorActionPreference
    try {
        [Environment]::SetEnvironmentVariable("DB_PASSWORD", $Password, "Process")
        [Environment]::SetEnvironmentVariable("DJANGO_SETTINGS_MODULE", "inventory.settings", "Process")
        $ErrorActionPreference = "Continue"
        $probe = (
            "import django; django.setup(); " +
            "from django.db import connection; " +
            "connection.ensure_connection(); connection.close()"
        )
        $probeOutput = @(& $python -c $probe 2>&1)
        $exitCode = $LASTEXITCODE
        return [pscustomobject]@{
            Succeeded = ($exitCode -eq 0)
            Detail = (($probeOutput | ForEach-Object { [string]$_ }) -join "`n")
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
        [Environment]::SetEnvironmentVariable("DB_PASSWORD", $previousPassword, "Process")
        [Environment]::SetEnvironmentVariable("DJANGO_SETTINGS_MODULE", $previousSettings, "Process")
    }
}

function Ensure-DatabaseLogin(
    [hashtable]$config,
    [bool]$AllowCredentialPrompt = $true
) {
    $databaseUser = if ($config.ContainsKey("DB_USER") -and $config["DB_USER"]) {
        [string]$config["DB_USER"]
    }
    else { "postgres" }
    $candidate = if ($config.ContainsKey("DB_PASSWORD")) {
        [string]$config["DB_PASSWORD"]
    }
    else { "" }
    $enteredInteractively = $false

    if (-not $candidate) {
        if (-not $AllowCredentialPrompt) {
            throw (
                "DB_PASSWORD is missing from .env. Open Pharmacy Admin Control and choose " +
                "Start once to repair the database login."
            )
        }
        Write-Host "Database setup needs one password before production can start." -ForegroundColor Yellow
        do {
            $candidate = Read-DatabasePassword $databaseUser
            if (-not $candidate) {
                Write-Host "The database password cannot be blank." -ForegroundColor Yellow
            }
        } while (-not $candidate)
        $enteredInteractively = $true
    }

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "Verifying the database login..." -ForegroundColor Cyan
        $result = Test-DatabaseLogin $candidate
        if ($result.Succeeded) {
            if ($enteredInteractively) {
                Set-DotEnvValue "DB_PASSWORD" (ConvertTo-DotEnvQuotedValue $candidate)
                $config["DB_PASSWORD"] = $candidate
                Write-Host "Database password verified and saved in .env." -ForegroundColor Green
            }
            else {
                Write-Host "Database login verified." -ForegroundColor Green
            }
            return
        }

        if ($result.Detail -notmatch '(?i)password authentication failed|no password supplied|fe_sendauth') {
            $detailLines = @($result.Detail -split "`r?`n" | Where-Object { $_.Trim() })
            $summary = ($detailLines | Select-Object -Last 4) -join " "
            throw "PostgreSQL could not be reached or opened. $summary"
        }
        if ($attempt -eq 3) {
            throw "The PostgreSQL password was not accepted after three attempts."
        }

        if (-not $AllowCredentialPrompt) {
            throw (
                "The saved PostgreSQL password was rejected. Open Pharmacy Admin Control and " +
                "choose Start once to repair the database login."
            )
        }

        Write-Host "That PostgreSQL password was not accepted. Please try again." -ForegroundColor Yellow
        do {
            $candidate = Read-DatabasePassword $databaseUser
        } while (-not $candidate)
        $enteredInteractively = $true
    }
}

function Invoke-DatabaseBackup([string]$Reason) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $backupScript -Reason $Reason
    if ($LASTEXITCODE -ne 0) {
        throw "Database backup failed. No migration or application start was attempted."
    }
}

function Assert-ProductionConfiguration(
    [hashtable]$config,
    [bool]$AllowCredentialPrompt = $true
) {
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
    $adminPasskey = if ($config.ContainsKey("ADMIN_PASSKEY")) {
        [string]$config["ADMIN_PASSKEY"]
    }
    else { "" }
    $unsafeAdminPasskeys = @("pharmacy-admin")
    $exampleConfig = Read-DotEnv (Join-Path $projectRoot ".env.example")
    if ($exampleConfig.ContainsKey("ADMIN_PASSKEY")) {
        $examplePasskey = [string]$exampleConfig["ADMIN_PASSKEY"]
        if (-not [string]::IsNullOrWhiteSpace($examplePasskey)) {
            $unsafeAdminPasskeys += $examplePasskey
        }
    }
    if (
        [string]::IsNullOrWhiteSpace($adminPasskey) -or
        $adminPasskey -cne $adminPasskey.Trim() -or
        $adminPasskey.Length -lt 12 -or
        $unsafeAdminPasskeys -contains $adminPasskey
    ) {
        throw (
            "Set a unique ADMIN_PASSKEY in .env before production startup. " +
            "It must be at least 12 characters, contain no leading or trailing whitespace, " +
            "and cannot use the built-in development default or the .env.example placeholder."
        )
    }
    Ensure-DatabaseLogin $config $AllowCredentialPrompt
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
            -not (Test-IsAdministrator)
        ) {
            if ($NonInteractive) {
                throw (
                    "Production repair needs Administrator approval. Open Pharmacy Admin Control " +
                    "and choose Restart / apply updates."
                )
            }
            Invoke-ElevatedProductionStop $ProcessId
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

function Test-ProductionOperatorStopped {
    return Test-Path -LiteralPath $operatorStoppedFile
}

function Set-ProductionOperatorStopped {
    # Release-engine stops are temporary deployment steps and must never turn
    # into an operator request that suppresses the scheduled recovery ensure.
    if ($ReleaseToken) { return }

    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
    $marker = [pscustomobject][ordered]@{
        schema_version = 1
        stopped_at = (Get-Date).ToUniversalTime().ToString("o")
        process_id = $PID
    }
    $marker | ConvertTo-Json | Set-Content `
        -LiteralPath $operatorStoppedFile -Encoding UTF8
    Write-ProductionControlLog "Recorded an interactive operator stop." "WARN"
}

function Clear-ProductionOperatorStopped([string]$Reason) {
    if (-not (Test-ProductionOperatorStopped)) { return }
    if (-not (Test-Path -LiteralPath $operatorStoppedFile -PathType Leaf)) {
        throw "The operator-stopped marker is invalid and could not be cleared."
    }
    Remove-Item -LiteralPath $operatorStoppedFile -Force
    if (Test-Path -LiteralPath $operatorStoppedFile) {
        throw "The operator-stopped marker could not be removed."
    }
    Write-ProductionControlLog "Cleared operator-stopped marker: $Reason."
}

function Stop-Production {
    $state = Read-ProcessState
    if (-not $state) {
        Write-Host "No tracked production processes are running."
    }
    else {
        foreach ($name in @("caddy_pid", "waitress_pid")) {
            if (Test-TrackedProcess $state $name) {
                $processId = [int]$state.$name
                Stop-TrackedProcessTree $processId
                Write-Host "Stopped process $processId ($name)."
            }
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

function Test-DjangoHealth {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 `
            -Uri "http://127.0.0.1:8000/healthz/"
        return $response.StatusCode -eq 200
    }
    catch { return $false }
}

function Get-ProductionHealth {
    $state = Read-ProcessState
    $waitressRunning = Test-TrackedProcess $state "waitress_pid"
    $caddyRunning = Test-TrackedProcess $state "caddy_pid"
    $hostName = if (
        $state -and
        $state.PSObject.Properties.Name -contains "host" -and
        $state.host
    ) {
        [string]$state.host
    }
    else { "" }
    $djangoHealthy = $waitressRunning -and (Test-DjangoHealth)
    $httpsHealthy = (
        $caddyRunning -and
        $hostName -and
        (Test-HttpsHealth $hostName)
    )

    return [pscustomobject]@{
        State = $state
        HostName = $hostName
        WaitressRunning = [bool]$waitressRunning
        CaddyRunning = [bool]$caddyRunning
        DjangoHealthy = [bool]$djangoHealthy
        HttpsHealthy = [bool]$httpsHealthy
        AnyTracked = [bool]($waitressRunning -or $caddyRunning)
        IsHealthy = [bool](
            $waitressRunning -and
            $caddyRunning -and
            $djangoHealthy -and
            $httpsHealthy
        )
    }
}

function Get-ProductionRecoveryBlock {
    $result = [pscustomobject]@{
        Exists = $false
        IsValid = $false
        ReleaseId = ""
        Error = ""
    }
    if (-not (Test-Path -LiteralPath $recoveryRequiredFile)) {
        return $result
    }

    $result.Exists = $true
    if (-not (Test-Path -LiteralPath $recoveryRequiredFile -PathType Leaf)) {
        $result.Error = "The recovery marker is not a regular file."
        return $result
    }
    try {
        $marker = Get-Content -LiteralPath $recoveryRequiredFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        $result.Error = "The recovery marker is unreadable or invalid JSON."
        return $result
    }
    if ($null -eq $marker) {
        $result.Error = "The recovery marker contains no object data."
        return $result
    }
    if (-not ($marker.PSObject.Properties.Name -contains "release_id")) {
        $result.Error = "The recovery marker is missing release_id."
        return $result
    }

    $releaseId = ([string]$marker.release_id).Trim()
    if (-not $releaseId -or $releaseId.Length -gt 200 -or $releaseId -match "[`r`n]") {
        $result.Error = "The recovery marker has an invalid release_id."
        return $result
    }
    $result.IsValid = $true
    $result.ReleaseId = $releaseId
    return $result
}

function Assert-ProductionRecoveryCleared {
    # The publisher creates the recovery journal before its first stop so a
    # crash cannot leave scheduled startup free to run half-deployed code. Its
    # authenticated child operations may proceed while the same OS release
    # lock remains held; token text or owner metadata alone is insufficient.
    if ($ReleaseToken -and (Test-ReleaseGateAuthorization $ReleaseToken)) {
        return
    }

    $block = Get-ProductionRecoveryBlock
    if (-not $block.Exists) { return }

    if ($block.IsValid) {
        throw (
            "Production startup is blocked because release '$($block.ReleaseId)' " +
            "requires manual recovery. Use Pharmacy Admin Control to inspect the " +
            "system, then choose Clear recovery block."
        )
    }
    throw (
        "Production startup is blocked by an invalid recovery marker. " +
        "$($block.Error) Repair the marker before attempting recovery."
    )
}

function Clear-ProductionRecoveryBlock {
    if ($NonInteractive) {
        throw "Clearing the production recovery block requires an interactive administrator."
    }

    $block = Get-ProductionRecoveryBlock
    if (-not $block.Exists) {
        throw "No production recovery block exists."
    }
    if (-not $block.IsValid) {
        throw "The production recovery block cannot be cleared: $($block.Error)"
    }

    $state = Read-ProcessState
    if ((Test-TrackedProcess $state "waitress_pid") -or
        (Test-TrackedProcess $state "caddy_pid")) {
        throw "Stop all tracked production processes before clearing the recovery block."
    }
    if ((Test-TcpPort "127.0.0.1" 8000) -or
        (Test-TcpPort "127.0.0.1" 443)) {
        throw "Ports 8000 and 443 must both be stopped before clearing the recovery block."
    }

    Write-Host ""
    Write-Host "WARNING: Production recovery is blocked for release:" -ForegroundColor Yellow
    Write-Host "  $($block.ReleaseId)" -ForegroundColor Yellow
    Write-Host "Only clear this after the database and production code have been recovered." -ForegroundColor Yellow
    $confirmation = Read-Host "Type the release ID exactly to clear the recovery block"
    if ($confirmation -cne $block.ReleaseId) {
        throw "The release ID did not match; the recovery block remains active."
    }

    $currentBlock = Get-ProductionRecoveryBlock
    if (-not $currentBlock.Exists -or
        -not $currentBlock.IsValid -or
        $currentBlock.ReleaseId -cne $block.ReleaseId) {
        throw "The recovery marker changed while confirmation was pending; it was not cleared."
    }

    Remove-Item -LiteralPath $recoveryRequiredFile -Force
    if (Test-Path -LiteralPath $recoveryRequiredFile) {
        throw "The recovery marker could not be removed."
    }
    Write-ProductionControlLog (
        "Interactive administrator cleared recovery block for release_id=$($block.ReleaseId) " +
        "after verifying production and ports were stopped."
    ) "WARN"
    Write-Host "Production recovery block cleared." -ForegroundColor Green
}

function Show-Status([object]$Health = $null) {
    if ($null -eq $Health) { $Health = Get-ProductionHealth }

    Write-Host "Waitress: $(if ($Health.WaitressRunning) { 'running' } else { 'stopped' })"
    Write-Host "Caddy:    $(if ($Health.CaddyRunning) { 'running' } else { 'stopped' })"

    if ($Health.WaitressRunning) {
        if ($Health.DjangoHealthy) {
            Write-Host "Django/DB: healthy (HTTP 200)" -ForegroundColor Green
        }
        else { Write-Host "Django/DB: unhealthy" -ForegroundColor Red }
    }
    if ($Health.CaddyRunning) {
        if ($Health.HttpsHealthy) {
            Write-Host "HTTPS:     healthy" -ForegroundColor Green
        }
        else { Write-Host "HTTPS:     unhealthy" -ForegroundColor Red }
    }

    $recoveryBlock = Get-ProductionRecoveryBlock
    if ($recoveryBlock.Exists) {
        if ($recoveryBlock.IsValid) {
            Write-Host (
                "Recovery: BLOCKED (release $($recoveryBlock.ReleaseId))"
            ) -ForegroundColor Red
        }
        else {
            Write-Host "Recovery: BLOCKED (invalid marker)" -ForegroundColor Red
        }
    }
    if (Test-ProductionOperatorStopped) {
        Write-Host "Automatic startup: PAUSED (operator stop)" -ForegroundColor Yellow
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
    Assert-ProductionRecoveryCleared
    $health = Get-ProductionHealth
    if ($health.IsHealthy) {
        Write-Host "Production is already healthy; nothing needs to be started." -ForegroundColor Green
        Show-Status $health
        Write-ProductionControlLog "Production was already healthy."
        if (-not $NoBrowser) { Open-ProductionSite $config }
        return
    }
    if ($health.AnyTracked) {
        throw (
            "Production is partially running or unhealthy. Use the Start Pharmacy " +
            "shortcut, or choose Restart / apply updates in Pharmacy Admin Control."
        )
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
        Write-ProductionControlLog "Production became healthy at $url."
        if (-not $NoBrowser) { Start-Process $url }
    }
    catch {
        if ($caddyProcess -and -not $caddyProcess.HasExited) { Stop-TrackedProcessTree $caddyProcess.Id }
        if ($waitressProcess -and -not $waitressProcess.HasExited) { Stop-TrackedProcessTree $waitressProcess.Id }
        throw
    }
}

function Ensure-Production(
    [hashtable]$config,
    [bool]$AllowCredentialPrompt
) {
    Assert-ProductionRecoveryCleared
    $operatorStopped = Test-ProductionOperatorStopped
    if ($operatorStopped -and -not $UserRequested) {
        Write-Host "Production remains stopped at the operator's request." -ForegroundColor Yellow
        Write-ProductionControlLog "Ensure honored the operator-stopped marker."
        return
    }

    $health = Get-ProductionHealth
    if ($health.IsHealthy) {
        if ($operatorStopped) {
            Clear-ProductionOperatorStopped "user-requested Pharmacy launch found production healthy"
        }
        Write-Host "Production is healthy." -ForegroundColor Green
        Show-Status $health
        Write-ProductionControlLog "Ensure found production healthy."
        if (-not $NoBrowser) { Open-ProductionSite $config }
        return
    }

    # Validate everything, including the database login, before replacing a
    # partially running pair. Hidden launches must fail without prompting.
    Assert-ProductionConfiguration $config $AllowCredentialPrompt
    if ($operatorStopped) {
        Clear-ProductionOperatorStopped "user requested Pharmacy startup"
    }
    if ($health.AnyTracked) {
        Write-Host "Production is incomplete or unhealthy; repairing it now." -ForegroundColor Yellow
        Write-ProductionControlLog "Ensure is replacing an incomplete or unhealthy process pair." "WARN"
        Stop-Production
    }
    else {
        Write-ProductionControlLog "Ensure is starting production from a stopped state."
    }
    Start-Production $config
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
        Write-Host "  [7] Clear recovery block"
        Write-Host "  [0] Exit this console"
        Write-Host ""

        $selection = Read-Host "Choose an option"
        if ($selection -eq "0") { return }

        try {
            switch ($selection) {
                "1" {
                    Invoke-WithProductionMutationLocks {
                        Assert-ProductionRecoveryCleared
                        Assert-ProductionConfiguration $config $true
                        Clear-ProductionOperatorStopped "administrator selected Start"
                        Start-Production $config
                    }
                }
                "2" {
                    Invoke-WithProductionMutationLocks {
                        Stop-Production
                        Set-ProductionOperatorStopped
                    }
                }
                "3" {
                    Invoke-WithProductionMutationLocks {
                        Assert-ProductionRecoveryCleared
                        Assert-ProductionConfiguration $config $true
                        Stop-Production
                        Clear-ProductionOperatorStopped "administrator selected Restart"
                        Start-Production $config
                    }
                }
                "4" { Open-ProductionSite $config }
                "5" { Open-ProductionLogs }
                "6" {
                    Invoke-WithProductionMutationLocks {
                        Invoke-DatabaseBackup "manual"
                    }
                }
                "7" {
                    Invoke-WithProductionMutationLocks {
                        Clear-ProductionRecoveryBlock
                    }
                }
                default { Write-Host "Please choose a number from 0 to 7." -ForegroundColor Yellow }
            }
        }
        catch {
            Write-Host "Production command failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        Wait-ForMenu
    }
}

$configuration = Read-DotEnv
$allowCredentialPrompt = -not $NonInteractive.IsPresent
Set-Location $projectRoot

try {
    switch ($Action) {
        "menu" { Show-ProductionMenu $configuration }
        "ensure" {
            # A sign-in task and a user double-click can overlap. The second
            # ensure waits for a release or prepared startup, then opens/checks
            # the same healthy server instead of launching a duplicate pair.
            Invoke-WithProductionMutationLocks `
                -ReleaseGateTimeoutMilliseconds 1200000 `
                -ControlTimeoutMilliseconds 600000 `
                -Operation {
                    Ensure-Production $configuration $allowCredentialPrompt
                }
        }
        "start" {
            Invoke-WithProductionMutationLocks {
                Assert-ProductionRecoveryCleared
                Assert-ProductionConfiguration $configuration $allowCredentialPrompt
                if (-not $ReleaseToken) {
                    Clear-ProductionOperatorStopped "explicit Start action"
                }
                Start-Production $configuration
            }
        }
        "stop" {
            Invoke-WithProductionMutationLocks {
                Stop-Production
                # A guarded release stop is temporary. Only an explicit
                # operator stop should suppress sign-in/recovery ensures.
                if (-not $ReleaseToken) {
                    Set-ProductionOperatorStopped
                }
            }
        }
        "status" { Show-Status }
        "update" {
            Invoke-WithProductionMutationLocks {
                Assert-ProductionRecoveryCleared
                Assert-ProductionConfiguration $configuration $allowCredentialPrompt
                Stop-Production
                if (-not $ReleaseToken) {
                    Clear-ProductionOperatorStopped "explicit Update action"
                }
                Start-Production $configuration
            }
        }
        "restart" {
            Invoke-WithProductionMutationLocks {
                Assert-ProductionRecoveryCleared
                Assert-ProductionConfiguration $configuration $allowCredentialPrompt
                Stop-Production
                if (-not $ReleaseToken) {
                    Clear-ProductionOperatorStopped "explicit Restart action"
                }
                Start-Production $configuration
            }
        }
        "logs" { Open-ProductionLogs }
        "open" { Open-ProductionSite $configuration }
        "backup" {
            Invoke-WithProductionMutationLocks {
                Assert-ProductionConfiguration $configuration $allowCredentialPrompt
                Invoke-DatabaseBackup "manual"
            }
        }
        "clear-recovery-block" {
            Invoke-WithProductionMutationLocks {
                Clear-ProductionRecoveryBlock
            }
        }
    }
}
catch {
    Write-ProductionControlLog $_.Exception.Message "ERROR"
    Write-Host "Production command failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
