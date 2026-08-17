param(
    [string]$Reason = "manual",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "database-backup.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Read-DotEnv {
    $values = @{}
    if (-not (Test-Path -LiteralPath $envFile)) {
        throw ".env is missing. Database backup cannot determine the connection settings."
    }
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

function Get-ConfigValue([hashtable]$Config, [string]$Name, [string]$Default = "") {
    if ($Config.ContainsKey($Name) -and $Config[$Name]) { return [string]$Config[$Name] }
    return $Default
}

function Find-PostgresTool([string]$Name, [hashtable]$Config) {
    $configuredBin = Get-ConfigValue $Config "POSTGRESQL_BIN"
    if ($configuredBin) {
        $candidate = Join-Path $configuredBin "$Name.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $roots = @(Get-ChildItem -LiteralPath "C:\Program Files\PostgreSQL" -Directory -ErrorAction SilentlyContinue |
        Sort-Object { try { [version]$_.Name } catch { [version]"0.0" } } -Descending)
    foreach ($root in $roots) {
        $candidate = Join-Path $root.FullName "bin\$Name.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "$Name was not found. Set POSTGRESQL_BIN in .env to PostgreSQL's bin folder."
}

function Get-Sha256FileHash([string]$Path) {
    # Get-FileHash is provided by Microsoft.PowerShell.Utility, which may not
    # be discoverable when this script inherits a reduced PSModulePath from a
    # launcher. The .NET fallback keeps verified backups available everywhere.
    $fileHashCommand = Get-Command Get-FileHash -ErrorAction SilentlyContinue
    if ($fileHashCommand) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }

    $stream = [IO.File]::OpenRead($Path)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha256.ComputeHash($stream)
        return ([BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

$config = Read-DotEnv
$databaseName = Get-ConfigValue $config "DB_NAME" "postgres"
$databaseUser = Get-ConfigValue $config "DB_USER" "postgres"
$databaseHost = Get-ConfigValue $config "DB_HOST" "127.0.0.1"
$databasePort = Get-ConfigValue $config "DB_PORT" "5432"
$databasePassword = Get-ConfigValue $config "DB_PASSWORD"
$retentionDaysRaw = Get-ConfigValue $config "PHARMACY_BACKUP_RETENTION_DAYS" "30"
$retentionDays = 30
if (-not [int]::TryParse($retentionDaysRaw, [ref]$retentionDays) -or $retentionDays -lt 1) {
    throw "PHARMACY_BACKUP_RETENTION_DAYS must be a positive whole number."
}

if (-not $OutputDirectory) {
    $OutputDirectory = Get-ConfigValue $config "PHARMACY_BACKUP_DIR" "backups\database"
}
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot $OutputDirectory
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$backupDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

$pgDump = Find-PostgresTool "pg_dump" $config
$pgRestore = Find-PostgresTool "pg_restore" $config
$safeReason = ($Reason -replace '[^A-Za-z0-9_-]', '-').Trim('-')
if (-not $safeReason) { $safeReason = "manual" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$finalPath = Join-Path $backupDirectory "pharmacy-$timestamp-$safeReason.dump"
$temporaryPath = "$finalPath.partial"
$checksumPath = "$finalPath.sha256"
$previousPgPassword = [Environment]::GetEnvironmentVariable("PGPASSWORD", "Process")

try {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $databasePassword, "Process")
    & $pgDump @(
        "--host=$databaseHost", "--port=$databasePort", "--username=$databaseUser",
        "--dbname=$databaseName", "--format=custom", "--compress=9",
        "--no-owner", "--no-acl", "--file=$temporaryPath"
    )
    if ($LASTEXITCODE -ne 0) { throw "pg_dump failed with exit code $LASTEXITCODE." }
    if (-not (Test-Path -LiteralPath $temporaryPath) -or (Get-Item -LiteralPath $temporaryPath).Length -eq 0) {
        throw "pg_dump did not create a usable backup file."
    }

    & $pgRestore @("--list", $temporaryPath) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Backup verification failed with exit code $LASTEXITCODE." }

    Move-Item -LiteralPath $temporaryPath -Destination $finalPath
    $hash = Get-Sha256FileHash $finalPath
    Set-Content -LiteralPath $checksumPath -Value "$hash  $([IO.Path]::GetFileName($finalPath))" -Encoding ASCII

    $cutoff = (Get-Date).AddDays(-$retentionDays)
    Get-ChildItem -LiteralPath $backupDirectory -Filter "pharmacy-*.dump" -File |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
            $expiredBackup = $_.FullName
            $expiredChecksum = "$expiredBackup.sha256"
            Remove-Item -LiteralPath $expiredBackup -Force
            if (Test-Path -LiteralPath $expiredChecksum) {
                Remove-Item -LiteralPath $expiredChecksum -Force
            }
        }

    Write-Host "Database backup verified: $finalPath" -ForegroundColor Green
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) OK $finalPath"
    Write-Output $finalPath
}
catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) FAILED $($_.Exception.Message)"
    throw
}
finally {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $previousPgPassword, "Process")
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}
