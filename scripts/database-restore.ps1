param(
    [Parameter(Mandatory = $true)]
    [string]$BackupPath,
    [string]$ConfirmDatabaseName = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"
$productionState = Join-Path $projectRoot ".runtime\production.json"
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"

function Read-DotEnv {
    $values = @{}
    if (-not (Test-Path -LiteralPath $envFile)) { throw ".env is missing." }
    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
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
    throw "$Name was not found. Set POSTGRESQL_BIN in .env."
}

if (Test-Path -LiteralPath $productionState) {
    try {
        $state = Get-Content -LiteralPath $productionState -Raw | ConvertFrom-Json
        foreach ($property in @("waitress_pid", "caddy_pid")) {
            if ($state.PSObject.Properties.Name -contains $property) {
                $processId = [int]$state.$property
                if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
                    throw "Production is running. Run production.bat stop before restoring."
                }
            }
        }
    }
    catch {
        if ($_.Exception.Message -like "Production is running*") { throw }
    }
}
$applicationListener = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
if ($applicationListener) {
    throw "The pharmacy application is still listening on port 8000. Stop it before restoring."
}

$resolvedBackup = (Resolve-Path -LiteralPath $BackupPath -ErrorAction Stop).Path
if ([IO.Path]::GetExtension($resolvedBackup) -ne ".dump") {
    throw "Restore accepts only a verified .dump backup created by database_backup.bat."
}

$config = Read-DotEnv
$databaseName = Get-ConfigValue $config "DB_NAME" "postgres"
$databaseUser = Get-ConfigValue $config "DB_USER" "postgres"
$databaseHost = Get-ConfigValue $config "DB_HOST" "127.0.0.1"
$databasePort = Get-ConfigValue $config "DB_PORT" "5432"
$databasePassword = Get-ConfigValue $config "DB_PASSWORD"

if (-not $ConfirmDatabaseName) {
    $ConfirmDatabaseName = Read-Host "Type the database name '$databaseName' to confirm the restore"
}
if ($ConfirmDatabaseName -cne $databaseName) {
    throw "Restore cancelled: database-name confirmation did not match."
}

$checksumPath = "$resolvedBackup.sha256"
if (Test-Path -LiteralPath $checksumPath) {
    $expectedHash = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
    $actualHash = (Get-FileHash -LiteralPath $resolvedBackup -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expectedHash -ne $actualHash) { throw "Backup checksum verification failed. Restore cancelled." }
}

$pgRestore = Find-PostgresTool "pg_restore" $config
& $pgRestore @("--list", $resolvedBackup) | Out-Null
if ($LASTEXITCODE -ne 0) { throw "The selected backup is not readable by pg_restore." }

# Create a verified safety backup before replacing any database objects.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $backupScript -Reason "pre-restore"
if ($LASTEXITCODE -ne 0) { throw "The pre-restore safety backup failed. Restore cancelled." }

$previousPgPassword = [Environment]::GetEnvironmentVariable("PGPASSWORD", "Process")
try {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $databasePassword, "Process")
    & $pgRestore @(
        "--host=$databaseHost", "--port=$databasePort", "--username=$databaseUser",
        "--dbname=$databaseName", "--clean", "--if-exists", "--no-owner", "--no-acl",
        "--exit-on-error", "--single-transaction", $resolvedBackup
    )
    if ($LASTEXITCODE -ne 0) { throw "Database restore failed with exit code $LASTEXITCODE." }
    Write-Host "Database restored successfully from $resolvedBackup" -ForegroundColor Green
    Write-Host "Run production.bat start to validate and reopen the pharmacy site."
}
finally {
    [Environment]::SetEnvironmentVariable("PGPASSWORD", $previousPgPassword, "Process")
}
