param(
    [ValidateSet("status", "configure", "initialize", "setup", "refresh")]
    [string]$Action = "status",
    [string]$ConfirmDatabaseName = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectParent = Split-Path $projectRoot -Parent
$workflowConfigFile = Join-Path $projectRoot ".runtime\development-workflow.json"
$productionRuntimeRoot = $projectRoot
if (Test-Path -LiteralPath $workflowConfigFile -PathType Leaf) {
    $workflow = Get-Content -LiteralPath $workflowConfigFile -Raw | ConvertFrom-Json
    if (-not ($workflow.PSObject.Properties.Name -contains "production_worktree") -or
        -not [string]$workflow.production_worktree) {
        throw "Development workflow configuration is missing production_worktree."
    }
    $candidateProductionRoot = [IO.Path]::GetFullPath(
        [string]$workflow.production_worktree
    ).TrimEnd('\')
    if ((Split-Path $candidateProductionRoot -Parent) -ine $projectParent -or
        $candidateProductionRoot -ieq $projectRoot) {
        throw "Configured production must be a sibling of development."
    }
    $productionRuntimeRoot = $candidateProductionRoot
}
$productionEnvFile = Join-Path $productionRuntimeRoot ".env"
$developmentEnvFile = Join-Path $projectRoot ".env.development"
$runtimeDirectory = Join-Path $projectRoot ".runtime\development-refresh"
$developmentStateFile = Join-Path $projectRoot ".runtime\development.json"
$refreshIncompleteMarker = Join-Path $projectRoot ".runtime\development-refresh-incomplete.json"
$developmentOperationLock = Join-Path $projectRoot ".runtime\development-operation.lock"
$backupScript = Join-Path $productionRuntimeRoot "scripts\database-backup.ps1"
$productionController = Join-Path $productionRuntimeRoot "scripts\production.ps1"
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "development-data.log"
$managedDatabaseComment = "pharmacy-development-managed:v1"
$requiredDevelopmentDatabase = "pharmacy_development"
$requiredDevelopmentRole = "pharmacy_development"
$requiredTestDatabase = "test_pharmacy_development"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-DevelopmentDataLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

function Read-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required environment file is missing: $Path"
    }

    $values = @{}
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

function Get-ConfigValue(
    [hashtable]$Config,
    [string]$Name,
    [string]$Default = ""
) {
    if ($Config.ContainsKey($Name) -and $Config[$Name]) {
        return [string]$Config[$Name]
    }
    return $Default
}

function ConvertTo-DotEnvValue([string]$Value) {
    if ($Value -notmatch '[\s#"'']') { return $Value }
    $escaped = $Value.Replace("\", "\\").Replace('"', '\"')
    return '"' + $escaped + '"'
}

function Ensure-SecureDevelopmentEnvironment([hashtable]$ProductionConfig) {
    $productionDatabase = Get-ConfigValue $ProductionConfig "DB_NAME" "postgres"
    if ($requiredDevelopmentDatabase -ieq $productionDatabase) {
        throw "Production already uses '$requiredDevelopmentDatabase'; development setup is unsafe."
    }

    $existing = @{}
    if (Test-Path -LiteralPath $developmentEnvFile -PathType Leaf) {
        $existing = Read-DotEnv $developmentEnvFile
        $existingDatabase = Get-ConfigValue $existing "DB_NAME"
        if ($existingDatabase -and $existingDatabase -ine $requiredDevelopmentDatabase) {
            throw (
                "Existing .env.development names '$existingDatabase'. " +
                "Only '$requiredDevelopmentDatabase' can be managed automatically."
            )
        }
    }

    $databasePassword = ""
    if ((Get-ConfigValue $existing "DB_USER") -ceq $requiredDevelopmentRole) {
        $databasePassword = Get-ConfigValue $existing "DB_PASSWORD"
    }
    if (-not $databasePassword -or $databasePassword -eq "generated-by-development-setup") {
        $databasePassword = (
            [Guid]::NewGuid().ToString("N") + [Guid]::NewGuid().ToString("N")
        )
    }

    Ensure-DevelopmentDatabaseRole $ProductionConfig $databasePassword

    $securedConfig = @{}
    foreach ($key in $ProductionConfig.Keys) {
        $securedConfig[$key] = $ProductionConfig[$key]
    }
    $securedConfig["DB_NAME"] = $requiredDevelopmentDatabase
    $securedConfig["DB_USER"] = $requiredDevelopmentRole
    $securedConfig["DB_PASSWORD"] = $databasePassword
    $securedConfig["DEVELOPMENT_TEST_DB_NAME"] = $requiredTestDatabase
    $legacyMetadata = Get-DatabaseMetadata `
        $securedConfig $requiredDevelopmentDatabase
    if ($legacyMetadata.Exists -and (
        $legacyMetadata.Owner -cne $requiredDevelopmentRole -or
        $legacyMetadata.Comment -cne $managedDatabaseComment
    )) {
        if ($ConfirmDatabaseName -cne $requiredDevelopmentDatabase) {
            throw (
                "The existing development database predates the ownership marker. " +
                "Rerun with -ConfirmDatabaseName $requiredDevelopmentDatabase to " +
                "adopt it, then refresh its data."
            )
        }
        Assert-DevelopmentStopped
        $databaseLiteral = ConvertTo-SqlLiteral $requiredDevelopmentDatabase
        $adoptionSql = @"
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $databaseLiteral AND pid <> pg_backend_pid();
ALTER DATABASE "$requiredDevelopmentDatabase" OWNER TO "$requiredDevelopmentRole";
COMMENT ON DATABASE "$requiredDevelopmentDatabase" IS 'pharmacy-development-managed:v1';
"@
        Invoke-PostgresQuery $ProductionConfig (
            Get-ConfigValue $ProductionConfig "DB_NAME" "postgres"
        ) $adoptionSql "Could not adopt the legacy development database safely." |
            Out-Null
        New-Item -ItemType Directory -Force `
            -Path (Split-Path $refreshIncompleteMarker -Parent) | Out-Null
        [ordered]@{
            schema_version = 1
            database = $requiredDevelopmentDatabase
            started_at = (Get-Date).ToString("o")
            reason = "Dedicated database role was installed; a fresh sanitized snapshot is required."
        } | ConvertTo-Json | Set-Content `
            -LiteralPath $refreshIncompleteMarker -Encoding UTF8
    }

    $developmentSecret = Get-ConfigValue $existing "DJANGO_SECRET_KEY"
    if (-not $developmentSecret -or $developmentSecret -eq "development-only-change-me") {
        $developmentSecret = "development-only-$([Guid]::NewGuid().ToString('N'))"
    }
    $lines = @(
        "# Generated by scripts/development-data.ps1. This file is gitignored.",
        "PHARMACY_ENVIRONMENT=development",
        "DB_NAME=$requiredDevelopmentDatabase",
        "PRODUCTION_DB_NAME=$(ConvertTo-DotEnvValue $productionDatabase)",
        "DB_USER=$requiredDevelopmentRole",
        "DB_PASSWORD=$(ConvertTo-DotEnvValue $databasePassword)",
        "DB_HOST=$(ConvertTo-DotEnvValue (Get-ConfigValue $ProductionConfig 'DB_HOST' '127.0.0.1'))",
        "DB_PORT=$(ConvertTo-DotEnvValue (Get-ConfigValue $ProductionConfig 'DB_PORT' '5432'))",
        "POSTGRESQL_BIN=$(ConvertTo-DotEnvValue (Get-ConfigValue $ProductionConfig 'POSTGRESQL_BIN'))",
        "DEVELOPMENT_TEST_DB_NAME=$requiredTestDatabase",
        "DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1",
        "DJANGO_SECRET_KEY=$developmentSecret"
    )

    $temporaryEnvFile = "$developmentEnvFile.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        Set-Content -LiteralPath $temporaryEnvFile -Value $lines -Encoding UTF8
        Set-Acl -LiteralPath $temporaryEnvFile `
            -AclObject (Get-Acl -LiteralPath $productionEnvFile)
        Move-Item -LiteralPath $temporaryEnvFile `
            -Destination $developmentEnvFile -Force
    }
    catch {
        $aclFailure = $_.Exception.Message
        if (Test-Path -LiteralPath $temporaryEnvFile) {
            Remove-Item -LiteralPath $temporaryEnvFile -Force
        }
        throw (
            "Could not write .env.development with the production .env access " +
            "controls: $aclFailure"
        )
    }
    Write-Host (
        "Secured .env.development for database '$requiredDevelopmentDatabase' " +
        "and the isolated '$requiredDevelopmentRole' PostgreSQL role."
    ) -ForegroundColor Green
}

function Find-PostgresTool([string]$Name, [hashtable]$Config) {
    $configuredBin = Get-ConfigValue $Config "POSTGRESQL_BIN"
    if ($configuredBin) {
        $candidate = Join-Path $configuredBin "$Name.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $postgresRoots = @(
        Get-ChildItem -LiteralPath "C:\Program Files\PostgreSQL" `
            -Directory -ErrorAction SilentlyContinue |
            Sort-Object {
                try { [version]$_.Name }
                catch { [version]"0.0" }
            } -Descending
    )
    foreach ($root in $postgresRoots) {
        $candidate = Join-Path $root.FullName "bin\$Name.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "$Name was not found. Set POSTGRESQL_BIN in .env.development."
}

function Assert-SafeDatabaseConfiguration(
    [hashtable]$ProductionConfig,
    [hashtable]$DevelopmentConfig
) {
    $environmentName = Get-ConfigValue $DevelopmentConfig "PHARMACY_ENVIRONMENT"
    if ($environmentName -cne "development") {
        throw ".env.development must set PHARMACY_ENVIRONMENT=development."
    }

    $productionName = Get-ConfigValue $ProductionConfig "DB_NAME" "postgres"
    $developmentName = Get-ConfigValue $DevelopmentConfig "DB_NAME"
    if (-not $developmentName) {
        throw ".env.development must set a non-empty DB_NAME."
    }
    if ($productionName -notmatch '^[A-Za-z][A-Za-z0-9_]{0,62}$') {
        throw "Production DB_NAME is not a safe PostgreSQL identifier."
    }
    if ($developmentName -cne $requiredDevelopmentDatabase) {
        throw "Development DB_NAME must be exactly '$requiredDevelopmentDatabase'."
    }
    if ($developmentName -ieq $productionName) {
        throw (
            "Development database '$developmentName' matches the production database. " +
            "Choose a separate database such as pharmacy_development."
        )
    }
    if ($developmentName -iin @("postgres", "template0", "template1")) {
        throw "Development DB_NAME '$developmentName' is a protected PostgreSQL database."
    }

    $developmentUser = Get-ConfigValue $DevelopmentConfig "DB_USER"
    $productionUser = Get-ConfigValue $ProductionConfig "DB_USER" "postgres"
    if ($developmentUser -cne $requiredDevelopmentRole) {
        throw "Development DB_USER must be exactly '$requiredDevelopmentRole'."
    }
    if ($developmentUser -ieq $productionUser) {
        throw "Development and production must use different PostgreSQL roles."
    }
    $testDatabase = Get-ConfigValue $DevelopmentConfig "DEVELOPMENT_TEST_DB_NAME"
    if ($testDatabase -cne $requiredTestDatabase) {
        throw "DEVELOPMENT_TEST_DB_NAME must be exactly '$requiredTestDatabase'."
    }

    $productionHost = Get-ConfigValue $ProductionConfig "DB_HOST" "127.0.0.1"
    $developmentHost = Get-ConfigValue $DevelopmentConfig "DB_HOST" "127.0.0.1"
    $productionPort = Get-ConfigValue $ProductionConfig "DB_PORT" "5432"
    $developmentPort = Get-ConfigValue $DevelopmentConfig "DB_PORT" "5432"

    return [pscustomobject]@{
        ProductionName = $productionName
        DevelopmentName = $developmentName
        DevelopmentUser = $developmentUser
        TestName = $testDatabase
        ProductionHost = $productionHost
        DevelopmentHost = $developmentHost
        ProductionPort = $productionPort
        DevelopmentPort = $developmentPort
        SameServer = (
            $productionHost -ieq $developmentHost -and
            $productionPort -eq $developmentPort
        )
    }
}

function Assert-DevelopmentStopped {
    if (Test-Path -LiteralPath $developmentStateFile) {
        try {
            $state = Get-Content -LiteralPath $developmentStateFile -Raw |
                ConvertFrom-Json
            foreach ($property in @(
                "pid", "port", "project_root", "python_path", "process_start_utc"
            )) {
                if (-not ($state.PSObject.Properties.Name -contains $property)) {
                    throw "Development runtime state is missing '$property'."
                }
            }
            if ([int]$state.port -ne 8001 -or
                [IO.Path]::GetFullPath([string]$state.project_root) -ine $projectRoot -or
                [IO.Path]::GetFullPath([string]$state.python_path) -ine $python) {
                throw "Development runtime state identifies another controller."
            }

            $processId = [int]$state.pid
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                $recordedStart = [DateTimeOffset]::Parse(
                    [string]$state.process_start_utc
                ).UtcDateTime
                $actualStart = $process.StartTime.ToUniversalTime()
                $sameStart = [Math]::Abs(
                    ($actualStart - $recordedStart).TotalSeconds
                ) -le 2
                $sameExecutable = -not $process.Path -or (
                    [IO.Path]::GetFullPath($process.Path) -ieq $python
                )
                if ($sameStart -and $sameExecutable) {
                    throw "Development is running. Stop it before replacing its database."
                }
            }
        }
        catch {
            if ($_.Exception.Message -like "Development is running*") { throw }
            throw (
                "Development runtime state is unreadable. Run development.bat " +
                "stop before replacing its database."
            )
        }
    }

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.BeginConnect("127.0.0.1", 8001, $null, $null)
        if ($connection.AsyncWaitHandle.WaitOne(500)) {
            $client.EndConnect($connection)
            throw (
                "Port 8001 is active without a safe stopped state. Stop the " +
                "development server before replacing its database."
            )
        }
    }
    catch [System.Net.Sockets.SocketException] { }
    finally { $client.Dispose() }
}

function Invoke-Native(
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$FailureMessage
) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)."
    }
}

function Invoke-WithPgPassword(
    [string]$Password,
    [scriptblock]$Operation
) {
    $previousPassword = [Environment]::GetEnvironmentVariable("PGPASSWORD", "Process")
    try {
        [Environment]::SetEnvironmentVariable("PGPASSWORD", $Password, "Process")
        & $Operation
    }
    finally {
        [Environment]::SetEnvironmentVariable("PGPASSWORD", $previousPassword, "Process")
    }
}

function ConvertTo-SqlLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function Invoke-PostgresQuery(
    [hashtable]$Config,
    [string]$Database,
    [string]$Sql,
    [string]$FailureMessage
) {
    $psql = Find-PostgresTool "psql" $Config
    $databaseUser = Get-ConfigValue $Config "DB_USER" "postgres"
    $databasePassword = Get-ConfigValue $Config "DB_PASSWORD"
    if (-not $databasePassword) {
        throw "DB_PASSWORD is missing for PostgreSQL user '$databaseUser'."
    }
    return Invoke-WithPgPassword $databasePassword {
        $output = & $psql @(
            "--host=$(Get-ConfigValue $Config 'DB_HOST' '127.0.0.1')",
            "--port=$(Get-ConfigValue $Config 'DB_PORT' '5432')",
            "--username=$databaseUser",
            "--dbname=$Database",
            "--no-align",
            "--tuples-only",
            "--set=ON_ERROR_STOP=1",
            "--command=$Sql"
        )
        if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
        return @($output)
    }
}

function Ensure-DevelopmentDatabaseRole(
    [hashtable]$ProductionConfig,
    [string]$DevelopmentPassword
) {
    $productionPassword = Get-ConfigValue $ProductionConfig "DB_PASSWORD"
    if (-not $productionPassword) {
        throw "DB_PASSWORD is missing from the production .env."
    }
    $productionDatabase = Get-ConfigValue $ProductionConfig "DB_NAME" "postgres"
    if ($productionDatabase -notmatch '^[A-Za-z][A-Za-z0-9_]{0,62}$') {
        throw "Production DB_NAME is not a safe PostgreSQL identifier."
    }
    $roleLiteral = ConvertTo-SqlLiteral $requiredDevelopmentRole
    $passwordLiteral = ConvertTo-SqlLiteral $DevelopmentPassword
    $sql = @"
DO `$pharmacy`$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = $roleLiteral) THEN
        CREATE ROLE "$requiredDevelopmentRole";
    END IF;
END
`$pharmacy`$;
ALTER ROLE "$requiredDevelopmentRole" WITH LOGIN NOINHERIT NOSUPERUSER NOCREATEROLE CREATEDB NOREPLICATION NOBYPASSRLS PASSWORD $passwordLiteral;
REVOKE ALL PRIVILEGES ON DATABASE "$productionDatabase" FROM "$requiredDevelopmentRole";
"@
    Invoke-PostgresQuery $ProductionConfig $productionDatabase $sql `
        "Could not create or secure the dedicated development PostgreSQL role." |
        Out-Null

    $roleCheck = (
        Invoke-PostgresQuery $ProductionConfig $productionDatabase (
            "SELECT rolcanlogin::text || '|' || rolinherit::text || '|' || " +
            "rolsuper::text || '|' || rolcreaterole::text || '|' || " +
            "rolcreatedb::text || '|' || rolreplication::text || '|' || " +
            "rolbypassrls::text || '|' || " +
            "(SELECT COUNT(*)::text FROM pg_auth_members m " +
            "WHERE m.member = r.oid) FROM pg_roles r WHERE rolname = $roleLiteral"
        ) "Could not verify the development PostgreSQL role."
    ) -join ""
    if ($roleCheck.Trim() -cne "true|false|false|false|true|false|false|0") {
        throw (
            "The development PostgreSQL role must be login-only, non-inheriting, " +
            "non-privileged, and have no role memberships."
        )
    }
}

function Get-DatabaseMetadata(
    [hashtable]$Config,
    [string]$DatabaseName
) {
    $databaseLiteral = ConvertTo-SqlLiteral $DatabaseName
    $line = @(
        Invoke-PostgresQuery $Config "template1" (
            "SELECT pg_get_userbyid(datdba) || '|' || " +
            "COALESCE(shobj_description(oid, 'pg_database'), '') " +
            "FROM pg_database WHERE datname = $databaseLiteral"
        ) "Could not inspect PostgreSQL database '$DatabaseName'."
    ) | Where-Object { ([string]$_).Trim() } | Select-Object -Last 1
    if (-not $line) {
        return [pscustomobject]@{ Exists = $false; Owner = ""; Comment = "" }
    }
    $parts = ([string]$line).Trim().Split("|", 2)
    return [pscustomobject]@{
        Exists = $true
        Owner = $parts[0]
        Comment = $(if ($parts.Count -gt 1) { $parts[1] } else { "" })
    }
}

function Assert-ManagedDevelopmentDatabase(
    [hashtable]$Config,
    [string]$DatabaseName
) {
    $metadata = Get-DatabaseMetadata $Config $DatabaseName
    if (-not $metadata.Exists) {
        throw "Managed development database '$DatabaseName' does not exist."
    }
    if ($metadata.Owner -cne $requiredDevelopmentRole -or
        $metadata.Comment -cne $managedDatabaseComment) {
        throw (
            "Refusing to modify '$DatabaseName': it is not owned and marked " +
            "by the pharmacy development workflow."
        )
    }
    return $metadata
}

function Set-ManagedDatabaseComment(
    [hashtable]$Config,
    [string]$DatabaseName
) {
    Invoke-PostgresQuery $Config "template1" (
        "COMMENT ON DATABASE `"$DatabaseName`" IS " +
        (ConvertTo-SqlLiteral $managedDatabaseComment)
    ) "Could not mark managed development database '$DatabaseName'." | Out-Null
}

function Invoke-WithDevelopmentOperationLock(
    [scriptblock]$Operation,
    [int]$TimeoutMilliseconds = 5000
) {
    New-Item -ItemType Directory -Force -Path (Split-Path $developmentOperationLock -Parent) |
        Out-Null
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
        catch [IO.IOException] {
            Start-Sleep -Milliseconds 100
        }
    }
    if ($null -eq $stream) {
        throw "Another development start, stop, or data refresh is already running."
    }
    try { & $Operation }
    finally { $stream.Dispose() }
}

function Assert-SnapshotChecksum([IO.FileInfo]$Snapshot) {
    $checksumPath = "$($Snapshot.FullName).sha256"
    if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
        throw "Production snapshot checksum is missing."
    }
    $expected = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]
    if ($expected -notmatch '^[a-fA-F0-9]{64}$') {
        throw "Production snapshot checksum record is invalid."
    }
    $actual = (Get-FileHash -LiteralPath $Snapshot.FullName -Algorithm SHA256).Hash
    if ($actual -ine $expected) {
        throw "Production snapshot checksum verification failed."
    }
}

function Get-ReportedSnapshot([object[]]$Output) {
    $candidate = @(
        $Output | ForEach-Object { ([string]$_).Trim() } |
            Where-Object { $_ -match '(?i)^[A-Z]:\\.*\.dump$' }
    ) | Select-Object -Last 1
    if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Production snapshot did not report a readable PostgreSQL dump."
    }
    if (-not (Test-Path -LiteralPath "$candidate.sha256" -PathType Leaf)) {
        throw "Production snapshot checksum is missing."
    }
    $snapshot = Get-Item -LiteralPath $candidate
    Assert-SnapshotChecksum $snapshot
    return $snapshot
}

function New-ProductionSnapshot([string]$OperationDirectory) {
    if ($productionRuntimeRoot -ine $projectRoot) {
        $recoveryBlock = Join-Path `
            $productionRuntimeRoot ".runtime\production-recovery-required.json"
        if (Test-Path -LiteralPath $recoveryBlock -PathType Leaf) {
            throw "Production requires recovery; development refresh is blocked."
        }
        if (-not (Test-Path -LiteralPath $productionController -PathType Leaf)) {
            throw "Production controller is missing: $productionController"
        }
        $controllerSource = Get-Content -LiteralPath $productionController -Raw
        $arguments = @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", $productionController, "-Action", "backup", "-NoBrowser"
        )
        if ($controllerSource -match '(?m)\[switch\]\s*\$NonInteractive\b') {
            $arguments += "-NonInteractive"
        }
        $output = @()
        $exitCode = 1
        $previousErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = @(& powershell.exe @arguments 2>&1)
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }
        $output | ForEach-Object { Write-Host ([string]$_) }
        if ($exitCode -ne 0) {
            throw "Production snapshot failed through the release gate (exit code $exitCode)."
        }
        return Get-ReportedSnapshot $output
    }

    $output = @()
    $exitCode = 1
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
            -File $backupScript -Reason "development-refresh" `
            -OutputDirectory $OperationDirectory -ForceNew 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    $output | ForEach-Object { Write-Host ([string]$_) }
    if ($exitCode -ne 0) {
        throw "Production snapshot failed with exit code $exitCode."
    }
    return Get-ReportedSnapshot $output
}

function Initialize-DevelopmentDatabase(
    [hashtable]$DevelopmentConfig,
    [object]$Safety
) {
    $createdb = Find-PostgresTool "createdb" $DevelopmentConfig
    $databaseUser = Get-ConfigValue $DevelopmentConfig "DB_USER"
    $databasePassword = Get-ConfigValue $DevelopmentConfig "DB_PASSWORD"
    if (-not $databasePassword) {
        throw "DB_PASSWORD is missing from .env.development."
    }

    $metadata = Get-DatabaseMetadata $DevelopmentConfig $Safety.DevelopmentName
    if ($metadata.Exists) {
        Assert-ManagedDevelopmentDatabase `
            $DevelopmentConfig $Safety.DevelopmentName | Out-Null
        Write-Host "Development database: $($Safety.DevelopmentName) (already exists)" -ForegroundColor Green
        return
    }

    Invoke-WithPgPassword $databasePassword {
        Invoke-Native $createdb @(
            "--host=$($Safety.DevelopmentHost)",
            "--port=$($Safety.DevelopmentPort)",
            "--username=$databaseUser",
            "--maintenance-db=template1",
            "--owner=$requiredDevelopmentRole",
            $Safety.DevelopmentName
        ) "Could not create development database"
    }
    Set-ManagedDatabaseComment $DevelopmentConfig $Safety.DevelopmentName
    Assert-ManagedDevelopmentDatabase `
        $DevelopmentConfig $Safety.DevelopmentName | Out-Null
    Write-Host "Created development database: $($Safety.DevelopmentName)" -ForegroundColor Green
}

function Invoke-DevelopmentDatabaseCheck(
    [string]$DatabaseName,
    [switch]$PrepareSnapshot
) {
    $previousOverride = [Environment]::GetEnvironmentVariable(
        "PHARMACY_DEVELOPMENT_DB_OVERRIDE", "Process"
    )
    $previousSettings = [Environment]::GetEnvironmentVariable(
        "DJANGO_SETTINGS_MODULE", "Process"
    )
    try {
        $overrideValue = if ($DatabaseName -ceq $requiredDevelopmentDatabase) {
            $null
        }
        else { $DatabaseName }
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_DEVELOPMENT_DB_OVERRIDE", $overrideValue, "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "DJANGO_SETTINGS_MODULE", "inventory.settings_development", "Process"
        )
        Push-Location $projectRoot
        try {
            if ($PrepareSnapshot) {
                & $python manage.py prepare_development_snapshot
                if ($LASTEXITCODE -ne 0) {
                    throw "Development snapshot cleanup failed with exit code $LASTEXITCODE."
                }
            }
            & $python manage.py check
            if ($LASTEXITCODE -ne 0) {
                throw "Development database check failed with exit code $LASTEXITCODE."
            }
        }
        finally { Pop-Location }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_DEVELOPMENT_DB_OVERRIDE", $previousOverride, "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "DJANGO_SETTINGS_MODULE", $previousSettings, "Process"
        )
    }
}

function Refresh-DevelopmentDatabase(
    [hashtable]$ProductionConfig,
    [hashtable]$DevelopmentConfig,
    [object]$Safety
) {
    Assert-DevelopmentStopped
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Virtual environment not found. Run setup_env.bat first."
    }
    if (-not (Test-Path -LiteralPath $backupScript)) {
        throw "Production backup script is missing: $backupScript"
    }
    if (-not $ConfirmDatabaseName) {
        $script:ConfirmDatabaseName = Read-Host (
            "Type the development database name '$($Safety.DevelopmentName)' to replace its contents"
        )
    }
    if ($ConfirmDatabaseName -cne $Safety.DevelopmentName) {
        throw "Development refresh cancelled: database-name confirmation did not match."
    }

    Assert-ManagedDevelopmentDatabase `
        $DevelopmentConfig $Safety.DevelopmentName | Out-Null

    $refreshRoot = New-Item -ItemType Directory -Force -Path $runtimeDirectory
    $operationDirectory = Join-Path $refreshRoot.FullName ([Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $operationDirectory | Out-Null
    $completed = $false
    $targetChanged = $false
    $rollbackRestored = $false
    $suffix = [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $stagingDatabase = "$($Safety.DevelopmentName)_refresh_$suffix"
    $previousDatabase = "$($Safety.DevelopmentName)_previous_$suffix"
    $failedDatabase = "$($Safety.DevelopmentName)_failed_$suffix"
    $oldDatabaseRenamed = $false
    $stagingRenamed = $false
    $dropdb = Find-PostgresTool "dropdb" $DevelopmentConfig
    $createdb = Find-PostgresTool "createdb" $DevelopmentConfig
    $pgRestore = Find-PostgresTool "pg_restore" $DevelopmentConfig
    $databaseUser = Get-ConfigValue $DevelopmentConfig "DB_USER"
    $databasePassword = Get-ConfigValue $DevelopmentConfig "DB_PASSWORD"
    if (-not $databasePassword) {
        throw "DB_PASSWORD is missing from .env.development."
    }

    try {
        New-Item -ItemType Directory -Force -Path (Split-Path $refreshIncompleteMarker -Parent) |
            Out-Null
        [ordered]@{
            schema_version = 1
            database = $Safety.DevelopmentName
            staging_database = $stagingDatabase
            started_at = (Get-Date).ToString("o")
            reason = "A development snapshot refresh is in progress."
        } | ConvertTo-Json | Set-Content -LiteralPath $refreshIncompleteMarker -Encoding UTF8

        Write-Host "Creating a verified production snapshot..." -ForegroundColor Cyan
        $snapshot = New-ProductionSnapshot $operationDirectory
        Invoke-Native $pgRestore @("--list", $snapshot.FullName) `
            "The production snapshot is not readable" | Out-Null

        Write-Host "Restoring into isolated staging database '$stagingDatabase'..." -ForegroundColor Cyan
        Invoke-WithPgPassword $databasePassword {
            Invoke-Native $createdb @(
                "--host=$($Safety.DevelopmentHost)",
                "--port=$($Safety.DevelopmentPort)",
                "--username=$databaseUser",
                "--maintenance-db=template1",
                "--owner=$requiredDevelopmentRole",
                $stagingDatabase
            ) "Could not create the staging development database"
        }
        Set-ManagedDatabaseComment $DevelopmentConfig $stagingDatabase

        Invoke-WithPgPassword $databasePassword {
            Invoke-Native $pgRestore @(
                "--host=$($Safety.DevelopmentHost)",
                "--port=$($Safety.DevelopmentPort)",
                "--username=$databaseUser",
                "--dbname=$stagingDatabase",
                "--no-owner",
                "--no-acl",
                "--exit-on-error",
                $snapshot.FullName
            ) "Could not restore the development snapshot"
        }
        Invoke-DevelopmentDatabaseCheck $stagingDatabase -PrepareSnapshot
        Assert-ManagedDevelopmentDatabase $DevelopmentConfig $stagingDatabase |
            Out-Null

        $targetLiteral = ConvertTo-SqlLiteral $Safety.DevelopmentName
        $stagingLiteral = ConvertTo-SqlLiteral $stagingDatabase
        Invoke-PostgresQuery $DevelopmentConfig "template1" (
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity " +
            "WHERE datname IN ($targetLiteral, $stagingLiteral) " +
            "AND pid <> pg_backend_pid()"
        ) "Could not close development database connections for the safe swap." |
            Out-Null

        Invoke-PostgresQuery $DevelopmentConfig "template1" (
            "ALTER DATABASE `"$($Safety.DevelopmentName)`" RENAME TO `"$previousDatabase`""
        ) "Could not preserve the previous development database." | Out-Null
        $oldDatabaseRenamed = $true
        $targetChanged = $true
        try {
            Invoke-PostgresQuery $DevelopmentConfig "template1" (
                "ALTER DATABASE `"$stagingDatabase`" RENAME TO `"$($Safety.DevelopmentName)`""
            ) "Could not activate the staged development database." | Out-Null
            $stagingRenamed = $true
        }
        catch {
            Invoke-PostgresQuery $DevelopmentConfig "template1" (
                "ALTER DATABASE `"$previousDatabase`" RENAME TO `"$($Safety.DevelopmentName)`""
            ) "Could not restore the previous development database name." | Out-Null
            $oldDatabaseRenamed = $false
            $targetChanged = $false
            throw
        }

        Invoke-DevelopmentDatabaseCheck $Safety.DevelopmentName
        Assert-ManagedDevelopmentDatabase `
            $DevelopmentConfig $previousDatabase | Out-Null
        Invoke-WithPgPassword $databasePassword {
            Invoke-Native $dropdb @(
                "--host=$($Safety.DevelopmentHost)",
                "--port=$($Safety.DevelopmentPort)",
                "--username=$databaseUser",
                "--maintenance-db=template1",
                "--force",
                $previousDatabase
            ) "Could not remove the preserved previous development database"
        }
        $oldDatabaseRenamed = $false

        if (Test-Path -LiteralPath $refreshIncompleteMarker) {
            Remove-Item -LiteralPath $refreshIncompleteMarker -Force
        }
        $completed = $true
        Write-Host (
            "Development now contains a safe snapshot of production data in " +
            "'$($Safety.DevelopmentName)'."
        ) -ForegroundColor Green
        Write-DevelopmentDataLog (
            "Refreshed '$($Safety.DevelopmentName)' from '$($Safety.ProductionName)' successfully."
        )
    }
    finally {
        if (-not $completed -and $oldDatabaseRenamed) {
            try {
                $activeName = if ($stagingRenamed) {
                    $Safety.DevelopmentName
                }
                else { $stagingDatabase }
                $activeLiteral = ConvertTo-SqlLiteral $activeName
                $previousLiteral = ConvertTo-SqlLiteral $previousDatabase
                Invoke-PostgresQuery $DevelopmentConfig "template1" (
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity " +
                    "WHERE datname IN ($activeLiteral, $previousLiteral) " +
                    "AND pid <> pg_backend_pid()"
                ) "Could not close development connections during rollback." |
                    Out-Null
                if ($stagingRenamed) {
                    Invoke-PostgresQuery $DevelopmentConfig "template1" (
                        "ALTER DATABASE `"$($Safety.DevelopmentName)`" RENAME TO `"$failedDatabase`""
                    ) "Could not quarantine the failed staged database." | Out-Null
                }
                Invoke-PostgresQuery $DevelopmentConfig "template1" (
                    "ALTER DATABASE `"$previousDatabase`" RENAME TO `"$($Safety.DevelopmentName)`""
                ) "Could not restore the previous development database." | Out-Null
                $rollbackRestored = $true
                $oldDatabaseRenamed = $false
                $targetChanged = $false
            }
            catch {
                Write-DevelopmentDataLog "CRITICAL Development database swap rollback failed: $($_.Exception.Message)"
            }
        }

        foreach ($temporaryDatabase in @($stagingDatabase, $failedDatabase)) {
            try {
                $metadata = Get-DatabaseMetadata $DevelopmentConfig $temporaryDatabase
                if ($metadata.Exists -and
                    $metadata.Owner -ceq $requiredDevelopmentRole -and
                    $metadata.Comment -ceq $managedDatabaseComment) {
                    Invoke-WithPgPassword $databasePassword {
                        Invoke-Native $dropdb @(
                            "--host=$($Safety.DevelopmentHost)",
                            "--port=$($Safety.DevelopmentPort)",
                            "--username=$databaseUser",
                            "--maintenance-db=template1",
                            "--force",
                            $temporaryDatabase
                        ) "Could not remove temporary development database"
                    }
                }
            }
            catch {
                Write-DevelopmentDataLog "Could not clean temporary database '$temporaryDatabase': $($_.Exception.Message)"
            }
        }

        if (-not $completed -and -not $targetChanged -and
            (Test-Path -LiteralPath $refreshIncompleteMarker)) {
            Remove-Item -LiteralPath $refreshIncompleteMarker -Force
        }
        if (Test-Path -LiteralPath $operationDirectory) {
            $resolvedOperation = (Resolve-Path -LiteralPath $operationDirectory).Path
            $resolvedRoot = (Resolve-Path -LiteralPath $runtimeDirectory).Path
            if ([IO.Path]::GetDirectoryName($resolvedOperation) -cne $resolvedRoot) {
                throw "Refusing to remove an unexpected refresh directory: $resolvedOperation"
            }
            Remove-Item -LiteralPath $resolvedOperation -Recurse -Force
        }
        if (-not $completed) {
            Write-DevelopmentDataLog (
                "Refresh failed; previous development database restored=$rollbackRestored. " +
                "Sensitive temporary snapshot files were removed."
            )
        }
    }
}

function Invoke-SelectedDevelopmentDataAction {
    $productionConfig = Read-DotEnv $productionEnvFile
    if ($Action -in @("configure", "setup")) {
        Ensure-SecureDevelopmentEnvironment $productionConfig
    }
    $developmentConfig = Read-DotEnv $developmentEnvFile
    $safety = Assert-SafeDatabaseConfiguration $productionConfig $developmentConfig

    switch ($Action) {
        "status" {
            if (Test-Path -LiteralPath $refreshIncompleteMarker) {
                throw (
                    "The last development data refresh did not complete its session and " +
                    "automation cleanup. Rerun Refresh Development Data before starting development."
                )
            }
            Assert-ManagedDevelopmentDatabase `
                $developmentConfig $safety.DevelopmentName | Out-Null
            Write-Host "Production database:  $($safety.ProductionName)@$($safety.ProductionHost):$($safety.ProductionPort)"
            Write-Host "Development database: $($safety.DevelopmentName)@$($safety.DevelopmentHost):$($safety.DevelopmentPort)"
            Write-Host "Development role:     $($safety.DevelopmentUser)"
            Write-Host "Django test database: $($safety.TestName)"
            Write-Host "Same PostgreSQL server: $(if ($safety.SameServer) { 'yes' } else { 'no' })"
            Write-Host "Database isolation: safe" -ForegroundColor Green
        }
        "configure" {
            Write-Host "Development environment configuration is ready." -ForegroundColor Green
        }
        "initialize" {
            Initialize-DevelopmentDatabase $developmentConfig $safety
        }
        "setup" {
            Initialize-DevelopmentDatabase $developmentConfig $safety
            Write-Host "Development environment setup is ready for a data refresh." -ForegroundColor Green
        }
        "refresh" {
            Refresh-DevelopmentDatabase $productionConfig $developmentConfig $safety
        }
    }
}

try {
    if ($Action -eq "status") {
        Invoke-SelectedDevelopmentDataAction
    }
    else {
        Invoke-WithDevelopmentOperationLock {
            Invoke-SelectedDevelopmentDataAction
        } -TimeoutMilliseconds 30000
    }
}
catch {
    Write-DevelopmentDataLog "FAILED $($_.Exception.Message)"
    Write-Host "Development data command failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
