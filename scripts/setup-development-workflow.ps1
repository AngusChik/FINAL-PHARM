param(
    [string]$ProductionWorktree = "",
    [string]$DevelopmentBranch = "development",
    [string]$ProductionBranch = "main",
    [string]$Remote = "origin",
    [switch]$CreateDevelopmentBranch,
    [switch]$SkipFetch,
    [switch]$SkipDependencies,
    [switch]$InstallStartup,
    [switch]$EnableAutoStart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectParent = Split-Path -Parent $projectRoot
$projectName = Split-Path -Leaf $projectRoot
$runtimeDirectory = Join-Path $projectRoot ".runtime"
$workflowConfigFile = Join-Path $runtimeDirectory "development-workflow.json"
$developmentDataScript = Join-Path $PSScriptRoot "development-data.ps1"
$expectedOriginUrl = "https://github.com/AngusChik/FINAL-PHARM.git"
$firstCutover = -not (Test-Path -LiteralPath $workflowConfigFile -PathType Leaf)

if (-not $ProductionWorktree) {
    $ProductionWorktree = Join-Path $projectParent "$projectName-PRODUCTION"
}
$ProductionWorktree = [IO.Path]::GetFullPath($ProductionWorktree)

function Invoke-Git([string[]]$Arguments, [string]$FailureMessage) {
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& git @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ($exitCode -ne 0) {
        throw "$FailureMessage`n$($output -join [Environment]::NewLine)"
    }
    return @($output)
}

function Write-JsonAtomic([string]$Path, [object]$Value) {
    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $temporary = Join-Path $directory (
        ".$(Split-Path -Leaf $Path).$([Guid]::NewGuid().ToString('N')).tmp"
    )
    try {
        $json = $Value | ConvertTo-Json -Depth 8
        [IO.File]::WriteAllText(
            $temporary,
            $json + [Environment]::NewLine,
            (New-Object Text.UTF8Encoding($false))
        )
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Test-GitReference([string]$Reference) {
    & git show-ref --verify --quiet $Reference
    return $LASTEXITCODE -eq 0
}

function Assert-SafeProductionPath {
    if ($DevelopmentBranch -cne "development" -or
        $ProductionBranch -cne "main" -or
        $Remote -cne "origin") {
        throw (
            "This workflow requires the local-only 'development' branch and " +
            "production 'origin/main'. Custom branch or remote names are not supported."
        )
    }
    if ($EnableAutoStart -and -not $InstallStartup) {
        throw "-EnableAutoStart requires -InstallStartup."
    }
    $developmentWithSeparator = $projectRoot.TrimEnd('\') + '\'
    $productionWithSeparator = $ProductionWorktree.TrimEnd('\') + '\'
    if ($ProductionWorktree -ieq $projectRoot -or
        $productionWithSeparator.StartsWith($developmentWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Production worktree must be a sibling of development, not development itself or one of its children."
    }
    if ([IO.Path]::GetPathRoot($ProductionWorktree) -eq $ProductionWorktree) {
        throw "A drive root cannot be used as the production worktree."
    }
    if ((Split-Path $ProductionWorktree -Parent) -ine $projectParent) {
        throw "Production worktree must be a direct sibling of development."
    }
}

function Test-TcpPort([string]$ComputerName, [int]$PortNumber) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect($ComputerName, $PortNumber, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne(500)) { return $false }
        $client.EndConnect($result)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Assert-LegacyProductionStopped {
    if (-not $firstCutover) { return }
    foreach ($port in @(8000, 443)) {
        if (Test-TcpPort "127.0.0.1" $port) {
            throw (
                "Legacy production is still using port $port. Run " +
                "production.bat stop from this checkout before the first cutover."
            )
        }
    }
    $legacyStatePath = Join-Path $projectRoot ".runtime\production.json"
    if (Test-Path -LiteralPath $legacyStatePath -PathType Leaf) {
        try {
            $state = Get-Content -LiteralPath $legacyStatePath -Raw | ConvertFrom-Json
            foreach ($property in @("waitress_pid", "caddy_pid")) {
                if ($state.PSObject.Properties.Name -contains $property) {
                    $processId = [int]$state.$property
                    if ($processId -gt 0 -and
                        (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
                        throw "Legacy production still has tracked process $processId."
                    }
                }
            }
        }
        catch {
            if ($_.Exception.Message -like "Legacy production still*") { throw }
            throw "Legacy production state is unreadable; verify and stop it before cutover."
        }
    }
}

function Assert-DevelopmentRepository {
    Set-Location $projectRoot
    $inside = @(
        Invoke-Git @("rev-parse", "--is-inside-work-tree") `
            "Development folder is not a Git worktree"
    )[-1].Trim()
    if ($inside -ne "true") { throw "Development folder is not a Git worktree." }

    $changes = @(Invoke-Git @("status", "--porcelain") "Could not inspect Git status")
    if ($changes.Count -gt 0 -and ($changes -join "").Trim()) {
        throw "Commit or discard development changes before provisioning production."
    }

    $originUrl = @(
        Invoke-Git @("remote", "get-url", $Remote) `
            "Git remote '$Remote' is not configured"
    )[-1].Trim()
    if ($originUrl.TrimEnd('/') -ine $expectedOriginUrl.TrimEnd('/')) {
        throw "Remote '$Remote' must resolve to $expectedOriginUrl, not $originUrl."
    }

    if (-not $SkipFetch) {
        Invoke-Git @("fetch", "--prune", $Remote) "Could not refresh $Remote" | Out-Null
    }

    $currentBranch = @(
        Invoke-Git @("branch", "--show-current") `
            "Could not determine the development branch"
    )[-1].Trim()
    $reviewedStartCommit = @(
        Invoke-Git @("rev-parse", "HEAD") `
            "Could not determine the reviewed development commit"
    )[-1].Trim()
    if ($currentBranch -ne $DevelopmentBranch) {
        if (-not $CreateDevelopmentBranch) {
            throw (
                "Current branch is '$currentBranch'. Switch to '$DevelopmentBranch' or rerun " +
                "with -CreateDevelopmentBranch after reviewing the current commit."
            )
        }
        if (Test-GitReference "refs/heads/$DevelopmentBranch") {
            $existingDevelopmentCommit = @(
                Invoke-Git @(
                    "rev-parse", "refs/heads/$DevelopmentBranch"
                ) "Could not inspect the existing development branch"
            )[-1].Trim()
            if ($existingDevelopmentCommit -ne $reviewedStartCommit) {
                throw (
                    "Local branch '$DevelopmentBranch' already exists at a different " +
                    "commit. Switch to it, reconcile it with the reviewed commit " +
                    "'$reviewedStartCommit', and rerun setup without " +
                    "-CreateDevelopmentBranch."
                )
            }
            Invoke-Git @("switch", $DevelopmentBranch) `
                "Could not switch to existing development branch" | Out-Null
        }
        else {
            Invoke-Git @("switch", "-c", $DevelopmentBranch) `
                "Could not create development branch" | Out-Null
        }
    }

    $removedTracking = @()
    foreach ($settingName in @("remote", "merge", "pushRemote")) {
        $configKey = "branch.$DevelopmentBranch.$settingName"
        $existingValue = @(& git config --get-all $configKey 2>$null)
        if ($LASTEXITCODE -eq 0 -and $existingValue.Count -gt 0) {
            Invoke-Git @("config", "--unset-all", $configKey) `
                "Could not remove local-only branch setting '$configKey'" |
                Out-Null
            $removedTracking += $configKey
        }
        $remainingValue = @(& git config --get-all $configKey 2>$null)
        if ($LASTEXITCODE -eq 0 -or $remainingValue.Count -gt 0) {
            throw "Development branch still has prohibited Git setting '$configKey'."
        }
    }
    if ($removedTracking.Count -gt 0) {
        Write-Host (
            "Removed Git tracking from local-only branch '$DevelopmentBranch': " +
            ($removedTracking -join ", ")
        ) -ForegroundColor Yellow
    }
    $previousErrorPreference = $ErrorActionPreference
    try {
        # A local-only branch is expected to make this Git probe return 128 and
        # write its diagnostic to stderr.  PowerShell 7 can promote that stderr
        # record to a terminating error when the script preference is Stop, so
        # inspect the native exit code explicitly just as Invoke-Git does.
        $ErrorActionPreference = "Continue"
        $remainingUpstream = @(
            & git rev-parse --abbrev-ref --symbolic-full-name `
                "@{upstream}" 2>$null
        )
        $upstreamExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ($upstreamExitCode -eq 0 -or ($remainingUpstream -join "").Trim()) {
        throw "Development branch must remain local-only and have no upstream."
    }

    $remoteProduction = "refs/remotes/$Remote/$ProductionBranch"
    if (-not (Test-GitReference $remoteProduction)) {
        throw "Remote production branch '$Remote/$ProductionBranch' does not exist."
    }
    if (-not (Test-GitReference "refs/heads/$ProductionBranch")) {
        Invoke-Git @("branch", $ProductionBranch, "$Remote/$ProductionBranch") `
            "Could not create local production branch" | Out-Null
    }
    else {
        $localProductionCommit = @(
            Invoke-Git @(
                "rev-parse", "refs/heads/$ProductionBranch"
            ) "Could not read local production branch"
        )[-1].Trim()
        $remoteProductionCommit = @(
            Invoke-Git @(
                "rev-parse", "$Remote/$ProductionBranch"
            ) "Could not read remote production branch"
        )[-1].Trim()
        if ($localProductionCommit -eq $remoteProductionCommit) {
            return
        }
        & git merge-base --is-ancestor $ProductionBranch "$Remote/$ProductionBranch"
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Local '$ProductionBranch' is not an ancestor of '$Remote/$ProductionBranch'. " +
                "Reconcile it manually before provisioning."
            )
        }
        $worktreeList = (Invoke-Git @("worktree", "list", "--porcelain") `
            "Could not inspect registered worktrees") -join "`n"
        if ($worktreeList -match "(?m)^branch refs/heads/$([regex]::Escape($ProductionBranch))$") {
            throw (
                "Local '$ProductionBranch' is checked out in the production worktree " +
                "but does not match '$Remote/$ProductionBranch'. Use the guarded release " +
                "or pending-sync workflow instead of changing live code during setup."
            )
        }
        Invoke-Git @("branch", "-f", $ProductionBranch, "$Remote/$ProductionBranch") `
            "Could not fast-forward local production branch" | Out-Null
    }
}

function Ensure-ProductionWorktree {
    $registeredWorktrees = (Invoke-Git @("worktree", "list", "--porcelain") `
        "Could not inspect Git worktrees") -join "`n"
    $escapedPath = [regex]::Escape(($ProductionWorktree -replace '\\', '/'))
    $registered = $registeredWorktrees -match "(?im)^worktree\s+$escapedPath$"

    if (Test-Path -LiteralPath $ProductionWorktree) {
        $existingItems = @(Get-ChildItem -LiteralPath $ProductionWorktree -Force -ErrorAction Stop)
        if (-not $registered -and $existingItems.Count -gt 0) {
            throw "Production target exists and is not the configured Git worktree: $ProductionWorktree"
        }
    }
    elseif (-not $registered) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ProductionWorktree) | Out-Null
    }

    if (-not $registered) {
        Invoke-Git @("worktree", "add", $ProductionWorktree, $ProductionBranch) `
            "Could not create production worktree" | Out-Null
    }

    $productionGitRoot = (& git -C $ProductionWorktree rev-parse --show-toplevel 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or
        ([IO.Path]::GetFullPath($productionGitRoot) -ine $ProductionWorktree)) {
        throw "Production worktree verification failed: $ProductionWorktree"
    }
    $branch = (& git -C $ProductionWorktree branch --show-current).Trim()
    if ($branch -ne $ProductionBranch) {
        throw "Production worktree must remain on '$ProductionBranch', not '$branch'."
    }

    $changes = @(& git -C $ProductionWorktree status --porcelain 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect production Git status." }
    if (($changes -join "").Trim()) {
        throw "Production worktree must be clean before setup continues."
    }
    $productionCommit = (& git -C $ProductionWorktree rev-parse HEAD).Trim()
    $remoteCommit = (& git -C $ProductionWorktree rev-parse "$Remote/$ProductionBranch").Trim()
    if ($LASTEXITCODE -ne 0 -or $productionCommit -ne $remoteCommit) {
        throw "Production must exactly match $Remote/$ProductionBranch during setup."
    }
}

function Copy-RuntimeFileIfMissing([string]$RelativePath, [switch]$Required) {
    $source = Join-Path $projectRoot $RelativePath
    $destination = Join-Path $ProductionWorktree $RelativePath
    if (Test-Path -LiteralPath $destination) { return }
    if (-not (Test-Path -LiteralPath $source)) {
        if ($Required) { throw "Required production runtime file is missing: $source" }
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    $temporary = "$destination.setup-$([Guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $source -Destination $temporary
        Move-Item -LiteralPath $temporary -Destination $destination
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Copy-RuntimeDirectoryIfMissing([string]$RelativePath) {
    $source = Join-Path $projectRoot $RelativePath
    $destination = Join-Path $ProductionWorktree $RelativePath
    if ((Test-Path -LiteralPath $destination) -or
        -not (Test-Path -LiteralPath $source)) { return }
    $resolvedSource = (Resolve-Path -LiteralPath $source).Path
    if (-not $resolvedSource.StartsWith($projectRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to copy runtime data from outside development: $resolvedSource"
    }
    $temporary = "$destination.setup-$([Guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $resolvedSource -Destination $temporary -Recurse
        if (Test-Path -LiteralPath $destination) {
            throw "Production runtime destination appeared during copy: $destination"
        }
        Move-Item -LiteralPath $temporary -Destination $destination
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Recurse -Force
        }
    }
}

function Copy-ProductionRuntime {
    Copy-RuntimeFileIfMissing ".env" -Required
    Copy-RuntimeFileIfMissing "caddy.exe" -Required
    Copy-RuntimeFileIfMissing "Pharmacy-Root-Certificate.crt"
    Copy-RuntimeFileIfMissing "google_credentials.json"
    Copy-RuntimeFileIfMissing "gsheet_sync_state.json"

    foreach ($directory in @(
        "caddy_data",
        "backups",
        "logs",
        ".mckesson_profile",
        ".kohlfrisch_profile"
    )) {
        Copy-RuntimeDirectoryIfMissing $directory
    }
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $ProductionWorktree ".runtime"), `
        (Join-Path $ProductionWorktree "logs") | Out-Null
}

function Install-ProductionDependencies {
    $productionPython = Join-Path $ProductionWorktree "env\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $productionPython)) {
        & python -m venv (Join-Path $ProductionWorktree "env")
        if ($LASTEXITCODE -ne 0) { throw "Could not create the production virtual environment." }
    }
    & $productionPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update production pip." }
    & $productionPython -m pip install -r (Join-Path $ProductionWorktree "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install production dependencies." }
    & $productionPython -m playwright install chromium
    if ($LASTEXITCODE -ne 0) { throw "Could not install production Chromium." }
}

function Test-ProductionWorktree {
    $productionPython = Join-Path $ProductionWorktree "env\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $productionPython)) {
        throw "Production virtual environment is missing. Rerun without -SkipDependencies."
    }
    Push-Location $ProductionWorktree
    try {
        $env:DJANGO_SETTINGS_MODULE = "inventory.settings_production"
        & $productionPython manage.py check --deploy
        if ($LASTEXITCODE -ne 0) { throw "Production Django deployment check failed." }
        & $productionPython manage.py makemigrations --check --dry-run
        if ($LASTEXITCODE -ne 0) { throw "Production has model changes without migrations." }

        $caddy = Join-Path $ProductionWorktree "caddy.exe"
        $envFile = Join-Path $ProductionWorktree ".env"
        $hostLine = Get-Content -LiteralPath $envFile |
            Where-Object { $_.TrimStart().StartsWith("PHARMACY_HOST=") } |
            Select-Object -First 1
        if (-not $hostLine) { throw "Production .env is missing PHARMACY_HOST." }
        $env:PHARMACY_HOST = $hostLine.Split("=", 2)[1].Trim().Trim('"').Trim("'")
        $env:XDG_DATA_HOME = Join-Path $ProductionWorktree "caddy_data"
        & $caddy validate --config (Join-Path $ProductionWorktree "Caddyfile")
        if ($LASTEXITCODE -ne 0) { throw "Production Caddy validation failed." }
    }
    finally {
        Pop-Location
    }
}

function Write-WorkflowConfiguration {
    Write-JsonAtomic $workflowConfigFile ([ordered]@{
        schema_version = 1
        development_branch = $DevelopmentBranch
        production_branch = $ProductionBranch
        remote = $Remote
        expected_origin_url = $expectedOriginUrl
        production_worktree = $ProductionWorktree
        configured_at = (Get-Date).ToString("o")
    })
}

function Write-ProductionRoleMarker {
    $productionRuntime = Join-Path $ProductionWorktree ".runtime"
    New-Item -ItemType Directory -Force -Path $productionRuntime | Out-Null
    Write-JsonAtomic (Join-Path $productionRuntime "production-role.json") ([ordered]@{
        schema_version = 1
        role = "production"
        worktree = $ProductionWorktree
        branch = $ProductionBranch
        remote = $Remote
        created_at = (Get-Date).ToString("o")
    })
}

function Assert-ExistingProductionRoleMarker {
    $path = Join-Path $ProductionWorktree ".runtime\production-role.json"
    try {
        $marker = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        $markedRoot = [IO.Path]::GetFullPath([string]$marker.worktree).TrimEnd('\')
        if ([int]$marker.schema_version -ne 1 -or
            [string]$marker.role -cne "production" -or
            [string]$marker.branch -cne $ProductionBranch -or
            [string]$marker.remote -cne $Remote -or
            $markedRoot -ine $ProductionWorktree.TrimEnd('\')) {
            throw "marker fields do not match the configured production worktree"
        }
        $createdAt = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse(
            [string]$marker.created_at,
            [ref]$createdAt
        )) {
            throw "created_at is invalid"
        }
    }
    catch {
        throw "Existing production role marker is invalid: $($_.Exception.Message)"
    }
}

function Install-ProductionStartupExperience {
    $taskMigrator = Join-Path `
        $ProductionWorktree "scripts\migrate-production-task-paths.ps1"
    if (-not (Test-Path -LiteralPath $taskMigrator)) {
        throw "Production task-path migrator is missing: $taskMigrator"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $taskMigrator `
        -DevelopmentWorktree $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Existing pharmacy scheduled-task path migration failed."
    }

    $installer = Join-Path $ProductionWorktree "scripts\install-production-startup.ps1"
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "Production startup installer is missing: $installer"
    }
    $arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $installer
    )
    if ($EnableAutoStart) { $arguments += "-EnableAutoStart" }
    & powershell.exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Production startup/shortcut installation failed." }
}

try {
    Assert-SafeProductionPath
    Assert-LegacyProductionStopped
    if (-not (Test-Path -LiteralPath $developmentDataScript)) {
        throw "Development data controller is missing: $developmentDataScript"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $developmentDataScript -Action status
    if ($LASTEXITCODE -ne 0) { throw "Development database isolation is not ready." }

    Assert-DevelopmentRepository
    Ensure-ProductionWorktree
    Copy-ProductionRuntime
    if (-not $SkipDependencies) { Install-ProductionDependencies }
    $roleMarkerPath = Join-Path `
        $ProductionWorktree ".runtime\production-role.json"
    $roleMarkerCreated = -not (
        Test-Path -LiteralPath $roleMarkerPath -PathType Leaf
    )
    if ($roleMarkerCreated) {
        Write-ProductionRoleMarker
    }
    else {
        Assert-ExistingProductionRoleMarker
    }
    try {
        Test-ProductionWorktree
    }
    catch {
        if ($roleMarkerCreated -and (Test-Path -LiteralPath $roleMarkerPath)) {
            Remove-Item -LiteralPath $roleMarkerPath -Force
        }
        throw
    }
    if ($InstallStartup) { Install-ProductionStartupExperience }
    Write-WorkflowConfiguration

    Write-Host "Development-first workflow configured." -ForegroundColor Green
    Write-Host "Development: $projectRoot ($DevelopmentBranch)"
    Write-Host "Production:  $ProductionWorktree ($ProductionBranch)"
    Write-Host "Configuration: $workflowConfigFile"
}
catch {
    Write-Host "Development workflow setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
