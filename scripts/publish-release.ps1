[CmdletBinding()]
param(
    [ValidateSet("check", "publish", "status")]
    [string]$Action = "check",

    [string]$DevelopmentBranch = "",
    [string]$ProductionBranch = "",
    [string]$Remote = "",
    [string]$ExpectedOriginUrl = "",
    [string]$DevelopmentWorktree = "",
    [string]$ProductionWorktree = "",
    [string]$StateDirectory = "",
    [string]$WorkflowConfig = "",

    [switch]$ConfirmRelease,
    [switch]$DryRun,

    # Test seams. Normal operators should not set these parameters.
    [scriptblock]$CommandInvoker,
    [scriptblock]$Clock,
    [switch]$ThrowOnError
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptRoot = $PSScriptRoot
$defaultDevelopmentWorktree = (Resolve-Path (Join-Path $scriptRoot "..")).Path
if (-not $DevelopmentWorktree) {
    $DevelopmentWorktree = $defaultDevelopmentWorktree
}
if (-not $WorkflowConfig) {
    $WorkflowConfig = Join-Path $DevelopmentWorktree ".runtime\development-workflow.json"
}
$workflow = $null
if (Test-Path -LiteralPath $WorkflowConfig -PathType Leaf) {
    try {
        $workflow = Get-Content -LiteralPath $WorkflowConfig -Raw | ConvertFrom-Json
    }
    catch {
        throw "Development workflow configuration is invalid: $WorkflowConfig"
    }
    if (-not ($workflow.PSObject.Properties.Name -contains "schema_version") -or
        [int]$workflow.schema_version -ne 1) {
        throw "Development workflow configuration must use schema_version 1."
    }
}

if (-not $DevelopmentBranch) {
    $DevelopmentBranch = if ($workflow -and
        $workflow.PSObject.Properties.Name -contains "development_branch") {
        [string]$workflow.development_branch
    }
    else { "development" }
}
if (-not $ProductionBranch) {
    $ProductionBranch = if ($workflow -and
        $workflow.PSObject.Properties.Name -contains "production_branch") {
        [string]$workflow.production_branch
    }
    else { "main" }
}
if (-not $Remote) {
    $Remote = if ($workflow -and
        $workflow.PSObject.Properties.Name -contains "remote") {
        [string]$workflow.remote
    }
    else { "origin" }
}
if (-not $ExpectedOriginUrl) {
    $ExpectedOriginUrl = if ($workflow -and
        $workflow.PSObject.Properties.Name -contains "expected_origin_url") {
        [string]$workflow.expected_origin_url
    }
    else { "https://github.com/AngusChik/FINAL-PHARM.git" }
}
if (-not $ProductionWorktree) {
    if ($workflow -and
        $workflow.PSObject.Properties.Name -contains "production_worktree" -and
        [string]$workflow.production_worktree) {
        $ProductionWorktree = [string]$workflow.production_worktree
    }
    elseif ($env:PHARMACY_PRODUCTION_WORKTREE) {
        $ProductionWorktree = $env:PHARMACY_PRODUCTION_WORKTREE
    }
    else {
        $repositoryParent = Split-Path $defaultDevelopmentWorktree -Parent
        $ProductionWorktree = Join-Path $repositoryParent "FINAL-PHARM-PRODUCTION"
    }
}
if (-not $StateDirectory) {
    $StateDirectory = Join-Path $DevelopmentWorktree ".runtime\release-engine"
}
if ($DevelopmentBranch -cne "development" -or
    $ProductionBranch -cne "main" -or
    $Remote -cne "origin") {
    throw (
        "The guarded publisher requires local-only development and the " +
        "production origin/main branch."
    )
}
$normalizedDevelopmentRoot = [IO.Path]::GetFullPath(
    $DevelopmentWorktree
).TrimEnd('\', '/')
$normalizedProductionRoot = [IO.Path]::GetFullPath(
    $ProductionWorktree
).TrimEnd('\', '/')
if ($normalizedDevelopmentRoot -ieq $normalizedProductionRoot -or
    (Split-Path $normalizedDevelopmentRoot -Parent) -ine
    (Split-Path $normalizedProductionRoot -Parent)) {
    throw "Development and production must be distinct sibling worktrees."
}
$normalizedStateDirectory = [IO.Path]::GetFullPath(
    $StateDirectory
).TrimEnd('\', '/')
if (-not $normalizedStateDirectory.StartsWith(
    $normalizedDevelopmentRoot + '\',
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "Release state must stay inside the development worktree runtime directory."
}

$pendingPath = Join-Path $StateDirectory "sync-pending.json"
$releasesDirectory = Join-Path $StateDirectory "releases"
$productionScriptRelativePath = "scripts\production.ps1"
$script:activeProductionReleaseToken = ""
$expectedHealthyStatus = @(
    "Waitress: running",
    "Caddy:    running",
    "Django/DB: healthy",
    "HTTPS:     healthy"
)

function Write-ReleaseMessage([string]$Message, [string]$Color = "Gray") {
    Write-Host "[release] $Message" -ForegroundColor $Color
}

function ConvertTo-NormalizedPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function ConvertTo-NormalizedRemote([string]$Url) {
    $normalized = $Url.Trim().TrimEnd('/')
    if ($normalized -match '^git@github\.com:(.+)$') {
        $normalized = "https://github.com/$($Matches[1])"
    }
    if ($normalized.EndsWith(".git", [StringComparison]::OrdinalIgnoreCase)) {
        $normalized = $normalized.Substring(0, $normalized.Length - 4)
    }
    return $normalized.ToLowerInvariant()
}

function ConvertTo-CommandDisplay(
    [string]$FilePath,
    [string[]]$Arguments
) {
    $displayArguments = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            '"' + $argument.Replace('"', '\"') + '"'
        }
        else { $argument }
    }
    $parts = @($FilePath) + @($displayArguments)
    return ($parts -join " ").Trim()
}

function New-ProcessResult([int]$ExitCode, [object]$Output, [bool]$Skipped = $false) {
    $text = (@($Output) | ForEach-Object { [string]$_ }) -join "`n"
    return [pscustomobject]@{
        ExitCode = $ExitCode
        Output = $text.TrimEnd()
        Skipped = $Skipped
    }
}

function Invoke-ReleaseProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [switch]$Mutation,
        [switch]$AllowFailure
    )

    $display = ConvertTo-CommandDisplay $FilePath $Arguments
    if ($Mutation -and $DryRun) {
        Write-ReleaseMessage "DRY RUN: $display" "DarkCyan"
        return New-ProcessResult 0 @() $true
    }

    Write-ReleaseMessage $display "DarkGray"
    if ($CommandInvoker) {
        $injected = & $CommandInvoker `
            $FilePath ([string[]]$Arguments) $WorkingDirectory ([bool]$Mutation)
        if ($null -eq $injected) {
            $result = New-ProcessResult 0 @()
        }
        elseif ($injected.PSObject.Properties.Name -contains "ExitCode") {
            $injectedOutput = if ($injected.PSObject.Properties.Name -contains "Output") {
                $injected.Output
            }
            else { @() }
            $result = New-ProcessResult ([int]$injected.ExitCode) $injectedOutput
        }
        else {
            $result = New-ProcessResult 0 $injected
        }
    }
    else {
        Push-Location $WorkingDirectory
        $previousErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $nativeOutput = @(& $FilePath @Arguments 2>&1)
            $exitCode = $LASTEXITCODE
            $result = New-ProcessResult $exitCode $nativeOutput
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
            Pop-Location
        }
    }

    if ($result.ExitCode -ne 0 -and -not $AllowFailure) {
        $detail = if ($result.Output) { "`n$($result.Output)" } else { "" }
        throw "Command failed ($($result.ExitCode)): $display$detail"
    }
    return $result
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)][string]$Worktree,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$Mutation,
        [switch]$AllowFailure
    )
    return Invoke-ReleaseProcess -FilePath "git.exe" -Arguments $Arguments `
        -WorkingDirectory $Worktree -Mutation:$Mutation -AllowFailure:$AllowFailure
}

function Get-GitValue([string]$Worktree, [string[]]$Arguments) {
    $result = Invoke-Git -Worktree $Worktree -Arguments $Arguments
    return $result.Output.Trim()
}

function Assert-ExistingDirectory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
}

function Assert-ExistingFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
}

function Resolve-GitInternalPath([string]$Worktree, [string]$Name) {
    $value = Get-GitValue $Worktree @("rev-parse", "--git-path", $Name)
    if ([IO.Path]::IsPathRooted($value)) { return $value }
    return [IO.Path]::GetFullPath((Join-Path $Worktree $value))
}

function Resolve-GitCommonDirectory([string]$Worktree) {
    $value = Get-GitValue $Worktree @("rev-parse", "--git-common-dir")
    if ([IO.Path]::IsPathRooted($value)) {
        return ConvertTo-NormalizedPath $value
    }
    return ConvertTo-NormalizedPath (Join-Path $Worktree $value)
}

function Assert-NoGitOperation([string]$Worktree) {
    foreach ($marker in @("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply")) {
        $markerPath = Resolve-GitInternalPath $Worktree $marker
        if (Test-Path -LiteralPath $markerPath) {
            throw "An unfinished Git operation exists in $Worktree ($marker)."
        }
    }
}

function Get-CleanBranchSnapshot(
    [string]$Worktree,
    [string]$ExpectedBranch,
    [string]$Label
) {
    Assert-ExistingDirectory $Worktree "$Label worktree"
    $resolvedWorktree = (Resolve-Path -LiteralPath $Worktree).Path
    $reportedRoot = Get-GitValue $resolvedWorktree @("rev-parse", "--show-toplevel")
    if ((ConvertTo-NormalizedPath $reportedRoot) -ne (ConvertTo-NormalizedPath $resolvedWorktree)) {
        throw "$Label path is not the root of its Git worktree: $resolvedWorktree"
    }

    Assert-NoGitOperation $resolvedWorktree
    $branch = Get-GitValue $resolvedWorktree @("symbolic-ref", "--quiet", "--short", "HEAD")
    if ($branch -ne $ExpectedBranch) {
        throw "$Label must be on '$ExpectedBranch'; found '$branch'."
    }

    $statusResult = Invoke-Git -Worktree $resolvedWorktree -Arguments @(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    # Git may emit a successful stderr warning for an unreadable ignored/cache
    # directory. Only porcelain records represent tracked or untracked changes.
    $statusRecords = @(
        $statusResult.Output -split "`r?`n" |
            Where-Object { $_ -match '^[ MTADRCU?!]{2} ' }
    )
    if ($statusRecords.Count -gt 0) {
        $status = $statusRecords -join "`n"
        throw "$Label worktree must be clean before release.`n$status"
    }

    $head = Get-GitValue $resolvedWorktree @("rev-parse", "HEAD")
    $branchHead = Get-GitValue $resolvedWorktree @(
        "rev-parse", "refs/heads/$ExpectedBranch"
    )
    if ($head -ne $branchHead) {
        throw "$Label HEAD is detached from refs/heads/$ExpectedBranch."
    }

    return [pscustomobject]@{
        Path = $resolvedWorktree
        Branch = $branch
        Commit = $head
        CommonGitDirectory = Resolve-GitCommonDirectory $resolvedWorktree
    }
}

function Assert-ExpectedRemote([string]$Worktree) {
    $actual = Get-GitValue $Worktree @("remote", "get-url", $Remote)
    if ((ConvertTo-NormalizedRemote $actual) -ne
        (ConvertTo-NormalizedRemote $ExpectedOriginUrl)) {
        throw "Remote '$Remote' is '$actual', expected '$ExpectedOriginUrl'."
    }
    return $actual
}

function Assert-DevelopmentIsLocalOnly([object]$Development) {
    $upstream = Invoke-Git -Worktree $Development.Path -AllowFailure -Arguments @(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    if ($upstream.ExitCode -eq 0 -and $upstream.Output.Trim()) {
        throw (
            "Development branch '$($Development.Branch)' must remain local-only, " +
            "but it tracks '$($upstream.Output.Trim())'. Remove its upstream before release."
        )
    }
}

function Update-ReleaseRefs([string]$Worktree) {
    $arguments = @(
        "fetch", "--prune", $Remote,
        "+refs/heads/${ProductionBranch}:refs/remotes/${Remote}/${ProductionBranch}"
    )
    Invoke-Git -Worktree $Worktree -Arguments $arguments -Mutation | Out-Null
}

function ConvertFrom-LeftRightCount([string]$Text, [string]$Label) {
    $parts = @($Text.Trim() -split '\s+')
    if ($parts.Count -ne 2) {
        throw "Could not parse Git divergence for ${Label}: '$Text'."
    }
    return [pscustomobject]@{
        Left = [int]$parts[0]
        Right = [int]$parts[1]
    }
}

function Assert-ReleaseRelationship(
    [object]$Development,
    [object]$Production
) {
    if ($Development.CommonGitDirectory -ne $Production.CommonGitDirectory) {
        throw "Development and production must be registered worktrees of the same Git repository."
    }

    $remoteProductionRef = "refs/remotes/${Remote}/${ProductionBranch}"
    $remoteProduction = Get-GitValue $Development.Path @("rev-parse", $remoteProductionRef)

    $ancestor = Invoke-Git -Worktree $Development.Path -Arguments @(
        "merge-base", "--is-ancestor", $remoteProductionRef, $Development.Commit
    ) -AllowFailure
    if ($ancestor.ExitCode -ne 0) {
        throw "Development does not contain the current $Remote/$ProductionBranch. Sync it first."
    }

    $aheadText = Get-GitValue $Development.Path @(
        "rev-list", "--count", "${Remote}/${ProductionBranch}..$($Development.Commit)"
    )
    $ahead = [int]$aheadText
    if ($ahead -lt 1) {
        throw "Development has no new commits to publish."
    }

    $productionDivergenceText = Get-GitValue $Production.Path @(
        "rev-list", "--left-right", "--count",
        "${Remote}/${ProductionBranch}...HEAD"
    )
    $productionDivergence = ConvertFrom-LeftRightCount `
        $productionDivergenceText "production"
    if ($productionDivergence.Left -ne 0 -or $productionDivergence.Right -ne 0 -or
        $Production.Commit -ne $remoteProduction) {
        throw (
            "Production must exactly match $Remote/$ProductionBranch before release. " +
            "Remote-only commits: $($productionDivergence.Left); " +
            "local-only commits: $($productionDivergence.Right)."
        )
    }

    return [pscustomobject]@{
        DevelopmentCommit = $Development.Commit
        ProductionCommit = $remoteProduction
        CommitsAhead = $ahead
        DevelopmentIsLocalOnly = $true
    }
}

function Assert-ReleasePrerequisites([object]$Development, [object]$Production) {
    Assert-ExistingFile (Join-Path $Development.Path "manage.py") "Development manage.py"
    Assert-ExistingFile (Join-Path $Development.Path "env\Scripts\python.exe") `
        "Development Python"
    Assert-ExistingFile (Join-Path $Production.Path "manage.py") "Production manage.py"
    Assert-ExistingFile (Join-Path $Production.Path "env\Scripts\python.exe") `
        "Production Python"
    Assert-ExistingFile (Join-Path $Production.Path $productionScriptRelativePath) `
        "Production control script"
    Assert-ExistingFile (Join-Path $Production.Path ".env") "Production environment file"
    $rolePath = Join-Path $Production.Path ".runtime\production-role.json"
    Assert-ExistingFile $rolePath "Production role marker"
    try {
        $role = Get-Content -LiteralPath $rolePath -Raw | ConvertFrom-Json
        if ([int]$role.schema_version -ne 1 -or
            [string]$role.role -cne "production" -or
            [string]$role.branch -cne $ProductionBranch -or
            (ConvertTo-NormalizedPath ([string]$role.worktree)) -ne
            (ConvertTo-NormalizedPath $Production.Path)) {
            throw "marker fields do not match this production worktree"
        }
    }
    catch {
        throw "Production role marker is invalid: $($_.Exception.Message)"
    }
    $recoveryRequiredPath = Join-Path `
        $Production.Path ".runtime\production-recovery-required.json"
    if (Test-Path -LiteralPath $recoveryRequiredPath -PathType Leaf) {
        throw (
            "Production is blocked for manual recovery. Use Pharmacy Admin " +
            "Control to inspect and clear the recovery block before another release."
        )
    }
}

function Invoke-CandidateChecks([object]$Development, [object]$Production) {
    $python = Join-Path $Development.Path "env\Scripts\python.exe"
    $productionPython = Join-Path $Production.Path "env\Scripts\python.exe"
    Write-ReleaseMessage "Running development configuration check..." "Cyan"
    Invoke-ReleaseProcess -FilePath $python -WorkingDirectory $Development.Path `
        -Arguments @("manage.py", "check", "--settings=inventory.settings_development") |
        Out-Null
    Write-ReleaseMessage "Checking for missing Django migrations..." "Cyan"
    Invoke-ReleaseProcess -FilePath $python -WorkingDirectory $Development.Path `
        -Arguments @(
            "manage.py", "makemigrations", "--check", "--dry-run",
            "--settings=inventory.settings_development"
        ) | Out-Null

    Write-ReleaseMessage "Running candidate code with production settings..." "Cyan"
    $previousProductionEnvironmentFile = [Environment]::GetEnvironmentVariable(
        "PHARMACY_PRODUCTION_ENV_FILE", "Process"
    )
    $previousProductionRoleRoot = [Environment]::GetEnvironmentVariable(
        "PHARMACY_PRODUCTION_ROLE_ROOT", "Process"
    )
    try {
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_PRODUCTION_ENV_FILE",
            (Join-Path $Production.Path ".env"),
            "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_PRODUCTION_ROLE_ROOT",
            $Production.Path,
            "Process"
        )
        Invoke-ReleaseProcess -FilePath $productionPython -WorkingDirectory $Development.Path `
            -Arguments @(
                "manage.py", "check", "--deploy",
                "--settings=inventory.settings_production"
            ) | Out-Null
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_PRODUCTION_ENV_FILE",
            $previousProductionEnvironmentFile,
            "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "PHARMACY_PRODUCTION_ROLE_ROOT",
            $previousProductionRoleRoot,
            "Process"
        )
    }

    # This verifies the real production environment without changing it. The
    # candidate itself is checked again by production.ps1 after the guarded
    # fast-forward, before Waitress or Caddy are reported healthy.
    Write-ReleaseMessage "Running the production environment deployment check..." "Cyan"
    Invoke-ReleaseProcess -FilePath $productionPython -WorkingDirectory $Production.Path `
        -Arguments @("manage.py", "check", "--deploy", "--settings=inventory.settings_production") |
        Out-Null

    Write-ReleaseMessage "Running the application test suite..." "Cyan"
    Invoke-ReleaseProcess -FilePath $python -WorkingDirectory $Development.Path `
        -Arguments @(
            "manage.py", "test",
            "--settings=inventory.settings_development",
            "--keepdb", "--noinput"
        ) | Out-Null

    foreach ($powerShellTest in @(
        "scripts\tests\test-automation-task-scripts.ps1",
        "scripts\tests\test-publish-release.ps1"
    )) {
        $testPath = Join-Path $Development.Path $powerShellTest
        Assert-ExistingFile $testPath "PowerShell release contract test"
        Write-ReleaseMessage "Running $powerShellTest..." "Cyan"
        Invoke-ReleaseProcess -FilePath "powershell.exe" `
            -WorkingDirectory $Development.Path `
            -Arguments @(
                "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", $testPath
            ) | Out-Null
    }
}

function Get-ReleasePreflight([switch]$Fetch, [switch]$RunChecks) {
    if (Test-Path -LiteralPath $pendingPath) {
        throw "A previous release is sync-pending. Run this script with -Action status."
    }

    $development = Get-CleanBranchSnapshot `
        $DevelopmentWorktree $DevelopmentBranch "Development"
    Assert-DevelopmentIsLocalOnly $development
    $production = Get-CleanBranchSnapshot `
        $ProductionWorktree $ProductionBranch "Production"
    $originUrl = Assert-ExpectedRemote $development.Path
    $productionOriginUrl = Assert-ExpectedRemote $production.Path
    if ((ConvertTo-NormalizedRemote $originUrl) -ne
        (ConvertTo-NormalizedRemote $productionOriginUrl)) {
        throw "Development and production remotes do not match."
    }

    if ($Fetch) {
        Update-ReleaseRefs $development.Path
        # Re-read snapshots because fetch may expose a new remote state.
        $development = Get-CleanBranchSnapshot `
            $DevelopmentWorktree $DevelopmentBranch "Development"
        $production = Get-CleanBranchSnapshot `
            $ProductionWorktree $ProductionBranch "Production"
    }

    $relationship = Assert-ReleaseRelationship $development $production
    Assert-ReleasePrerequisites $development $production
    if ($RunChecks) { Invoke-CandidateChecks $development $production }

    return [pscustomobject]@{
        Development = $development
        Production = $production
        Relationship = $relationship
        OriginUrl = $originUrl
    }
}

function Get-UtcNow {
    if ($Clock) {
        return [DateTimeOffset](& $Clock)
    }
    return [DateTimeOffset]::UtcNow
}

function Write-JsonAtomic([string]$Path, [object]$Value) {
    if ($DryRun) {
        Write-ReleaseMessage "DRY RUN: write $Path" "DarkCyan"
        return
    }
    $directory = Split-Path $Path -Parent
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $temporary = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"
    $json = $Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $temporary,
        $json + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Remove-StateFile([string]$Path) {
    if ($DryRun) {
        Write-ReleaseMessage "DRY RUN: remove $Path" "DarkCyan"
        return
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Set-ProductionRecoveryBlock(
    [object]$Production,
    [object]$Artifacts,
    [string]$Failure,
    [object]$Rollback
) {
    $path = Join-Path `
        $Production.Path ".runtime\production-recovery-required.json"
    Write-JsonAtomic $path ([pscustomobject][ordered]@{
        schema_version = 1
        release_id = $Artifacts.Id
        failed_release_commit = $Artifacts.Manifest.source_commit
        previous_production_commit = $Artifacts.Manifest.previous_production_commit
        production_backup_path = $Artifacts.Manifest.production_backup_path
        failure = $Failure
        rollback = $Rollback
        created_utc = (Get-UtcNow).ToString("o")
    })
}

function Clear-ProductionRecoveryBlock([object]$Production) {
    Remove-StateFile (Join-Path `
        $Production.Path ".runtime\production-recovery-required.json")
}

function Enter-PublishLock {
    if ($DryRun) { return $null }
    New-Item -ItemType Directory -Force -Path $StateDirectory | Out-Null
    $lockPath = Join-Path $StateDirectory "publish.lock"
    try {
        $stream = [IO.File]::Open(
            $lockPath,
            [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    }
    catch {
        throw "Another publish operation is already running."
    }

    try {
        $stream.SetLength(0)
        $lockText = "pid=$PID started_utc=$((Get-UtcNow).ToString('o'))"
        $bytes = [Text.Encoding]::UTF8.GetBytes($lockText)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()
        return $stream
    }
    catch {
        $stream.Dispose()
        throw
    }
}

function Enter-ProductionReleaseLock([object]$Production) {
    if ($DryRun) { return $null }
    $runtimeDirectory = Join-Path $Production.Path ".runtime"
    New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
    $lockPath = Join-Path $runtimeDirectory "production-release.lock"
    $ownerPath = Join-Path $runtimeDirectory "production-release.owner.json"
    $token = [Guid]::NewGuid().ToString("N")
    try {
        $stream = [IO.File]::Open(
            $lockPath,
            [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    }
    catch {
        throw (
            "Production is busy with another startup, control, or release operation. " +
            "Wait for it to finish, then retry."
        )
    }

    try {
        $stream.SetLength(0)
        $startedUtc = (Get-UtcNow).ToString("o")
        $lockText = "release_pid=$PID started_utc=$startedUtc"
        $bytes = [Text.Encoding]::UTF8.GetBytes($lockText)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()
        Write-JsonAtomic $ownerPath ([pscustomobject][ordered]@{
            schema_version = 1
            release_token = $token
            process_id = $PID
            started_utc = $startedUtc
        })
        return [pscustomobject]@{
            Stream = $stream
            Token = $token
            OwnerPath = $ownerPath
        }
    }
    catch {
        $stream.Dispose()
        throw
    }
}

function Exit-ProductionReleaseLock([object]$Lock) {
    $script:activeProductionReleaseToken = ""
    if (-not $Lock) { return }
    try {
        Remove-StateFile $Lock.OwnerPath
    }
    catch {
        Write-ReleaseMessage (
            "Could not remove stale release-owner metadata; the OS lock " +
            "was still released safely. $($_.Exception.Message)"
        ) "Yellow"
    }
    finally {
        $Lock.Stream.Dispose()
    }
}

function Confirm-Publish([string]$Commit, [string]$Mode = "PUBLISH") {
    if ($ConfirmRelease -or $DryRun) { return }
    $shortCommit = $Commit.Substring(0, [Math]::Min(12, $Commit.Length))
    $expected = "$Mode $shortCommit"
    Write-Host ""
    Write-ReleaseMessage "This will change the live production worktree." "Yellow"
    $answer = Read-Host "Type '$expected' to continue"
    if ($answer -cne $expected) {
        throw "Release confirmation did not match; nothing was published."
    }
}

function New-ReleaseArtifacts([object]$Preflight) {
    $now = Get-UtcNow
    $commit = $Preflight.Relationship.DevelopmentCommit
    $shortCommit = $commit.Substring(0, [Math]::Min(12, $commit.Length))
    $releaseId = "pharmacy-release-$($now.ToString('yyyyMMddTHHmmssZ'))-$shortCommit"
    $releaseDirectory = Join-Path $releasesDirectory $releaseId
    $manifestPath = Join-Path $releaseDirectory "manifest.json"
    $bundlePath = Join-Path $releaseDirectory "$releaseId.bundle"
    $partialBundlePath = "$bundlePath.partial"

    $tagCheck = Invoke-Git -Worktree $Preflight.Development.Path -Arguments @(
        "show-ref", "--verify", "--quiet", "refs/tags/$releaseId"
    ) -AllowFailure
    if ($tagCheck.ExitCode -eq 0) {
        throw "Release tag already exists: $releaseId"
    }

    $manifest = [pscustomobject][ordered]@{
        schema_version = 1
        release_id = $releaseId
        release_tag = $releaseId
        created_utc = $now.ToString("o")
        source_branch = $DevelopmentBranch
        source_commit = $commit
        production_branch = $ProductionBranch
        previous_production_commit = $Preflight.Relationship.ProductionCommit
        remote = $Remote
        remote_url = $Preflight.OriginUrl
        commits_ahead = $Preflight.Relationship.CommitsAhead
        development_worktree = $Preflight.Development.Path
        production_worktree = $Preflight.Production.Path
        bundle_path = $bundlePath
        bundle_sha256 = $null
        production_backup_path = $null
        checks = [ordered]@{
            repository = "passed"
            candidate = "passed"
            backup = "pending"
            production_health = "pending"
        }
        deployment = [ordered]@{
            status = "not_started"
            completed_utc = $null
            rollback = $null
        }
        synchronization = [ordered]@{
            status = "not_started"
            completed_utc = $null
            error = $null
        }
    }

    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null
    }
    Write-JsonAtomic $manifestPath $manifest

    Invoke-Git -Worktree $Preflight.Development.Path -Mutation -Arguments @(
        "tag", "--annotate", $releaseId, $commit,
        "--message", "Pharmacy production release $releaseId"
    ) | Out-Null

    Invoke-Git -Worktree $Preflight.Development.Path -Mutation -Arguments @(
        "bundle", "create", $partialBundlePath,
        "refs/tags/$releaseId",
        "refs/remotes/${Remote}/${ProductionBranch}"
    ) | Out-Null

    if (-not $DryRun) {
        Invoke-Git -Worktree $Preflight.Development.Path -Arguments @(
            "bundle", "verify", $partialBundlePath
        ) | Out-Null
        Move-Item -LiteralPath $partialBundlePath -Destination $bundlePath -Force
        $manifest.bundle_sha256 = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
        Write-JsonAtomic $manifestPath $manifest
    }

    return [pscustomobject]@{
        Id = $releaseId
        Tag = $releaseId
        Directory = $releaseDirectory
        ManifestPath = $manifestPath
        Manifest = $manifest
    }
}

function Invoke-ProductionControl(
    [object]$Production,
    [string]$ProductionAction,
    [switch]$AllowFailure
) {
    $scriptPath = Join-Path $Production.Path $productionScriptRelativePath
    $arguments = @(
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", $scriptPath, "-Action", $ProductionAction, "-NoBrowser"
    )
    # The first release is controlled by the older production script until the
    # worktree advances. Host-level non-interactive mode is universal; only pass
    # the newer script switch when that checked-out controller declares it.
    $controllerSource = Get-Content -LiteralPath $scriptPath -Raw
    if ($controllerSource -match '(?m)\[switch\]\s*\$NonInteractive\b') {
        $arguments += "-NonInteractive"
    }
    if ($script:activeProductionReleaseToken -and
        $controllerSource -match '(?m)\$ReleaseToken\b') {
        $arguments += @("-ReleaseToken", $script:activeProductionReleaseToken)
    }
    $changesState = $ProductionAction -ne "status"
    return Invoke-ReleaseProcess -FilePath "powershell.exe" -Arguments $arguments `
        -WorkingDirectory $Production.Path -Mutation:$changesState `
        -AllowFailure:$AllowFailure
}

function Get-VerifiedBackupPath([object]$BackupResult) {
    $candidates = @(
        $BackupResult.Output -split "`r?`n" |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -match '(?i)\.dump$' }
    )
    if ($candidates.Count -eq 0) {
        throw "Production backup completed without reporting its verified dump path."
    }
    $path = $candidates[-1]
    if (-not $CommandInvoker) {
        Assert-ExistingFile $path "Verified production database backup"
        Assert-ExistingFile "$path.sha256" "Verified production backup checksum"
    }
    return $path
}

function Get-DotEnvValue([string]$Path, [string]$Name, [string]$Default = "") {
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or
            -not $trimmed.Contains("=")) { continue }
        $parts = $trimmed.Split("=", 2)
        if ($parts[0].Trim() -eq $Name) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }
    return $Default
}

function Restore-ProductionDatabase(
    [object]$Production,
    [string]$BackupPath
) {
    if (-not $BackupPath) {
        return New-ProcessResult 1 "No verified pre-release backup was recorded."
    }
    $restoreScript = Join-Path $Production.Path "scripts\database-restore.ps1"
    Assert-ExistingFile $restoreScript "Production database restore script"
    $databaseName = Get-DotEnvValue (Join-Path $Production.Path ".env") "DB_NAME" "postgres"
    return Invoke-ReleaseProcess -FilePath "powershell.exe" -Mutation -AllowFailure `
        -WorkingDirectory $Production.Path -Arguments @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", $restoreScript,
            "-BackupPath", $BackupPath,
            "-ConfirmDatabaseName", $databaseName
        )
}

function Assert-ProductionHealthy([object]$Production) {
    $status = Invoke-ProductionControl $Production "status"
    foreach ($expected in $expectedHealthyStatus) {
        if ($status.Output -notmatch [regex]::Escape($expected)) {
            throw "Production health check did not report '$expected'.`n$($status.Output)"
        }
    }
}

function Invoke-BestEffortRollback(
    [object]$Production,
    [string]$PreviousCommit,
    [string]$BackupPath,
    [bool]$DatabaseRestoreRequired
) {
    $notes = New-Object Collections.Generic.List[string]
    $safeToRestart = $true
    $stopSucceeded = $false
    $rollbackHealthy = $false
    try {
        $stopResult = Invoke-ProductionControl $Production "stop" -AllowFailure
        if ($stopResult.ExitCode -eq 0) {
            $stopSucceeded = $true
            $notes.Add("production stopped")
        }
        else {
            $notes.Add("stop failed: $($stopResult.Output)")
        }
    }
    catch { $notes.Add("stop failed: $($_.Exception.Message)") }

    if (-not $stopSucceeded) {
        $notes.Add("rollback aborted before code or database changes because production could not be stopped")
        return [pscustomobject][ordered]@{
            status = "failed"
            database_restore_required = $DatabaseRestoreRequired
            completed_utc = (Get-UtcNow).ToString("o")
            notes = [string[]]$notes
        }
    }

    try {
        Invoke-Git -Worktree $Production.Path -Mutation -Arguments @(
            "reset", "--hard", $PreviousCommit
        ) | Out-Null
        $notes.Add("code reset to $PreviousCommit")
    }
    catch {
        $safeToRestart = $false
        $notes.Add("code reset failed: $($_.Exception.Message)")
    }

    if ($DatabaseRestoreRequired) {
        try {
            $restore = Restore-ProductionDatabase $Production $BackupPath
            if ($restore.ExitCode -eq 0) {
                $notes.Add("database restored from $BackupPath")
            }
            else {
                $safeToRestart = $false
                $notes.Add("database restore failed: $($restore.Output)")
            }
        }
        catch {
            $safeToRestart = $false
            $notes.Add("database restore failed: $($_.Exception.Message)")
        }
    }
    else {
        $notes.Add("database restore not required; deployment startup was not attempted")
    }

    if ($safeToRestart) {
        try {
            $startResult = Invoke-ProductionControl $Production "start" -AllowFailure
            if ($startResult.ExitCode -ne 0) {
                $notes.Add("previous production start failed: $($startResult.Output)")
            }
            else {
                Assert-ProductionHealthy $Production
                $notes.Add("previous production restored and healthy")
                $rollbackHealthy = $true
            }
        }
        catch { $notes.Add("previous production start failed: $($_.Exception.Message)") }
    }
    else {
        $notes.Add("production left stopped because rollback was incomplete")
    }

    return [pscustomobject][ordered]@{
        status = if ($rollbackHealthy) { "healthy" } else { "failed" }
        database_restore_required = $DatabaseRestoreRequired
        completed_utc = (Get-UtcNow).ToString("o")
        notes = [string[]]$notes
    }
}

function Save-SyncPending([object]$Artifacts, [string]$ErrorMessage) {
    $manifest = $Artifacts.Manifest
    $manifest.synchronization.status = "pending"
    $manifest.synchronization.error = $ErrorMessage
    Write-JsonAtomic $Artifacts.ManifestPath $manifest

    $pending = [pscustomobject][ordered]@{
        schema_version = 1
        release_id = $Artifacts.Id
        tag = $Artifacts.Tag
        commit = $manifest.source_commit
        production_branch = $ProductionBranch
        remote = $Remote
        production_worktree = $manifest.production_worktree
        manifest_path = $Artifacts.ManifestPath
        created_utc = (Get-UtcNow).ToString("o")
        last_error = $ErrorMessage
    }
    Write-JsonAtomic $pendingPath $pending
}

function Push-Release([object]$Preflight, [object]$Artifacts) {
    $push = Invoke-Git -Worktree $Preflight.Production.Path -Mutation -AllowFailure `
        -Arguments @(
            "push", "--atomic", $Remote,
            "refs/heads/${ProductionBranch}:refs/heads/${ProductionBranch}",
            "refs/tags/$($Artifacts.Tag):refs/tags/$($Artifacts.Tag)"
        )
    if ($push.ExitCode -ne 0) {
        $detail = if ($push.Output) { $push.Output } else { "Git push failed." }
        Save-SyncPending $Artifacts $detail
        throw (
            "Production is healthy, but Git synchronization is pending. " +
            "Run -Action status for recovery details.`n$detail"
        )
    }

    $Artifacts.Manifest.synchronization.status = "complete"
    $Artifacts.Manifest.synchronization.completed_utc = (Get-UtcNow).ToString("o")
    $Artifacts.Manifest.synchronization.error = $null
    Write-JsonAtomic $Artifacts.ManifestPath $Artifacts.Manifest
    Remove-StateFile $pendingPath
}

function Publish-NewRelease([object]$Preflight) {
    Confirm-Publish $Preflight.Relationship.DevelopmentCommit
    if ($DryRun) {
        $shortCommit = $Preflight.Relationship.DevelopmentCommit.Substring(0, 12)
        Write-ReleaseMessage (
            "DRY RUN complete: would publish $shortCommit to " +
            "$($Preflight.Production.Path), verify health, then push atomically."
        ) "Green"
        return
    }

    # Full tests can take several minutes and confirmation can add an arbitrary
    # pause. Refresh both remote refs and all clean-worktree invariants before
    # creating a tag, stopping production, or changing main.
    $checkedDevelopmentCommit = $Preflight.Relationship.DevelopmentCommit
    $checkedProductionCommit = $Preflight.Relationship.ProductionCommit
    $Preflight = Get-ReleasePreflight -Fetch
    if ($Preflight.Relationship.DevelopmentCommit -ne $checkedDevelopmentCommit -or
        $Preflight.Relationship.ProductionCommit -ne $checkedProductionCommit) {
        throw "Release inputs changed after validation; run -Action check again."
    }

    $artifacts = New-ReleaseArtifacts $Preflight
    $manifest = $artifacts.Manifest
    $mainAdvanced = $false
    $productionStopAttempted = $false
    $productionHealthy = $false
    $deploymentStartAttempted = $false
    $backupPath = ""
    $productionReleaseLock = Enter-ProductionReleaseLock $Preflight.Production
    $script:activeProductionReleaseToken = $productionReleaseLock.Token

    try {
        try {
            $lockedProduction = Get-CleanBranchSnapshot `
                $Preflight.Production.Path $ProductionBranch "Production"
            if ($lockedProduction.Commit -ne $Preflight.Relationship.ProductionCommit) {
                throw "Production HEAD changed after release validation; no deployment was attempted."
            }

            # This durable journal is written before the first outage. If this
            # PowerShell process or the machine dies mid-release, scheduled
            # ensure cannot start a partially migrated or mixed code/database
            # state. Only the currently held release token may pass the gate.
            Set-ProductionRecoveryBlock `
                $Preflight.Production $artifacts `
                "Release deployment is in progress." `
                ([pscustomobject]@{
                    status = "not_started"
                    database_restore_required = $false
                    completed_utc = $null
                    notes = @("Automatic startup is blocked until release verification or rollback completes.")
                })

            Write-ReleaseMessage "Stopping production before the final release backup..." "Cyan"
            $productionStopAttempted = $true
            Invoke-ProductionControl $Preflight.Production "stop" | Out-Null

            Write-ReleaseMessage "Creating the verified final production database backup..." "Cyan"
            $backupResult = Invoke-ProductionControl $Preflight.Production "backup"
            $backupPath = Get-VerifiedBackupPath $backupResult
            $manifest.checks.backup = "passed"
            $manifest.production_backup_path = $backupPath
            Write-JsonAtomic $artifacts.ManifestPath $manifest
            Set-ProductionRecoveryBlock `
                $Preflight.Production $artifacts `
                "Release deployment is in progress after the final backup." `
                ([pscustomobject]@{
                    status = "backup_verified"
                    database_restore_required = $false
                    completed_utc = $null
                    notes = @("Verified rollback backup: $backupPath")
                })

            $preMergeProduction = Get-CleanBranchSnapshot `
                $Preflight.Production.Path $ProductionBranch "Production"
            if ($preMergeProduction.Commit -ne $Preflight.Relationship.ProductionCommit) {
                throw "Production worktree changed while the final backup was running."
            }

            Write-ReleaseMessage "Fast-forwarding the production worktree to the release commit..." "Cyan"
            Invoke-Git -Worktree $Preflight.Production.Path -Mutation -Arguments @(
                "merge", "--ff-only", $Preflight.Relationship.DevelopmentCommit
            ) | Out-Null
            $mainAdvanced = $true
            $deployedProduction = Get-CleanBranchSnapshot `
                $Preflight.Production.Path $ProductionBranch "Production"
            if ($deployedProduction.Commit -ne $Preflight.Relationship.DevelopmentCommit) {
                throw "Production worktree did not land on the exact tested release commit."
            }
            $manifest.deployment.status = "starting"
            Write-JsonAtomic $artifacts.ManifestPath $manifest

            Write-ReleaseMessage "Starting production with deployment checks and migrations..." "Cyan"
            $deploymentStartAttempted = $true
            Invoke-ProductionControl $Preflight.Production "start" | Out-Null
            Assert-ProductionHealthy $Preflight.Production
            $productionHealthy = $true
            $manifest.checks.production_health = "passed"
            $manifest.deployment.status = "healthy"
            $manifest.deployment.completed_utc = (Get-UtcNow).ToString("o")
            Write-JsonAtomic $artifacts.ManifestPath $manifest

        }
        catch {
            $failure = $_.Exception.Message
            $manifest.deployment.status = "failed"
            if (-not $backupPath) { $manifest.checks.backup = "failed" }
            if ($deploymentStartAttempted) {
                $manifest.checks.production_health = "failed"
            }
            if (($mainAdvanced -or $productionStopAttempted) -and -not $productionHealthy) {
                $rollback = Invoke-BestEffortRollback `
                    $Preflight.Production $Preflight.Relationship.ProductionCommit `
                    $backupPath $deploymentStartAttempted
                $manifest.deployment.rollback = $rollback
                if ($rollback.status -eq "healthy") {
                    Clear-ProductionRecoveryBlock $Preflight.Production
                }
                else {
                    Set-ProductionRecoveryBlock `
                        $Preflight.Production $artifacts $failure $rollback
                }
            }
            Write-JsonAtomic $artifacts.ManifestPath $manifest
            throw "Production release failed before synchronization: $failure"
        }

        # Persist synchronization intent before clearing the recovery journal.
        # A crash at any later point can therefore retry only the exact atomic
        # main+tag push without rerunning deployment.
        Save-SyncPending $artifacts "Awaiting initial atomic Git synchronization."
        Clear-ProductionRecoveryBlock $Preflight.Production
        Write-ReleaseMessage "Production is healthy; synchronizing main and the release tag..." "Cyan"
        Push-Release $Preflight $artifacts
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    Write-ReleaseMessage "Release $($artifacts.Id) published successfully." "Green"
}

function Read-JsonFile([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Assert-SyncPendingState([object]$Pending) {
    $required = @(
        "schema_version", "release_id", "tag", "commit",
        "production_branch", "remote", "production_worktree",
        "manifest_path", "created_utc", "last_error"
    )
    foreach ($property in $required) {
        if (-not ($Pending.PSObject.Properties.Name -contains $property)) {
            throw "Sync-pending state is missing '$property'."
        }
    }
    if ([int]$Pending.schema_version -ne 1) {
        throw "Sync-pending state must use schema_version 1."
    }
    $releaseId = [string]$Pending.release_id
    if ($releaseId -notmatch '^pharmacy-release-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$' -or
        [string]$Pending.tag -cne $releaseId) {
        throw "Sync-pending release ID or tag is invalid."
    }
    $commit = [string]$Pending.commit
    if ($commit -notmatch '^[a-f0-9]{40}$') {
        throw "Sync-pending commit is not a full Git object ID."
    }
    if ([string]$Pending.production_branch -cne $ProductionBranch -or
        [string]$Pending.remote -cne $Remote) {
        throw "Sync-pending branch or remote does not match origin/main."
    }
    if ((ConvertTo-NormalizedPath ([string]$Pending.production_worktree)) -ne
        (ConvertTo-NormalizedPath $ProductionWorktree)) {
        throw "Sync-pending production worktree does not match configuration."
    }

    $expectedManifest = Join-Path `
        (Join-Path $releasesDirectory $releaseId) "manifest.json"
    if ((ConvertTo-NormalizedPath ([string]$Pending.manifest_path)) -ne
        (ConvertTo-NormalizedPath $expectedManifest)) {
        throw "Sync-pending manifest path is outside the expected release directory."
    }
    Assert-ExistingFile $expectedManifest "Sync-pending release manifest"
    $manifest = Read-JsonFile $expectedManifest
    if ([string]$manifest.release_id -cne $releaseId -or
        [string]$manifest.release_tag -cne $releaseId -or
        [string]$manifest.source_commit -cne $commit -or
        (ConvertTo-NormalizedPath ([string]$manifest.production_worktree)) -ne
        (ConvertTo-NormalizedPath $ProductionWorktree)) {
        throw "Sync-pending state does not match its release manifest."
    }
    $created = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse(
        [string]$Pending.created_utc,
        [ref]$created
    )) {
        throw "Sync-pending created_utc is invalid."
    }
    return $manifest
}

function Resume-PendingSynchronization([object]$Pending) {
    $validatedManifest = Assert-SyncPendingState $Pending
    Confirm-Publish ([string]$Pending.commit) "SYNC"
    if ($DryRun) {
        Write-ReleaseMessage (
            "DRY RUN complete: would revalidate production and retry the " +
            "atomic push for $($Pending.release_id)."
        ) "Green"
        return
    }

    $productionPath = $ProductionWorktree
    $production = Get-CleanBranchSnapshot `
        $productionPath $ProductionBranch "Production"
    $productionReleaseLock = Enter-ProductionReleaseLock $production
    $script:activeProductionReleaseToken = $productionReleaseLock.Token
    try {
        $production = Get-CleanBranchSnapshot `
            $productionPath $ProductionBranch "Production"
        if ($production.Commit -ne [string]$Pending.commit) {
            throw "Sync-pending commit does not match production HEAD. Manual review is required."
        }
        Assert-ExpectedRemote $production.Path | Out-Null
        $tagCommit = Get-GitValue $production.Path @(
            "rev-parse", "refs/tags/$($Pending.tag)^{}"
        )
        if ($tagCommit -ne [string]$Pending.commit) {
            throw "Sync-pending release tag does not resolve to production HEAD."
        }

        Assert-ProductionHealthy $production
        $push = Invoke-Git -Worktree $production.Path -Mutation -AllowFailure -Arguments @(
            "push", "--atomic", $Remote,
            "refs/heads/${ProductionBranch}:refs/heads/${ProductionBranch}",
            "refs/tags/$($Pending.tag):refs/tags/$($Pending.tag)"
        )
        if ($push.ExitCode -ne 0) {
            $Pending.last_error = $push.Output
            Write-JsonAtomic $pendingPath $Pending
            throw "Git synchronization is still pending.`n$($push.Output)"
        }
        Clear-ProductionRecoveryBlock $production
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    $validatedManifest.synchronization.status = "complete"
    $validatedManifest.synchronization.completed_utc = (Get-UtcNow).ToString("o")
    $validatedManifest.synchronization.error = $null
    Write-JsonAtomic ([string]$Pending.manifest_path) $validatedManifest
    Remove-StateFile $pendingPath
    Write-ReleaseMessage "Pending Git synchronization completed." "Green"
}

function Show-ReleaseStatus {
    if (Test-Path -LiteralPath $pendingPath) {
        $pending = Read-JsonFile $pendingPath
        Write-ReleaseMessage "SYNC PENDING" "Yellow"
        Write-Host "Release:    $($pending.release_id)"
        Write-Host "Commit:     $($pending.commit)"
        Write-Host "Production: $($pending.production_worktree)"
        Write-Host "Error:      $($pending.last_error)"
        Write-Host ""
        Write-Host (
            "After confirming production is healthy, rerun -Action publish " +
            "to retry only the atomic Git synchronization."
        )
        return
    }

    Write-ReleaseMessage "No release synchronization is pending." "Green"
    if (Test-Path -LiteralPath $releasesDirectory) {
        $latestManifest = Get-ChildItem -LiteralPath $releasesDirectory `
            -Recurse -File -Filter "manifest.json" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($latestManifest) {
            $manifest = Read-JsonFile $latestManifest.FullName
            Write-Host "Latest release: $($manifest.release_id)"
            Write-Host "Deployment:     $($manifest.deployment.status)"
            Write-Host "Synchronization: $($manifest.synchronization.status)"
        }
    }
}

$publishLock = $null
try {
    switch ($Action) {
        "status" {
            Show-ReleaseStatus
        }
        "check" {
            $preflight = Get-ReleasePreflight -Fetch -RunChecks
            $shortCommit = $preflight.Relationship.DevelopmentCommit.Substring(0, 12)
            Write-ReleaseMessage (
                "Release check passed: $DevelopmentBranch at $shortCommit is " +
                "$($preflight.Relationship.CommitsAhead) commit(s) ahead of " +
                "$Remote/$ProductionBranch."
            ) "Green"
        }
        "publish" {
            $publishLock = Enter-PublishLock
            if (Test-Path -LiteralPath $pendingPath) {
                Resume-PendingSynchronization (Read-JsonFile $pendingPath)
            }
            else {
                $preflight = Get-ReleasePreflight -Fetch -RunChecks
                Publish-NewRelease $preflight
            }
        }
    }
}
catch {
    Write-Host "Release command failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($ThrowOnError) { throw }
    exit 1
}
finally {
    if ($publishLock) { $publishLock.Dispose() }
}
