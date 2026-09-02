$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$releaseScript = Join-Path $projectRoot "scripts\publish-release.ps1"
$productionScript = Join-Path $projectRoot "scripts\production.ps1"

$tokens = $null
$errors = $null
[Management.Automation.Language.Parser]::ParseFile(
    $releaseScript, [ref]$tokens, [ref]$errors
) | Out-Null
if ($errors.Count -gt 0) {
    throw "publish-release.ps1 has a PowerShell parse error: $($errors[0])"
}

$source = Get-Content -LiteralPath $releaseScript -Raw
$productionSource = Get-Content -LiteralPath $productionScript -Raw
$nativeInvokeStart = $source.IndexOf('else {', $source.IndexOf('function Invoke-ReleaseProcess'))
$nativeInvokeEnd = $source.IndexOf('if ($result.ExitCode', $nativeInvokeStart)
$nativeInvokeSource = $source.Substring(
    $nativeInvokeStart,
    $nativeInvokeEnd - $nativeInvokeStart
)
if ($nativeInvokeSource -notmatch '\$ErrorActionPreference = "Continue"' -or
    $nativeInvokeSource -notmatch '\$exitCode = \$LASTEXITCODE') {
    throw (
        "Release native commands must capture stderr without terminating " +
        "before their actual exit code is handled."
    )
}
if ($source -notmatch '\[switch\]\$ControllerProcess' -or
    $source -notmatch 'elseif \(\$ControllerProcess\)' -or
    $source -notmatch 'Start-Process[\s\S]+-PassThru' -or
    $source -notmatch '\$controller\.WaitForExit\(\)' -or
    $source -notmatch '-EncodedCommand \$encodedCommand' -or
    $source -notmatch '@controllerParameters \*>' -or
    $source -notmatch 'request\.result_path' -or
    $source -notmatch 'Repair-DuplicatePathEnvironment' -or
    $source -notmatch '-ControllerProcess' -or
    $source -match 'RedirectStandard(?:Output|Error)' -or
    $source -match 'Start-Process[^\r\n]+-Wait(?:\s|`|$)') {
    throw (
        "Production controls must use an unredirected encoded wrapper, wait " +
        "only for its direct handle, and never inherit a service output pipe."
    )
}
$requiredContracts = @(
    '\[ValidateSet\("check", "publish", "status", "register-pr", "finalize-pr"\)\]',
    '\[switch\]\$ConfirmRelease',
    '\[switch\]\$DryRun',
    '\[switch\]\$PullRequest',
    '\[string\]\$PullRequestUrl',
    '\[string\]\$SecurityReviewPathForTests',
    '\.runtime\\development-workflow\.json',
    '"status", "--porcelain=v1"',
    'remote", "get-url"',
    'fetch", "--prune"',
    'merge-base", "--is-ancestor"',
    'bundle", "create"',
    'manifest\.json',
    '"tag", "--annotate"',
    '\[IO\.FileShare\]::None',
    '"production-release\.lock"',
    '"production-release\.owner\.json"',
    '\[Guid\]::NewGuid\(\)\.ToString\("N"\)',
    'release_token = \$token',
    '\$script:activeProductionReleaseToken = \$productionReleaseLock\.Token',
    '\$arguments \+= @\("-ReleaseToken", \$script:activeProductionReleaseToken\)',
    'function Exit-ProductionReleaseLock',
    '"-Action", \$ProductionAction',
    '"backup"',
    '"merge", "--ff-only"',
    'Django/DB: healthy',
    'HTTPS:     healthy',
    '"push", "--atomic"',
    'sync-pending\.json',
    'pull-request-pending\.json',
    'security-review-required\.json',
    'CommonApplicationData',
    'FINAL-PHARM\\release-security\\security-review-required\.json',
    'restricted to temporary test worktrees',
    'FileAttributes\]::ReparsePoint',
    'function Get-SecurityReviewBlock',
    'function Assert-SecurityReviewComplete',
    'Production and GitHub release are blocked for security review',
    'function Assert-SyncPendingState',
    'function Assert-PullRequestPendingState',
    'function Assert-PendingRemoteIdentity',
    'function Assert-RunningPublisherMatchesRelease',
    '"diff", "--quiet"',
    'function Clear-MatchingProductionRecoveryBlock',
    'function Push-PullRequestCandidate',
    'function Register-PullRequest',
    'function Finalize-PullRequest',
    'candidate_branch_pending',
    'awaiting_pull_request',
    'awaiting_exact_main',
    '--force-with-lease=refs/heads/',
    '\$\(\$Pending\.commit\):refs/heads/\$\(\$Pending\.review_branch\)',
    'function Get-InterruptedDeploymentState',
    'function Resume-InterruptedDeployment',
    'Assert-FileChecksumSidecar',
    'Interrupted release bundle checksum verification failed',
    'Interrupted production recovery failed',
    'Awaiting interrupted-release Git synchronization',
    'Sync-pending production worktree does not match configuration',
    'Sync-pending manifest path is outside the expected release directory',
    'Sync-pending state does not match its release manifest',
    'production-recovery-required\.json',
    'function Set-ProductionRecoveryBlock',
    'database_restore_required',
    'previous production restored and healthy',
    'rollback aborted before code or database changes'
)
foreach ($pattern in $requiredContracts) {
    if ($source -notmatch $pattern) {
        throw "Release engine source contract is missing: $pattern"
    }
}

$productionGuardContracts = @(
    '\[string\]\$ReleaseToken = ""',
    '"production-release\.lock"',
    '"production-release\.owner\.json"',
    'function Test-ReleaseGateAuthorization',
    '\$owner\.PSObject\.Properties\["release_token"\]',
    '\[Guid\]::TryParse',
    '\[IO\.FileShare\]::None',
    'function Invoke-WithProductionReleaseGate',
    'function Invoke-WithProductionMutationLocks',
    'if \(-not \$ReleaseToken\) \{\s*Set-ProductionOperatorStopped',
    'function Invoke-Django\(\[string\[\]\]\$Arguments\)',
    '\$ErrorActionPreference = "Continue"',
    '\$djangoExitCode = \$LASTEXITCODE'
)
foreach ($pattern in $productionGuardContracts) {
    if ($productionSource -notmatch $pattern) {
        throw "Production release-gate contract is missing: $pattern"
    }
}

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

$publishTransactionStart = $source.IndexOf(
    '$productionReleaseLock = Enter-ProductionReleaseLock'
)
$publishRecoveryBlock = $source.IndexOf(
    'Set-ProductionRecoveryBlock',
    $publishTransactionStart
)
$publishStop = $source.IndexOf(
    'Invoke-ProductionControl $Preflight.Production "stop"',
    $publishTransactionStart
)
$publishBackup = $source.IndexOf(
    'Invoke-ProductionControl $Preflight.Production "backup"',
    $publishStop
)
$publishMerge = $source.IndexOf('"merge", "--ff-only"', $publishBackup)
$publishStart = $source.IndexOf(
    'Invoke-ProductionControl $Preflight.Production "start"',
    $publishMerge
)
$publishHealth = $source.IndexOf(
    'Assert-ProductionHealthy $Preflight.Production',
    $publishStart
)
$publishSyncPending = $source.IndexOf(
    'Save-SyncPending $artifacts "Awaiting initial atomic Git synchronization."',
    $publishHealth
)
$publishRecoveryClear = $source.IndexOf(
    'Clear-ProductionRecoveryBlock $Preflight.Production',
    $publishSyncPending
)
$publishPush = $source.IndexOf(
    'Push-Release $Preflight $artifacts',
    $publishRecoveryClear
)
$publishUnlock = $source.IndexOf(
    'Exit-ProductionReleaseLock $productionReleaseLock',
    $publishPush
)
Assert-True (
    $publishTransactionStart -ge 0 -and
    $publishRecoveryBlock -gt $publishTransactionStart -and
    $publishStop -gt $publishRecoveryBlock -and
    $publishBackup -gt $publishStop -and
    $publishMerge -gt $publishBackup -and
    $publishStart -gt $publishMerge -and
    $publishHealth -gt $publishStart -and
    $publishSyncPending -gt $publishHealth -and
    $publishRecoveryClear -gt $publishSyncPending -and
    $publishPush -gt $publishRecoveryClear -and
    $publishUnlock -gt $publishPush
) (
    "The production release lock must cover the durable recovery block, stop, " +
    "backup, merge, start, health, sync intent, recovery clearance, and push."
)

$mutationLocksStart = $productionSource.IndexOf(
    'function Invoke-WithProductionMutationLocks'
)
$mutationLocksEnd = $productionSource.IndexOf(
    'function Test-IsAdministrator',
    $mutationLocksStart
)
$mutationLocksSource = $productionSource.Substring(
    $mutationLocksStart,
    $mutationLocksEnd - $mutationLocksStart
)
Assert-True (
    $mutationLocksSource.IndexOf('Invoke-WithProductionReleaseGate') -lt
    $mutationLocksSource.IndexOf('Invoke-WithProductionControlLock')
) "Production mutations must take the shared release gate before the control lock."

function New-ReleaseFixture([string]$Name) {
    $root = Join-Path ([IO.Path]::GetTempPath()) (
        "pharmacy-release-$Name-$([Guid]::NewGuid().ToString('N'))"
    )
    $development = Join-Path $root "development"
    $production = Join-Path $root "production"
    $state = Join-Path $development ".runtime\release-engine"
    foreach ($directory in @(
        $development,
        $production,
        (Join-Path $development "env\Scripts"),
        (Join-Path $production "env\Scripts"),
        (Join-Path $production "scripts"),
        (Join-Path $production ".runtime"),
        (Join-Path $development "scripts\tests"),
        (Join-Path $development ".runtime")
    )) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    foreach ($file in @(
        (Join-Path $development "manage.py"),
        (Join-Path $development "env\Scripts\python.exe"),
        (Join-Path $production "manage.py"),
        (Join-Path $production "env\Scripts\python.exe"),
        (Join-Path $production "scripts\production.ps1"),
        (Join-Path $production "scripts\database-restore.ps1"),
        (Join-Path $production ".env"),
        (Join-Path $development "scripts\tests\test-automation-task-scripts.ps1"),
        (Join-Path $development "scripts\tests\test-publish-release.ps1")
    )) {
        [IO.File]::WriteAllText($file, "test fixture")
    }

    $configuration = [ordered]@{
        schema_version = 1
        development_branch = "development"
        production_branch = "main"
        remote = "origin"
        expected_origin_url = "https://github.com/AngusChik/FINAL-PHARM.git"
        production_worktree = $production
    }
    $configurationPath = Join-Path $development ".runtime\development-workflow.json"
    [IO.File]::WriteAllText(
        $configurationPath,
        ($configuration | ConvertTo-Json) + [Environment]::NewLine
    )
    $productionRole = [ordered]@{
        schema_version = 1
        role = "production"
        worktree = $production
        branch = "main"
        remote = "origin"
        created_at = "2026-08-28T12:00:00Z"
    }
    [IO.File]::WriteAllText(
        (Join-Path $production ".runtime\production-role.json"),
        ($productionRole | ConvertTo-Json) + [Environment]::NewLine
    )

    return [pscustomobject]@{
        Root = $root
        Development = $development
        Production = $production
        State = $state
        Configuration = $configurationPath
    }
}

function New-FakeInvoker(
    [object]$Fixture,
    [Collections.Generic.List[object]]$Calls,
    [bool]$DirtyDevelopment = $false,
    [bool]$FailPush = $false,
    [bool]$FailFirstStart = $false,
    [string]$InitialProductionHead = "",
    [bool]$FailRollbackStop = $false,
    [string]$InitialRemoteMain = "",
    [string]$InitialRemoteReleaseCommit = "",
    [string]$InitialRemoteTagCommit = "",
    [string]$InitialRemoteTagObject = "",
    [string]$PullRequestState = "OPEN",
    [string]$ReviewDecision = "APPROVED",
    [object[]]$StatusCheckRollup = @(),
    [bool]$MergePullRequestAfterMainPush = $true,
    [string]$RemoteMainAfterPullRequestView = "",
    [string]$DevelopmentRemoteUrl = "https://github.com/AngusChik/FINAL-PHARM.git",
    [string]$ProductionRemoteUrl = "https://github.com/AngusChik/FINAL-PHARM.git",
    [bool]$PublisherDiffers = $false,
    [bool]$FailPostSyncChecks = $false
) {
    $developmentPath = [IO.Path]::GetFullPath($Fixture.Development).TrimEnd('\', '/')
    $productionPath = [IO.Path]::GetFullPath($Fixture.Production).TrimEnd('\', '/')
    $developmentCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $productionCommit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    $tagObject = "dddddddddddddddddddddddddddddddddddddddd"
    $sharedGitDirectory = Join-Path $Fixture.Root "shared.git"
    $markerDirectory = Join-Path $Fixture.Root "markers"
    $releaseLockPath = Join-Path $Fixture.Production ".runtime\production-release.lock"
    $releaseOwnerPath = Join-Path $Fixture.Production ".runtime\production-release.owner.json"
    $recoveryBlockPath = Join-Path `
        $Fixture.Production ".runtime\production-recovery-required.json"
    $syncPendingPath = Join-Path $Fixture.State "sync-pending.json"
    $pullRequestPendingPath = Join-Path `
        $Fixture.State "pull-request-pending.json"
    $fakeState = @{
        StartCalls = 0
        StopCalls = 0
        ProductionHead = if ($InitialProductionHead) {
            $InitialProductionHead
        }
        else { $productionCommit }
        RemoteMain = if ($InitialRemoteMain) {
            $InitialRemoteMain
        }
        else { $productionCommit }
        RemoteReleaseCommit = $InitialRemoteReleaseCommit
        RemoteTagCommit = $InitialRemoteTagCommit
        RemoteTagObject = $InitialRemoteTagObject
        PullRequestState = $PullRequestState
        PullRequestViewCalls = 0
    }

    return {
        param(
            [string]$FilePath,
            [string[]]$Arguments,
            [string]$WorkingDirectory,
            [bool]$Mutation
        )
        $workingPath = [IO.Path]::GetFullPath($WorkingDirectory).TrimEnd('\', '/')
        $argumentText = $Arguments -join " "
        $releaseLockHeld = $false
        $releaseOwnerToken = ""
        if (Test-Path -LiteralPath $releaseLockPath -PathType Leaf) {
            $probeStream = $null
            try {
                $probeStream = [IO.File]::Open(
                    $releaseLockPath,
                    [IO.FileMode]::Open,
                    [IO.FileAccess]::ReadWrite,
                    [IO.FileShare]::None
                )
            }
            catch [IO.IOException] {
                $releaseLockHeld = $true
            }
            finally {
                if ($null -ne $probeStream) { $probeStream.Dispose() }
            }
        }
        if (Test-Path -LiteralPath $releaseOwnerPath -PathType Leaf) {
            $releaseOwner = Get-Content -LiteralPath $releaseOwnerPath -Raw |
                ConvertFrom-Json
            $releaseOwnerToken = [string]$releaseOwner.release_token
        }
        $Calls.Add([pscustomobject]@{
            FilePath = $FilePath
            Arguments = $argumentText
            WorkingDirectory = $workingPath
            Mutation = $Mutation
            ReleaseLockHeld = $releaseLockHeld
            ReleaseOwnerToken = $releaseOwnerToken
            RecoveryBlockExists = Test-Path `
                -LiteralPath $recoveryBlockPath -PathType Leaf
            SyncPendingExists = Test-Path `
                -LiteralPath $syncPendingPath -PathType Leaf
            PullRequestPendingExists = Test-Path `
                -LiteralPath $pullRequestPendingPath -PathType Leaf
        }) | Out-Null

        $exitCode = 0
        $output = ""
        if ([IO.Path]::GetFileName($FilePath) -ieq "git.exe") {
            if ($argumentText -eq "rev-parse --show-toplevel") {
                $output = $workingPath
            }
            elseif ($argumentText -eq "rev-parse --git-common-dir") {
                $output = $sharedGitDirectory
            }
            elseif ($argumentText -like "rev-parse --git-path *") {
                $marker = $Arguments[$Arguments.Count - 1]
                $output = Join-Path $markerDirectory $marker
            }
            elseif ($argumentText -eq "symbolic-ref --quiet --short HEAD") {
                $output = if ($workingPath -eq $developmentPath) {
                    "development"
                }
                else { "main" }
            }
            elseif ($argumentText -eq "status --porcelain=v1 --untracked-files=all") {
                if ($DirtyDevelopment -and $workingPath -eq $developmentPath) {
                    $output = " M app/views.py"
                }
            }
            elseif ($argumentText -eq "rev-parse HEAD") {
                $output = if ($workingPath -eq $developmentPath) {
                    $developmentCommit
                }
                else { [string]$fakeState.ProductionHead }
            }
            elseif ($argumentText -eq "rev-parse refs/heads/development") {
                $output = $developmentCommit
            }
            elseif ($argumentText -eq "rev-parse refs/heads/main") {
                $output = [string]$fakeState.ProductionHead
            }
            elseif ($argumentText -eq "remote get-url origin") {
                $output = if ($workingPath -eq $developmentPath) {
                    $DevelopmentRemoteUrl
                }
                elseif ($workingPath -eq $productionPath) {
                    $ProductionRemoteUrl
                }
                else { "https://github.com/AngusChik/FINAL-PHARM.git" }
            }
            elseif ($argumentText -like
                "diff --quiet * scripts/publish-release.ps1") {
                if ($PublisherDiffers) {
                    $exitCode = 1
                    $output = "simulated publisher mutation"
                }
            }
            elseif ($argumentText -eq "rev-parse refs/remotes/origin/development") {
                $output = $developmentCommit
            }
            elseif ($argumentText -eq "rev-parse refs/remotes/origin/main") {
                $output = $productionCommit
            }
            elseif ($argumentText -eq "ls-remote origin refs/heads/main") {
                $output = "$($fakeState.RemoteMain)`trefs/heads/main"
            }
            elseif ($argumentText -like "ls-remote origin refs/heads/release/*") {
                $remoteRef = $Arguments[2]
                if ($fakeState.RemoteReleaseCommit) {
                    $output = "$($fakeState.RemoteReleaseCommit)`t$remoteRef"
                }
            }
            elseif ($argumentText -like "ls-remote origin refs/tags/*^{}") {
                $remoteRef = $Arguments[2]
                if ($fakeState.RemoteTagCommit) {
                    $output = "$($fakeState.RemoteTagCommit)`t$remoteRef"
                }
            }
            elseif ($argumentText -like "ls-remote origin refs/tags/*") {
                $remoteRef = $Arguments[2]
                if ($fakeState.RemoteTagObject) {
                    $output = "$($fakeState.RemoteTagObject)`t$remoteRef"
                }
            }
            elseif ($argumentText -like "rev-parse refs/tags/*^{}") {
                $output = $developmentCommit
            }
            elseif ($argumentText -like "rev-parse refs/tags/*") {
                $output = $tagObject
            }
            elseif ($argumentText -eq "rev-list --left-right --count origin/development...development") {
                $output = "0`t0"
            }
            elseif ($argumentText -eq "rev-list --left-right --count origin/main...HEAD") {
                $output = "0`t0"
            }
            elseif ($argumentText -like "rev-list --count origin/main..*") {
                $output = "3"
            }
            elseif ($argumentText -like "show-ref --verify --quiet refs/tags/*") {
                $exitCode = 1
            }
            elseif ($argumentText -like "bundle create *") {
                $bundlePath = $Arguments[2]
                $bundleParent = Split-Path $bundlePath -Parent
                New-Item -ItemType Directory -Force -Path $bundleParent | Out-Null
                [IO.File]::WriteAllText($bundlePath, "fake git bundle")
            }
            elseif ($argumentText -eq "merge --ff-only $developmentCommit") {
                $fakeState.ProductionHead = $developmentCommit
                [IO.File]::WriteAllText(
                    (Join-Path $Fixture.Production "scripts\production.ps1"),
                    'param([switch]$NonInteractive, [string]$ReleaseToken = "")'
                )
            }
            elseif ($argumentText -eq "reset --hard $productionCommit") {
                $fakeState.ProductionHead = $productionCommit
                [IO.File]::WriteAllText(
                    (Join-Path $Fixture.Production "scripts\production.ps1"),
                    "test fixture"
                )
            }
            elseif ($argumentText -like "push --atomic *" -and $FailPush) {
                $exitCode = 1
                $output = "simulated network failure"
            }
            elseif ($argumentText -like "push origin *:refs/heads/release/*" -and $FailPush) {
                $exitCode = 1
                $output = "simulated network failure"
            }
            elseif ($argumentText -like "push origin *:refs/heads/release/*") {
                $fakeState.RemoteReleaseCommit = $developmentCommit
            }
            elseif ($argumentText -like "push --atomic *:refs/heads/main*") {
                $fakeState.RemoteMain = $developmentCommit
                $fakeState.RemoteTagCommit = $developmentCommit
                $fakeState.RemoteTagObject = $tagObject
                if ($MergePullRequestAfterMainPush) {
                    $fakeState.PullRequestState = "MERGED"
                }
            }
            elseif ($argumentText -like "push --atomic *refs/tags/*:refs/tags/*") {
                $fakeState.RemoteTagCommit = $developmentCommit
                $fakeState.RemoteTagObject = $tagObject
            }
        }
        elseif ([IO.Path]::GetFileName($FilePath) -ieq "gh.exe") {
            if ($argumentText -eq "auth status --hostname github.com") {
                $output = "authenticated"
            }
            elseif ($argumentText -like "repo view * --json nameWithOwner --jq .nameWithOwner") {
                $output = "anguschik/final-pharm"
            }
            elseif ($argumentText -like "pr view * --repo * --json *") {
                $fakeState.PullRequestViewCalls += 1
                $pendingState = Get-Content -LiteralPath $pullRequestPendingPath -Raw |
                    ConvertFrom-Json
                $reportedChecks = if ($FailPostSyncChecks -and
                    $fakeState.PullRequestViewCalls -gt 1) {
                    @([pscustomobject]@{
                        __typename = "StatusContext"
                        state = "FAILURE"
                    })
                }
                else { @($StatusCheckRollup) }
                $output = [ordered]@{
                    number = 321
                    url = "https://github.com/anguschik/final-pharm/pull/321"
                    state = [string]$fakeState.PullRequestState
                    isDraft = $false
                    headRefName = [string]$pendingState.review_branch
                    headRefOid = $developmentCommit
                    baseRefName = "main"
                    reviewDecision = $ReviewDecision
                    statusCheckRollup = @($reportedChecks)
                } | ConvertTo-Json -Depth 8 -Compress
                if ($RemoteMainAfterPullRequestView) {
                    $fakeState.RemoteMain = $RemoteMainAfterPullRequestView
                }
            }
        }
        elseif ([IO.Path]::GetFileName($FilePath) -ieq "powershell.exe") {
            $actionIndex = [Array]::IndexOf($Arguments, "-Action")
            $productionAction = if ($actionIndex -ge 0) {
                $Arguments[$actionIndex + 1]
            }
            else { "" }
            if ($productionAction -eq "status") {
                $output = @(
                    "Waitress: running",
                    "Caddy:    running",
                    "Django/DB: healthy (HTTP 200)",
                    "HTTPS:     healthy"
                ) -join "`n"
            }
            elseif ($productionAction -eq "backup") {
                $backupDirectory = Join-Path $Fixture.Production "backups\database"
                New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
                $backupPath = Join-Path $backupDirectory "verified-final-backup.dump"
                [IO.File]::WriteAllText($backupPath, "verified final database backup")
                $backupHash = (Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash
                [IO.File]::WriteAllText(
                    "$backupPath.sha256",
                    "$backupHash  $(Split-Path -Leaf $backupPath)$([Environment]::NewLine)"
                )
                $output = $backupPath
            }
            elseif ($productionAction -eq "start") {
                $fakeState.StartCalls += 1
                if ($FailFirstStart -and $fakeState.StartCalls -eq 1) {
                    $exitCode = 1
                    $output = "simulated candidate startup failure"
                }
            }
            elseif ($productionAction -eq "stop") {
                $fakeState.StopCalls += 1
                if ($FailRollbackStop -and $fakeState.StopCalls -gt 1) {
                    $exitCode = 1
                    $output = "simulated rollback stop failure"
                }
            }
        }

        return [pscustomobject]@{ ExitCode = $exitCode; Output = $output }
    }.GetNewClosure()
}

function Initialize-InterruptedReleaseFixture([object]$Fixture) {
    [IO.File]::WriteAllText(
        (Join-Path $Fixture.Production "scripts\production.ps1"),
        'param([switch]$NonInteractive, [string]$ReleaseToken = "")'
    )
    $releaseId = "pharmacy-release-20260828T120000Z-aaaaaaaaaaaa"
    $releaseDirectory = Join-Path $Fixture.State "releases\$releaseId"
    $backupDirectory = Join-Path $Fixture.Production "backups\database"
    New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
    $bundlePath = Join-Path $releaseDirectory "$releaseId.bundle"
    $backupPath = Join-Path $backupDirectory "pharmacy-20260828-120000-manual.dump"
    [IO.File]::WriteAllText($bundlePath, "verified interrupted bundle")
    [IO.File]::WriteAllText($backupPath, "verified interrupted database")
    $bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
    $backupHash = (Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash
    $manifestPath = Join-Path $releaseDirectory "manifest.json"
    $manifest = [ordered]@{
        schema_version = 1
        release_id = $releaseId
        release_tag = $releaseId
        source_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        previous_production_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        production_branch = "main"
        remote = "origin"
        remote_url = "https://github.com/AngusChik/FINAL-PHARM.git"
        production_worktree = $Fixture.Production
        bundle_path = $bundlePath
        bundle_sha256 = $bundleHash
        production_backup_path = $backupPath
        checks = [ordered]@{
            repository = "passed"
            candidate = "passed"
            backup = "passed"
            production_health = "pending"
        }
        deployment = [ordered]@{
            status = "starting"
            completed_utc = $null
            rollback = $null
        }
        synchronization = [ordered]@{
            status = "not_started"
            completed_utc = $null
            error = $null
        }
    }
    [IO.File]::WriteAllText(
        $manifestPath,
        ($manifest | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $recoveryPath = Join-Path `
        $Fixture.Production ".runtime\production-recovery-required.json"
    $journal = [ordered]@{
        schema_version = 1
        release_id = $releaseId
        failed_release_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        previous_production_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        production_backup_path = $backupPath
        failure = "Release deployment is in progress after the final backup."
        rollback = [ordered]@{
            status = "backup_verified"
            database_restore_required = $false
            completed_utc = $null
            notes = @("Verified rollback backup: $backupPath")
        }
        created_utc = "2026-08-28T12:00:00Z"
    }
    [IO.File]::WriteAllText(
        $recoveryPath,
        ($journal | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    [IO.File]::WriteAllText(
        "$backupPath.sha256",
        "$backupHash  $(Split-Path -Leaf $backupPath)$([Environment]::NewLine)"
    )
    return [pscustomobject]@{
        ReleaseId = $releaseId
        ManifestPath = $manifestPath
        RecoveryPath = $recoveryPath
        BackupPath = $backupPath
        BackupHash = $backupHash
    }
}

function Write-TestSecurityReviewMarker([string]$Path) {
    $directory = Split-Path $Path -Parent
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $block = [ordered]@{
        schema_version = 1
        status = "review_required"
        reason = "Executable release-gate test hold."
    }
    [IO.File]::WriteAllText(
        $Path,
        ($block | ConvertTo-Json) + [Environment]::NewLine
    )
}

$fixtures = New-Object Collections.Generic.List[string]
try {
    # The machine security hold is fixed independently of release state. Tests
    # may override it only while all worktrees and injected commands are safely
    # contained beneath the OS temporary directory.
    $securityFixture = New-ReleaseFixture "security-hold"
    $fixtures.Add($securityFixture.Root)
    $securityPath = Join-Path `
        $securityFixture.Root "machine-security\security-review-required.json"
    $alternateState = Join-Path `
        $securityFixture.Development ".runtime\alternate-release-engine"
    New-Item -ItemType Directory -Force -Path $alternateState | Out-Null
    $securityCalls = New-Object Collections.Generic.List[object]
    $securityInvoker = New-FakeInvoker $securityFixture $securityCalls

    $unsafeTestSeamRejected = $false
    try {
        & $releaseScript -Action status `
            -DevelopmentWorktree $securityFixture.Development `
            -WorkflowConfig $securityFixture.Configuration `
            -StateDirectory $alternateState `
            -SecurityReviewPathForTests $securityPath -ThrowOnError
    }
    catch {
        $unsafeTestSeamRejected = $_.Exception.Message -match `
            'requires the temporary injected-command test seam'
    }
    Assert-True $unsafeTestSeamRejected `
        "A normal release invocation must not be able to override the machine security path."

    Write-TestSecurityReviewMarker $securityPath
    foreach ($blockedAction in @("publish", "register-pr", "finalize-pr")) {
        $blocked = $false
        try {
            & $releaseScript -Action $blockedAction -ConfirmRelease `
                -DevelopmentWorktree $securityFixture.Development `
                -WorkflowConfig $securityFixture.Configuration `
                -StateDirectory $alternateState `
                -CommandInvoker $securityInvoker `
                -SecurityReviewPathForTests $securityPath -ThrowOnError
        }
        catch {
            $blocked = $_.Exception.Message -match `
                'blocked for security review: Executable release-gate test hold'
        }
        Assert-True $blocked `
            "Security hold must block $blockedAction even with an alternate release-state directory."
    }
    Assert-True ($securityCalls.Count -eq 0) `
        "Blocked release actions must fail before invoking external dependencies."

    Remove-Item -LiteralPath $securityPath -Force
    New-Item -ItemType Directory -Path $securityPath | Out-Null
    $directoryStateBlocked = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $securityFixture.Development `
            -WorkflowConfig $securityFixture.Configuration `
            -StateDirectory $alternateState `
            -CommandInvoker $securityInvoker `
            -SecurityReviewPathForTests $securityPath -ThrowOnError
    }
    catch {
        $directoryStateBlocked = $_.Exception.Message -match `
            'not a regular local file; release remains blocked'
    }
    Assert-True $directoryStateBlocked `
        "A directory at the security marker path must fail closed."

    Remove-Item -LiteralPath $securityPath -Recurse -Force
    [IO.File]::WriteAllText($securityPath, '{not-json')
    $malformedStateBlocked = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $securityFixture.Development `
            -WorkflowConfig $securityFixture.Configuration `
            -StateDirectory $alternateState `
            -CommandInvoker $securityInvoker `
            -SecurityReviewPathForTests $securityPath -ThrowOnError
    }
    catch {
        $malformedStateBlocked = $_.Exception.Message -match `
            'Security review state is invalid; release remains blocked'
    }
    Assert-True $malformedStateBlocked `
        "Malformed security-review state must fail closed."

    # Simulate a hold appearing after the action-level check. The boundary
    # recheck must stop before any production control or GitHub push.
    $raceFixture = New-ReleaseFixture "security-race"
    $fixtures.Add($raceFixture.Root)
    $raceSecurityPath = Join-Path `
        $raceFixture.Root "machine-security\security-review-required.json"
    New-Item -ItemType Directory -Force `
        -Path (Split-Path $raceSecurityPath -Parent) | Out-Null
    $raceCalls = New-Object Collections.Generic.List[object]
    $raceBaseInvoker = New-FakeInvoker $raceFixture $raceCalls
    $raceMarkerWritten = $false
    $raceMarkerJson = (@{
        schema_version = 1
        status = "review_required"
        reason = "Hold appeared during release checks."
    } | ConvertTo-Json) + [Environment]::NewLine
    $raceInvoker = {
        param(
            [string]$FilePath,
            [string[]]$Arguments,
            [string]$WorkingDirectory,
            [bool]$Mutation
        )
        $result = & $raceBaseInvoker `
            $FilePath ([string[]]$Arguments) $WorkingDirectory $Mutation
        if (-not $raceMarkerWritten) {
            [IO.File]::WriteAllText($raceSecurityPath, $raceMarkerJson)
            $raceMarkerWritten = $true
        }
        return $result
    }.GetNewClosure()
    $raceBlocked = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $raceFixture.Development `
            -WorkflowConfig $raceFixture.Configuration `
            -StateDirectory $raceFixture.State `
            -CommandInvoker $raceInvoker `
            -SecurityReviewPathForTests $raceSecurityPath -ThrowOnError
    }
    catch {
        $raceBlocked = $_.Exception.Message -match `
            'blocked for security review: Hold appeared during release checks'
    }
    Assert-True $raceBlocked `
        "A security hold created after dispatch must be caught at the production boundary."
    $raceText = ($raceCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($raceText -notmatch '-Action (stop|backup|start)|^push ') `
        "A newly raised hold must stop before production control or any GitHub push."

    # Raise the hold only after the candidate is deployed and its health check
    # returns. The pre-push boundary must retain pending state without touching
    # any GitHub ref.
    $pushRaceFixture = New-ReleaseFixture "security-push-race"
    $fixtures.Add($pushRaceFixture.Root)
    $pushRaceSecurityPath = Join-Path `
        $pushRaceFixture.Root "machine-security\security-review-required.json"
    New-Item -ItemType Directory -Force `
        -Path (Split-Path $pushRaceSecurityPath -Parent) | Out-Null
    $pushRaceCalls = New-Object Collections.Generic.List[object]
    $pushRaceBaseInvoker = New-FakeInvoker $pushRaceFixture $pushRaceCalls
    $pushRaceMarkerWritten = $false
    $pushRaceMarkerJson = (@{
        schema_version = 1
        status = "review_required"
        reason = "Hold appeared before GitHub push."
    } | ConvertTo-Json) + [Environment]::NewLine
    $pushRaceInvoker = {
        param(
            [string]$FilePath,
            [string[]]$Arguments,
            [string]$WorkingDirectory,
            [bool]$Mutation
        )
        $result = & $pushRaceBaseInvoker `
            $FilePath ([string[]]$Arguments) $WorkingDirectory $Mutation
        $argumentText = $Arguments -join " "
        if (-not $pushRaceMarkerWritten -and
            $FilePath -match '(?i)powershell(?:\.exe)?$' -and
            $argumentText -match '(?:^| )-Action status(?: |$)') {
            [IO.File]::WriteAllText(
                $pushRaceSecurityPath,
                $pushRaceMarkerJson
            )
            $pushRaceMarkerWritten = $true
        }
        return $result
    }.GetNewClosure()
    $pushRaceBlocked = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $pushRaceFixture.Development `
            -WorkflowConfig $pushRaceFixture.Configuration `
            -StateDirectory $pushRaceFixture.State `
            -CommandInvoker $pushRaceInvoker `
            -SecurityReviewPathForTests $pushRaceSecurityPath -ThrowOnError
    }
    catch {
        $pushRaceBlocked = $_.Exception.Message -match `
            'blocked for security review: Hold appeared before GitHub push'
    }
    Assert-True $pushRaceBlocked `
        "A security hold raised after production health must block the review-branch push."
    Assert-True (Test-Path -LiteralPath $pushRaceSecurityPath -PathType Leaf) `
        "The push-boundary fixture must raise its hold after production status."
    Assert-True (-not ($pushRaceCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "No GitHub ref may be pushed after the security hold appears."
    $pushRacePendingPath = Join-Path `
        $pushRaceFixture.State "pull-request-pending.json"
    Assert-True (Test-Path -LiteralPath $pushRacePendingPath -PathType Leaf) `
        "A blocked post-health push must retain durable pull-request pending state."

    # New releases must use the production-first pull-request workflow. Direct
    # main publication remains available only to resume an existing legacy
    # sync-pending journal.
    $directFixture = New-ReleaseFixture "direct-rejected"
    $fixtures.Add($directFixture.Root)
    $directCalls = New-Object Collections.Generic.List[object]
    $directInvoker = New-FakeInvoker $directFixture $directCalls
    $directRejected = $false
    try {
        & $releaseScript -Action check `
            -DevelopmentWorktree $directFixture.Development `
            -WorkflowConfig $directFixture.Configuration `
            -StateDirectory $directFixture.State `
            -CommandInvoker $directInvoker -ThrowOnError
    }
    catch {
        $directRejected = $_.Exception.Message -match 'require -PullRequest'
    }
    Assert-True $directRejected `
        "A new release check without -PullRequest must fail closed."
    Assert-True ($directCalls.Count -eq 0) `
        "Rejected direct release mode must fail before invoking dependencies."

    # `check` may fetch and run tests, but must not tag, bundle, deploy, or push.
    $checkFixture = New-ReleaseFixture "check"
    $fixtures.Add($checkFixture.Root)
    $checkCalls = New-Object Collections.Generic.List[object]
    $checkInvoker = New-FakeInvoker $checkFixture $checkCalls
    & $releaseScript -Action check -PullRequest `
        -DevelopmentWorktree $checkFixture.Development `
        -WorkflowConfig $checkFixture.Configuration `
        -StateDirectory $checkFixture.State `
        -CommandInvoker $checkInvoker -ThrowOnError

    $checkText = ($checkCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($checkText -match 'fetch --prune origin') `
        "Check must refresh origin refs."
    $checkFetch = $checkCalls | Where-Object { $_.Arguments -match '^fetch ' } |
        Select-Object -ExpandProperty Arguments -First 1
    Assert-True ($checkFetch -notmatch 'development') `
        "Local development must not be pushed or fetched from GitHub before release."
    Assert-True ($checkText -match 'manage\.py test(?: |$)') `
        "Check must run the application test suite."
    Assert-True ($checkText -notmatch 'tag --annotate|bundle create|merge --ff-only|push --atomic') `
        "Check must not create or deploy a release."
    Assert-True ($checkText -notmatch '-Action (backup|stop|start)') `
        "Check must not control production."

    # A publish dry run must perform read-only validation without allowing any
    # injected mutating command or creating release state.
    $dryFixture = New-ReleaseFixture "dry-run"
    $fixtures.Add($dryFixture.Root)
    $dryCalls = New-Object Collections.Generic.List[object]
    $dryInvoker = New-FakeInvoker $dryFixture $dryCalls
    & $releaseScript -Action publish -PullRequest -DryRun `
        -DevelopmentWorktree $dryFixture.Development `
        -WorkflowConfig $dryFixture.Configuration `
        -StateDirectory $dryFixture.State `
        -CommandInvoker $dryInvoker -ThrowOnError
    Assert-True (-not ($dryCalls | Where-Object { $_.Mutation })) `
        "Dry-run publish must not invoke mutating dependencies."
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $dryFixture.State "releases"))) `
        "Dry-run publish must not create release artifacts."

    # A dirty development tree must be rejected before fetch or tests.
    $dirtyFixture = New-ReleaseFixture "dirty"
    $fixtures.Add($dirtyFixture.Root)
    $dirtyCalls = New-Object Collections.Generic.List[object]
    $dirtyInvoker = New-FakeInvoker $dirtyFixture $dirtyCalls $true
    $dirtyRejected = $false
    try {
        & $releaseScript -Action check -PullRequest `
            -DevelopmentWorktree $dirtyFixture.Development `
            -WorkflowConfig $dirtyFixture.Configuration `
            -StateDirectory $dirtyFixture.State `
            -CommandInvoker $dirtyInvoker -ThrowOnError
    }
    catch {
        $dirtyRejected = $_.Exception.Message -match 'must be clean'
    }
    Assert-True $dirtyRejected "Dirty development must fail release preflight."
    $dirtyText = ($dirtyCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($dirtyText -notmatch 'fetch --prune|manage\.py') `
        "Dirty development must fail before fetch and candidate checks."

    # Direct-main publication is retained only to recover a durable legacy
    # sync-pending journal. Build that journal explicitly rather than using the
    # now-prohibited path for creating a new direct release.
    $pendingFixture = New-ReleaseFixture "pending"
    $fixtures.Add($pendingFixture.Root)
    [IO.File]::WriteAllText(
        (Join-Path $pendingFixture.Production "scripts\production.ps1"),
        'param([switch]$NonInteractive, [string]$ReleaseToken = "")'
    )
    $legacyReleaseId = "pharmacy-release-20260828T123456Z-aaaaaaaaaaaa"
    $legacyReleaseDirectory = Join-Path `
        $pendingFixture.State "releases\$legacyReleaseId"
    New-Item -ItemType Directory -Force -Path $legacyReleaseDirectory |
        Out-Null
    $syncPendingPath = Join-Path $pendingFixture.State "sync-pending.json"
    $legacyManifestPath = Join-Path $legacyReleaseDirectory "manifest.json"
    $legacyManifest = [ordered]@{
        schema_version = 1
        release_id = $legacyReleaseId
        release_tag = $legacyReleaseId
        source_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        production_branch = "main"
        remote = "origin"
        remote_url = "https://github.com/AngusChik/FINAL-PHARM.git"
        production_worktree = $pendingFixture.Production
        deployment = [ordered]@{
            status = "healthy"
            completed_utc = "2026-08-28T12:34:56Z"
            rollback = $null
        }
        synchronization = [ordered]@{
            status = "pending"
            completed_utc = $null
            error = "simulated legacy network failure"
        }
    }
    [IO.File]::WriteAllText(
        $legacyManifestPath,
        ($legacyManifest | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $legacyPending = [ordered]@{
        schema_version = 1
        release_id = $legacyReleaseId
        tag = $legacyReleaseId
        commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        production_branch = "main"
        remote = "origin"
        remote_url = "https://github.com/AngusChik/FINAL-PHARM.git"
        production_worktree = $pendingFixture.Production
        manifest_path = $legacyManifestPath
        created_utc = "2026-08-28T12:34:56Z"
        last_error = "simulated legacy network failure"
    }
    [IO.File]::WriteAllText(
        $syncPendingPath,
        ($legacyPending | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $pending = Get-Content -LiteralPath $syncPendingPath -Raw | ConvertFrom-Json
    $manifest = Get-Content -LiteralPath $pending.manifest_path -Raw | ConvertFrom-Json
    $validLegacyManifestJson = Get-Content -LiteralPath $pending.manifest_path -Raw
    $pendingRecoveryPath = Join-Path `
        $pendingFixture.Production ".runtime\production-recovery-required.json"

    # The pending record is ignored runtime state, so corruption must fail
    # closed before it can redirect a health check, manifest write, or push.
    $validPendingJson = Get-Content -LiteralPath $syncPendingPath -Raw
    $corruptPending = $validPendingJson | ConvertFrom-Json
    $corruptPending.remote = "unexpected-remote"
    [IO.File]::WriteAllText(
        $syncPendingPath,
        ($corruptPending | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $corruptCalls = New-Object Collections.Generic.List[object]
    $corruptInvoker = New-FakeInvoker `
        -Fixture $pendingFixture `
        -Calls $corruptCalls `
        -InitialProductionHead "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $corruptRejected = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $pendingFixture.Development `
            -WorkflowConfig $pendingFixture.Configuration `
            -StateDirectory $pendingFixture.State `
            -CommandInvoker $corruptInvoker `
            -ThrowOnError
    }
    catch {
        $corruptRejected = `
            $_.Exception.Message -match 'branch or remote does not match'
    }
    Assert-True $corruptRejected `
        "Corrupt sync-pending branch or remote must fail closed."
    Assert-True ($corruptCalls.Count -eq 0) `
        "Corrupt sync-pending state must fail before commands or mutations."
    Assert-True (Test-Path -LiteralPath $syncPendingPath -PathType Leaf) `
        "Rejected sync-pending state must remain available for manual review."
    [IO.File]::WriteAllText($syncPendingPath, $validPendingJson)

    $missingPendingRemote = $validPendingJson | ConvertFrom-Json
    $missingPendingRemote.PSObject.Properties.Remove("remote_url")
    [IO.File]::WriteAllText(
        $syncPendingPath,
        ($missingPendingRemote | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $missingPendingCalls = New-Object Collections.Generic.List[object]
    $missingPendingInvoker = New-FakeInvoker `
        -Fixture $pendingFixture -Calls $missingPendingCalls `
        -InitialProductionHead ("a" * 40)
    $missingPendingRejected = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $pendingFixture.Development `
            -WorkflowConfig $pendingFixture.Configuration `
            -StateDirectory $pendingFixture.State `
            -CommandInvoker $missingPendingInvoker -ThrowOnError
    }
    catch {
        $missingPendingRejected = $_.Exception.Message -match
            "Sync-pending state is missing 'remote_url'"
    }
    Assert-True $missingPendingRejected `
        "Legacy pending state must require its recorded remote URL."
    Assert-True ($missingPendingCalls.Count -eq 0) `
        "Missing legacy remote identity must fail before commands."
    [IO.File]::WriteAllText($syncPendingPath, $validPendingJson)

    foreach ($manifestRemoteProperty in @(
        "production_branch", "remote", "remote_url"
    )) {
        $missingManifestRemote = $validLegacyManifestJson | ConvertFrom-Json
        $missingManifestRemote.PSObject.Properties.Remove($manifestRemoteProperty)
        [IO.File]::WriteAllText(
            $legacyManifestPath,
            ($missingManifestRemote | ConvertTo-Json -Depth 12) +
                [Environment]::NewLine
        )
        $missingManifestCalls = New-Object Collections.Generic.List[object]
        $missingManifestInvoker = New-FakeInvoker `
            -Fixture $pendingFixture -Calls $missingManifestCalls `
            -InitialProductionHead ("a" * 40)
        $missingManifestRejected = $false
        try {
            & $releaseScript -Action publish -ConfirmRelease `
                -DevelopmentWorktree $pendingFixture.Development `
                -WorkflowConfig $pendingFixture.Configuration `
                -StateDirectory $pendingFixture.State `
                -CommandInvoker $missingManifestInvoker -ThrowOnError
        }
        catch {
            $missingManifestRejected = $_.Exception.Message -match
                "Sync-pending release manifest is missing '$manifestRemoteProperty'"
        }
        finally {
            [IO.File]::WriteAllText(
                $legacyManifestPath,
                $validLegacyManifestJson
            )
        }
        Assert-True $missingManifestRejected `
            "Legacy manifest must require $manifestRemoteProperty."
        Assert-True ($missingManifestCalls.Count -eq 0) `
            "Missing legacy manifest identity must fail before commands."
    }

    $legacyRedirectCalls = New-Object Collections.Generic.List[object]
    $legacyRedirectInvoker = New-FakeInvoker `
        -Fixture $pendingFixture -Calls $legacyRedirectCalls `
        -InitialProductionHead ("a" * 40) `
        -ProductionRemoteUrl "https://github.com/attacker/redirected.git"
    $legacyRedirectRejected = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $pendingFixture.Development `
            -WorkflowConfig $pendingFixture.Configuration `
            -StateDirectory $pendingFixture.State `
            -CommandInvoker $legacyRedirectInvoker -ThrowOnError
    }
    catch {
        $legacyRedirectRejected = $_.Exception.Message -match
            "Remote 'origin'.*expected"
    }
    Assert-True $legacyRedirectRejected `
        "Legacy pending synchronization must reject a redirected production remote."
    Assert-True (-not ($legacyRedirectCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "Legacy remote redirection must fail before synchronization."

    $legacyPublisherCalls = New-Object Collections.Generic.List[object]
    $legacyPublisherInvoker = New-FakeInvoker `
        -Fixture $pendingFixture -Calls $legacyPublisherCalls `
        -InitialProductionHead ("a" * 40) `
        -PublisherDiffers $true
    $legacyPublisherRejected = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $pendingFixture.Development `
            -WorkflowConfig $pendingFixture.Configuration `
            -StateDirectory $pendingFixture.State `
            -CommandInvoker $legacyPublisherInvoker -ThrowOnError
    }
    catch {
        $legacyPublisherRejected = $_.Exception.Message -match
            'running publisher differs from the production release commit'
    }
    Assert-True $legacyPublisherRejected `
        "Legacy pending synchronization must reject a mutated publisher."
    Assert-True (-not ($legacyPublisherCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "Legacy publisher mutation must fail before synchronization."

    # Rerunning publish with a durable pending record must only revalidate the
    # deployed commit and health, then retry the same atomic Git sync while the
    # production release gate is held.
    $syncCalls = New-Object Collections.Generic.List[object]
    $syncInvoker = New-FakeInvoker `
        -Fixture $pendingFixture `
        -Calls $syncCalls `
        -InitialProductionHead "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    & $releaseScript -Action publish -ConfirmRelease `
        -DevelopmentWorktree $pendingFixture.Development `
        -WorkflowConfig $pendingFixture.Configuration `
        -StateDirectory $pendingFixture.State `
        -CommandInvoker $syncInvoker `
        -Clock { [DateTimeOffset]::Parse("2026-08-28T12:35:06Z") } `
        -ThrowOnError

    Assert-True (-not (Test-Path -LiteralPath $syncPendingPath)) `
        "Successful pending synchronization must remove sync-pending.json."
    Assert-True (-not (Test-Path -LiteralPath $pendingRecoveryPath)) `
        "Pending synchronization must not create a production recovery block."
    $completedManifest = Get-Content -LiteralPath $pending.manifest_path -Raw |
        ConvertFrom-Json
    Assert-True ($completedManifest.synchronization.status -eq "complete") `
        "Pending synchronization must mark the original manifest complete."
    $syncText = ($syncCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($syncText -match '-Action status') `
        "Pending synchronization must revalidate production health."
    Assert-True ($syncText -match 'push --atomic') `
        "Pending synchronization must retry the atomic main-and-tag push."
    Assert-True ($syncText -notmatch
        'manage\.py|tag --annotate|bundle create|merge --ff-only|-Action (?:backup|stop|start)') `
        "Pending synchronization must not rerun checks, artifacts, deployment, or migrations."
    foreach ($syncGuardedCall in @(
        ($syncCalls | Where-Object {
            $_.Arguments -match '-Action status(?: |$)'
        } | Select-Object -First 1),
        ($syncCalls | Where-Object {
            $_.Arguments -match '^push --atomic'
        } | Select-Object -First 1)
    )) {
        Assert-True ($syncGuardedCall.ReleaseLockHeld) `
            "Pending health and push must remain inside the production release gate."
    }

    # A review-branch network failure happens after healthy production and must
    # therefore retain a resumable branch-only phase without rolling back.
    $prRetryFixture = New-ReleaseFixture "pull-request-retry"
    $fixtures.Add($prRetryFixture.Root)
    $prFailureCalls = New-Object Collections.Generic.List[object]
    $prFailureInvoker = New-FakeInvoker `
        -Fixture $prRetryFixture -Calls $prFailureCalls -FailPush $true
    $prBranchFailed = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $prRetryFixture.Development `
            -WorkflowConfig $prRetryFixture.Configuration `
            -StateDirectory $prRetryFixture.State `
            -CommandInvoker $prFailureInvoker `
            -Clock { [DateTimeOffset]::Parse("2026-08-28T12:35:30Z") } `
            -ThrowOnError
    }
    catch {
        $prBranchFailed = $_.Exception.Message -match 'review branch is pending'
    }
    Assert-True $prBranchFailed `
        "A failed review-branch push must report a resumable pending state."
    $prRetryPath = Join-Path `
        $prRetryFixture.State "pull-request-pending.json"
    $prRetryPending = Get-Content -LiteralPath $prRetryPath -Raw |
        ConvertFrom-Json
    Assert-True ($prRetryPending.phase -eq "candidate_branch_pending") `
        "Failed review-branch publication must retain its exact retry phase."
    $prRetryManifest = Get-Content -LiteralPath `
        $prRetryPending.manifest_path -Raw | ConvertFrom-Json
    Assert-True ($prRetryManifest.deployment.status -eq "healthy") `
        "A post-health branch failure must leave the production deployment healthy."
    $prFailureText = ($prFailureCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($prFailureText -notmatch 'reset --hard') `
        "A post-health review-branch failure must never roll production back."

    $prRetryCalls = New-Object Collections.Generic.List[object]
    $prRetryInvoker = New-FakeInvoker `
        -Fixture $prRetryFixture -Calls $prRetryCalls `
        -InitialProductionHead "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    & $releaseScript -Action publish -PullRequest -ConfirmRelease `
        -DevelopmentWorktree $prRetryFixture.Development `
        -WorkflowConfig $prRetryFixture.Configuration `
        -StateDirectory $prRetryFixture.State `
        -CommandInvoker $prRetryInvoker -ThrowOnError
    $prRetryPending = Get-Content -LiteralPath $prRetryPath -Raw |
        ConvertFrom-Json
    Assert-True ($prRetryPending.phase -eq "awaiting_pull_request") `
        "A successful branch-only retry must advance to PR creation."
    $prRetryText = ($prRetryCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($prRetryText -notmatch
        'manage\.py|tag --annotate|bundle create|merge --ff-only|-Action (?:backup|stop|start)') `
        "A review-branch retry must not repeat checks, artifacts, or deployment."

    # Pull-request mode deploys and proves production first, then publishes only
    # an immutable review branch. Main and the public release tag remain
    # untouched until the registered PR is explicitly finalized.
    $prFixture = New-ReleaseFixture "pull-request"
    $fixtures.Add($prFixture.Root)
    $prCalls = New-Object Collections.Generic.List[object]
    $prInvoker = New-FakeInvoker $prFixture $prCalls
    & $releaseScript -Action publish -PullRequest -ConfirmRelease `
        -DevelopmentWorktree $prFixture.Development `
        -WorkflowConfig $prFixture.Configuration `
        -StateDirectory $prFixture.State `
        -CommandInvoker $prInvoker `
        -Clock { [DateTimeOffset]::Parse("2026-08-28T12:36:00Z") } `
        -ThrowOnError

    $prPendingPath = Join-Path $prFixture.State "pull-request-pending.json"
    Assert-True (Test-Path -LiteralPath $prPendingPath -PathType Leaf) `
        "A healthy PR-first deployment must retain a durable release block."
    $prPending = Get-Content -LiteralPath $prPendingPath -Raw | ConvertFrom-Json
    Assert-True ($prPending.phase -eq "awaiting_pull_request") `
        "Successful review-branch publication must await PR registration."
    Assert-True ($prPending.review_branch -eq
        "release/$($prPending.release_id)") `
        "The review branch must be derived from the immutable release ID."
    $prManifest = Get-Content -LiteralPath $prPending.manifest_path -Raw |
        ConvertFrom-Json
    Assert-True ($prManifest.deployment.status -eq "healthy" -and
        $prManifest.synchronization.mode -eq "pull_request" -and
        $prManifest.synchronization.status -eq "pull_request_pending") `
        "PR mode must record healthy production and pending GitHub review."
    Assert-True ($prPending.remote_url -eq
        "https://github.com/AngusChik/FINAL-PHARM.git" -and
        $prManifest.production_branch -eq "main" -and
        $prManifest.remote -eq "origin" -and
        $prManifest.remote_url -eq $prPending.remote_url) `
        "PR pending state and manifest must retain the guarded remote identity."
    $prPublishText = ($prCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($prPublishText -match
        'push origin a{40}:refs/heads/release/pharmacy-release-') `
        "PR mode must publish the exact commit to its release branch."
    Assert-True ($prPublishText -notmatch
        'push --atomic .*refs/heads/main:refs/heads/main') `
        "PR-mode publication must not update origin/main before review."
    Assert-True ($prPublishText -notmatch
        'push .*refs/tags/.+:refs/tags/') `
        "PR-mode publication must delay the public tag until finalization."

    # A superficially matching recovery marker must still satisfy the complete
    # rollback schema before a healthy pending release is allowed to clear it.
    $malformedRecoveryPath = Join-Path `
        $prFixture.Production ".runtime\production-recovery-required.json"
    $malformedRecovery = [ordered]@{
        schema_version = 1
        release_id = [string]$prPending.release_id
        failed_release_commit = [string]$prPending.commit
        previous_production_commit = [string]$prPending.previous_main_commit
        production_backup_path = [string]$prManifest.production_backup_path
        failure = "Release deployment is in progress after the final backup."
        rollback = [ordered]@{
            status = "backup_verified"
            database_restore_required = $false
            completed_utc = $null
            # Deliberately omit notes: matching identity is not enough.
        }
        created_utc = "2026-08-28T12:36:10Z"
    }
    [IO.File]::WriteAllText(
        $malformedRecoveryPath,
        ($malformedRecovery | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $malformedRecoveryCalls = New-Object Collections.Generic.List[object]
    $malformedRecoveryInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $malformedRecoveryCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40)
    $malformedRecoveryRejected = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $malformedRecoveryInvoker -ThrowOnError
    }
    catch {
        $malformedRecoveryRejected = $_.Exception.Message -match
            "Production recovery rollback journal is missing 'notes'"
    }
    Assert-True $malformedRecoveryRejected `
        "A malformed matching recovery marker must fail closed."
    Assert-True (Test-Path -LiteralPath $malformedRecoveryPath -PathType Leaf) `
        "Rejected recovery evidence must remain available for audited recovery."
    Assert-True (-not ($malformedRecoveryCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "Malformed recovery evidence must fail before a remote write."
    Remove-Item -LiteralPath $malformedRecoveryPath -Force

    $blockedCallCount = $prCalls.Count
    $prCheckBlocked = $false
    try {
        & $releaseScript -Action check -PullRequest `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $prInvoker -ThrowOnError
    }
    catch {
        $prCheckBlocked = $_.Exception.Message -match 'pull request is pending'
    }
    Assert-True $prCheckBlocked `
        "Another release check must be blocked while a production PR is pending."
    Assert-True ($prCalls.Count -eq $blockedCallCount) `
        "The durable PR block must fail before commands or mutations."

    $wrongUrlRejected = $false
    try {
        & $releaseScript -Action register-pr `
            -PullRequestUrl "https://github.com/another/repository/pull/321" `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $prInvoker -ThrowOnError
    }
    catch {
        $wrongUrlRejected = $_.Exception.Message -match 'must belong to'
    }
    Assert-True $wrongUrlRejected `
        "PR registration must reject URLs outside the configured repository."
    $prPending = Get-Content -LiteralPath $prPendingPath -Raw | ConvertFrom-Json
    Assert-True ($prPending.phase -eq "awaiting_pull_request") `
        "Rejected PR metadata must not advance the pending release."

    $registeredUrl = "https://github.com/anguschik/final-pharm/pull/321"
    $registerStart = $prCalls.Count
    & $releaseScript -Action register-pr -PullRequestUrl $registeredUrl `
        -DevelopmentWorktree $prFixture.Development `
        -WorkflowConfig $prFixture.Configuration `
        -StateDirectory $prFixture.State `
        -CommandInvoker $prInvoker -ThrowOnError
    $prPending = Get-Content -LiteralPath $prPendingPath -Raw | ConvertFrom-Json
    Assert-True ($prPending.phase -eq "awaiting_exact_main" -and
        $prPending.pull_request_number -eq 321 -and
        $prPending.pull_request_url -eq $registeredUrl) `
        "PR registration must persist the verified PR identity."
    $registerCalls = @($prCalls | Select-Object -Skip $registerStart)
    Assert-True (($registerCalls | ForEach-Object { $_.Arguments }) -join "`n" -match
        'pr view .*pull/321') `
        "Registration must verify GitHub PR metadata."
    Assert-True (-not ($registerCalls | Where-Object {
        $_.Arguments -match 'merge --ff-only|-Action (?:backup|stop|start)|push '
    })) "PR registration must not redeploy or write remote refs."

    $registeredPendingJson = Get-Content -LiteralPath $prPendingPath -Raw
    $registeredManifestJson = Get-Content -LiteralPath $prPending.manifest_path -Raw

    # Pending actions attest both live worktree remotes against the immutable
    # recorded URL. Redirecting either checkout must fail before GitHub access
    # or any remote ref mutation.
    $redirectedRemoteCalls = New-Object Collections.Generic.List[object]
    $redirectedRemoteInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $redirectedRemoteCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -DevelopmentRemoteUrl "https://github.com/attacker/redirected.git"
    $redirectedRemoteRejected = $false
    try {
        & $releaseScript -Action register-pr -PullRequestUrl $registeredUrl `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $redirectedRemoteInvoker -ThrowOnError
    }
    catch {
        $redirectedRemoteRejected = $_.Exception.Message -match
            "Remote 'origin'.*expected"
    }
    Assert-True $redirectedRemoteRejected `
        "A redirected live development remote must fail PR registration."
    Assert-True (-not ($redirectedRemoteCalls | Where-Object {
        $_.Arguments -match '^push |^pr view '
    })) "Remote redirection must fail before GitHub review or ref writes."

    # Pending control must run from the exact publisher implementation deployed
    # with production, not a locally mutated release script.
    $mutatedPublisherCalls = New-Object Collections.Generic.List[object]
    $mutatedPublisherInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $mutatedPublisherCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -PublisherDiffers $true
    $mutatedPublisherRejected = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $mutatedPublisherInvoker -ThrowOnError
    }
    catch {
        $mutatedPublisherRejected = $_.Exception.Message -match
            'running publisher differs from the production release commit'
    }
    Assert-True $mutatedPublisherRejected `
        "A mutated running publisher must fail pending finalization."
    Assert-True (-not ($mutatedPublisherCalls | Where-Object {
        $_.Arguments -match '^push |^pr view '
    })) "Publisher mutation must fail before GitHub review or ref writes."

    # Every action that could interpret release state must reject simultaneous
    # direct and pull-request journals before reading either untrusted record.
    $dualPendingPath = Join-Path $prFixture.State "sync-pending.json"
    foreach ($dualAction in @("status", "register-pr", "finalize-pr")) {
        [IO.File]::WriteAllText($dualPendingPath, "{}$([Environment]::NewLine)")
        $dualCallCount = $prCalls.Count
        $dualRejected = $false
        try {
            switch ($dualAction) {
                "status" {
                    & $releaseScript -Action status `
                        -DevelopmentWorktree $prFixture.Development `
                        -WorkflowConfig $prFixture.Configuration `
                        -StateDirectory $prFixture.State `
                        -CommandInvoker $prInvoker -ThrowOnError
                }
                "register-pr" {
                    & $releaseScript -Action register-pr `
                        -PullRequestUrl $registeredUrl `
                        -DevelopmentWorktree $prFixture.Development `
                        -WorkflowConfig $prFixture.Configuration `
                        -StateDirectory $prFixture.State `
                        -CommandInvoker $prInvoker -ThrowOnError
                }
                "finalize-pr" {
                    & $releaseScript -Action finalize-pr -ConfirmRelease `
                        -DevelopmentWorktree $prFixture.Development `
                        -WorkflowConfig $prFixture.Configuration `
                        -StateDirectory $prFixture.State `
                        -CommandInvoker $prInvoker -ThrowOnError
                }
            }
        }
        catch {
            $dualRejected = $_.Exception.Message -match
                'Both direct and pull-request pending states exist'
        }
        finally {
            Remove-Item -LiteralPath $dualPendingPath -Force -ErrorAction SilentlyContinue
            [IO.File]::WriteAllText($prPendingPath, $registeredPendingJson)
            [IO.File]::WriteAllText(
                [string]$prPending.manifest_path,
                $registeredManifestJson
            )
        }
        Assert-True $dualRejected `
            "Action $dualAction must reject conflicting pending journals."
        Assert-True ($prCalls.Count -eq $dualCallCount) `
            "Dual pending journals must fail before action $dualAction invokes commands."
    }

    # Both explicit review requirements and future/unknown GitHub decisions
    # fail closed before the exact-main push.
    foreach ($reviewDecision in @("REVIEW_REQUIRED", "FUTURE_UNKNOWN_VALUE")) {
        $reviewCalls = New-Object Collections.Generic.List[object]
        $reviewInvoker = New-FakeInvoker `
            -Fixture $prFixture -Calls $reviewCalls `
            -InitialProductionHead ("a" * 40) `
            -InitialRemoteReleaseCommit ("a" * 40) `
            -ReviewDecision $reviewDecision
        $reviewRejected = $false
        try {
            & $releaseScript -Action finalize-pr -ConfirmRelease `
                -DevelopmentWorktree $prFixture.Development `
                -WorkflowConfig $prFixture.Configuration `
                -StateDirectory $prFixture.State `
                -CommandInvoker $reviewInvoker -ThrowOnError
        }
        catch {
            $reviewRejected = $_.Exception.Message -match 'has not been approved'
        }
        finally {
            [IO.File]::WriteAllText($prPendingPath, $registeredPendingJson)
            [IO.File]::WriteAllText(
                [string]$prPending.manifest_path,
                $registeredManifestJson
            )
        }
        Assert-True $reviewRejected `
            "Review decision $reviewDecision must fail closed."
        Assert-True (-not ($reviewCalls | Where-Object {
            $_.Arguments -match '^push '
        })) "A rejected review decision must not update remote refs."
    }

    # Legacy StatusContext entries report their result through `state`; any
    # non-success value must block finalization just like a modern failed check.
    $failedLegacyStatus = @([pscustomobject]@{
        __typename = "StatusContext"
        state = "FAILURE"
    })
    $legacyStatusCalls = New-Object Collections.Generic.List[object]
    $legacyStatusInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $legacyStatusCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -StatusCheckRollup $failedLegacyStatus
    $legacyStatusRejected = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $legacyStatusInvoker -ThrowOnError
    }
    catch {
        $legacyStatusRejected = $_.Exception.Message -match
            'pull-request status is not successful'
    }
    finally {
        [IO.File]::WriteAllText($prPendingPath, $registeredPendingJson)
        [IO.File]::WriteAllText(
            [string]$prPending.manifest_path,
            $registeredManifestJson
        )
    }
    Assert-True $legacyStatusRejected `
        "A failed legacy StatusContext must block finalization."
    Assert-True (-not ($legacyStatusCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "A failed legacy StatusContext must not update remote refs."

    # A PR cannot be accepted as merged while origin/main is still the recorded
    # baseline. This catches a reset or a merge performed outside the controller.
    $resetMainCalls = New-Object Collections.Generic.List[object]
    $resetMainInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $resetMainCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -PullRequestState "MERGED"
    $resetMainRejected = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $resetMainInvoker -ThrowOnError
    }
    catch {
        $resetMainRejected = $_.Exception.Message -match 'MERGED, not OPEN'
    }
    finally {
        [IO.File]::WriteAllText($prPendingPath, $registeredPendingJson)
        [IO.File]::WriteAllText(
            [string]$prPending.manifest_path,
            $registeredManifestJson
        )
    }
    Assert-True $resetMainRejected `
        "A merged PR with origin/main reset to baseline must fail closed."
    Assert-True (-not ($resetMainCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "A merged/reset mismatch must not rewrite origin/main."

    # The immutable review branch is part of the synchronization transaction.
    # If it has moved, finalization must stop instead of overwriting it.
    $movedReviewCalls = New-Object Collections.Generic.List[object]
    $movedReviewInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $movedReviewCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("c" * 40)
    $movedReviewRejected = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $movedReviewInvoker -ThrowOnError
    }
    catch {
        $movedReviewRejected = $_.Exception.Message -match
            'remote review branch no longer resolves to the deployed commit'
    }
    Assert-True $movedReviewRejected `
        "A moved review branch must fail before exact-main synchronization."
    Assert-True (-not ($movedReviewCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "A moved review branch must never be overwritten."

    # Review readiness is checked again after exact refs synchronize. A check
    # that turns red in that narrow window must retain the durable post-sync
    # phase instead of declaring the release complete.
    $postSyncFailureCalls = New-Object Collections.Generic.List[object]
    $postSyncFailureInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $postSyncFailureCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -FailPostSyncChecks $true
    $postSyncFailureRejected = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $postSyncFailureInvoker -ThrowOnError
    }
    catch {
        $postSyncFailureRejected = $_.Exception.Message -match
            'pull-request status is not successful'
    }
    Assert-True $postSyncFailureRejected `
        "A failed post-sync check must prevent final completion."
    $postSyncFailurePending = Get-Content -LiteralPath $prPendingPath -Raw |
        ConvertFrom-Json
    Assert-True ($postSyncFailurePending.phase -eq
        "main_synced_pr_status_pending") `
        "A failed post-sync readiness check must retain the exact-ref phase."
    Assert-True (@($postSyncFailureCalls | Where-Object {
        $_.Arguments -match '^push --atomic'
    }).Count -eq 1) `
        "Post-sync readiness failure must occur only after one exact ref transaction."
    [IO.File]::WriteAllText($prPendingPath, $registeredPendingJson)
    [IO.File]::WriteAllText(
        [string]$prPending.manifest_path,
        $registeredManifestJson
    )

    # If exact main/tag synchronization succeeds before GitHub reports MERGED,
    # retain a distinct resumable phase and keep every later release blocked.
    $openFinalizeCalls = New-Object Collections.Generic.List[object]
    $openFinalizeInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $openFinalizeCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -MergePullRequestAfterMainPush $false
    $githubStatusPending = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $openFinalizeInvoker -ThrowOnError
    }
    catch {
        $githubStatusPending = $_.Exception.Message -match
            'has not yet marked the pull request merged'
    }
    Assert-True $githubStatusPending `
        "Exact main synchronization must wait for GitHub to report MERGED."
    Assert-True (Test-Path -LiteralPath $prPendingPath -PathType Leaf) `
        "A delayed GitHub PR status must retain the durable release block."
    $mainSyncedPending = Get-Content -LiteralPath $prPendingPath -Raw |
        ConvertFrom-Json
    Assert-True ($mainSyncedPending.phase -eq
        "main_synced_pr_status_pending") `
        "Exact refs with an open PR must use the post-sync status phase."
    $firstFinalizePushes = @($openFinalizeCalls | Where-Object {
        $_.Arguments -match '^push '
    }).Count

    $stillOpen = $false
    try {
        & $releaseScript -Action finalize-pr -ConfirmRelease `
            -DevelopmentWorktree $prFixture.Development `
            -WorkflowConfig $prFixture.Configuration `
            -StateDirectory $prFixture.State `
            -CommandInvoker $openFinalizeInvoker -ThrowOnError
    }
    catch {
        $stillOpen = $_.Exception.Message -match
            'has not yet marked the pull request merged'
    }
    Assert-True $stillOpen `
        "The post-sync phase must remain blocked while GitHub still reports OPEN."
    Assert-True (@($openFinalizeCalls | Where-Object {
        $_.Arguments -match '^push '
    }).Count -eq $firstFinalizePushes) `
        "A PR-status-only retry must not push exact refs again."

    $finalizeStart = $openFinalizeCalls.Count
    $mergedFinalizeCalls = New-Object Collections.Generic.List[object]
    $mergedFinalizeInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $mergedFinalizeCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteMain ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -InitialRemoteTagCommit ("a" * 40) `
        -InitialRemoteTagObject ("d" * 40) `
        -PullRequestState "MERGED"
    & $releaseScript -Action finalize-pr -ConfirmRelease `
        -DevelopmentWorktree $prFixture.Development `
        -WorkflowConfig $prFixture.Configuration `
        -StateDirectory $prFixture.State `
        -CommandInvoker $mergedFinalizeInvoker -ThrowOnError
    Assert-True (-not (Test-Path -LiteralPath $prPendingPath)) `
        "GitHub MERGED confirmation must clear the exact PR release block."
    $prManifest = Get-Content -LiteralPath $prPending.manifest_path -Raw |
        ConvertFrom-Json
    Assert-True ($prManifest.synchronization.status -eq "complete") `
        "Finalization must mark the original release manifest complete."
    $finalizeText = ($openFinalizeCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($finalizeText -match
        'push --atomic --force-with-lease=refs/heads/main:b{40} --force-with-lease=refs/heads/release/[^ ]+:a{40} origin') `
        "Finalization must lease both main and the immutable review branch."
    Assert-True ($finalizeText -match
        'a{40}:refs/heads/main.*a{40}:refs/heads/release/.*d{40}:refs/tags/') `
        "Finalization must atomically publish exact main, review branch, and annotated tag."
    Assert-True (-not ($openFinalizeCalls + $mergedFinalizeCalls | Where-Object {
        $_.Arguments -match 'manage\.py|bundle create|merge --ff-only|-Action (?:backup|stop|start)'
    })) "Finalization must not rerun deployment or database operations."
    foreach ($finalizeGuardedCall in @(
        $openFinalizeCalls + $mergedFinalizeCalls | Where-Object {
            $_.Arguments -match '-Action status(?: |$)|^push --atomic'
        }
    )) {
        Assert-True $finalizeGuardedCall.ReleaseLockHeld `
            "Final PR health and Git synchronization must hold the production release lock."
    }

    # A crash after writing a complete manifest but before removing its pending
    # journal must be recognized and finalized without another ref mutation.
    $idempotentPending = $mainSyncedPending
    $idempotentPending.phase = "main_synced_pr_status_pending"
    $idempotentPending.last_error = "simulated crash before pending cleanup"
    [IO.File]::WriteAllText(
        $prPendingPath,
        ($idempotentPending | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $idempotentCalls = New-Object Collections.Generic.List[object]
    $idempotentInvoker = New-FakeInvoker `
        -Fixture $prFixture -Calls $idempotentCalls `
        -InitialProductionHead ("a" * 40) `
        -InitialRemoteMain ("a" * 40) `
        -InitialRemoteReleaseCommit ("a" * 40) `
        -InitialRemoteTagCommit ("a" * 40) `
        -InitialRemoteTagObject ("d" * 40) `
        -PullRequestState "MERGED"
    & $releaseScript -Action finalize-pr -ConfirmRelease `
        -DevelopmentWorktree $prFixture.Development `
        -WorkflowConfig $prFixture.Configuration `
        -StateDirectory $prFixture.State `
        -CommandInvoker $idempotentInvoker -ThrowOnError
    Assert-True (-not (Test-Path -LiteralPath $prPendingPath)) `
        "A complete-manifest crash window must clear its stale pending state."
    Assert-True (-not ($idempotentCalls | Where-Object {
        $_.Arguments -match '^push '
    })) "Idempotent complete-manifest recovery must not push remote refs again."

    # A crash after PR-mode production health was persisted but before the
    # review journal was installed must continue only the remote review phase.
    # The live database may already have served traffic, so no redeploy or
    # automatic rollback is safe at this point.
    $postHealthFixture = New-ReleaseFixture "post-health-pr-recovery"
    $fixtures.Add($postHealthFixture.Root)
    $postHealthState = Initialize-InterruptedReleaseFixture $postHealthFixture
    $postHealthManifest = Get-Content `
        -LiteralPath $postHealthState.ManifestPath -Raw | ConvertFrom-Json
    $postHealthManifest.checks.production_health = "passed"
    $postHealthManifest.deployment.status = "healthy"
    $postHealthManifest.deployment.completed_utc = "2026-08-28T12:39:00Z"
    $postHealthManifest.synchronization | Add-Member `
        -NotePropertyName mode -NotePropertyValue "pull_request"
    $postHealthManifest.synchronization | Add-Member `
        -NotePropertyName review_branch `
        -NotePropertyValue "release/$($postHealthState.ReleaseId)"
    $postHealthManifest.synchronization | Add-Member `
        -NotePropertyName pull_request_number -NotePropertyValue $null
    $postHealthManifest.synchronization | Add-Member `
        -NotePropertyName pull_request_url -NotePropertyValue $null
    [IO.File]::WriteAllText(
        $postHealthState.ManifestPath,
        ($postHealthManifest | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $postHealthCalls = New-Object Collections.Generic.List[object]
    $postHealthInvoker = New-FakeInvoker `
        -Fixture $postHealthFixture -Calls $postHealthCalls `
        -InitialProductionHead ("a" * 40)
    & $releaseScript -Action publish -PullRequest -ConfirmRelease `
        -DevelopmentWorktree $postHealthFixture.Development `
        -WorkflowConfig $postHealthFixture.Configuration `
        -StateDirectory $postHealthFixture.State `
        -CommandInvoker $postHealthInvoker -ThrowOnError
    Assert-True (-not (Test-Path -LiteralPath $postHealthState.RecoveryPath)) `
        "Post-health PR recovery must clear the obsolete recovery journal."
    $postHealthPendingPath = Join-Path `
        $postHealthFixture.State "pull-request-pending.json"
    Assert-True (Test-Path -LiteralPath $postHealthPendingPath -PathType Leaf) `
        "Post-health PR recovery must install a durable review block."
    $postHealthPending = Get-Content -LiteralPath $postHealthPendingPath -Raw |
        ConvertFrom-Json
    Assert-True ($postHealthPending.phase -eq "awaiting_pull_request") `
        "Post-health recovery must advance only to pull-request creation."
    $postHealthText = ($postHealthCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($postHealthText -match '-Action status' -and
        $postHealthText -match 'push origin a{40}:refs/heads/release/') `
        "Post-health recovery must recheck health and publish only the review branch."
    Assert-True ($postHealthText -notmatch
        'manage\.py|tag --annotate|bundle create|merge --ff-only|-Action (?:backup|stop|start)|reset --hard|database-restore') `
        "Post-health recovery must not redeploy, restore, or recreate release artifacts."

    # A crash after deployment startup begins but before health state is
    # persisted must resume only the exact journaled release. Corrupt backup
    # evidence fails closed before any process or Git command is invoked.
    $recoveryFixture = New-ReleaseFixture "interrupted-recovery"
    $fixtures.Add($recoveryFixture.Root)
    [IO.File]::WriteAllText(
        (Join-Path $recoveryFixture.Production "scripts\production.ps1"),
        'param([switch]$NonInteractive, [string]$ReleaseToken = "")'
    )
    $releaseId = "pharmacy-release-20260828T120000Z-aaaaaaaaaaaa"
    $releaseDirectory = Join-Path $recoveryFixture.State "releases\$releaseId"
    $backupDirectory = Join-Path $recoveryFixture.Production "backups\database"
    New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
    $bundlePath = Join-Path $releaseDirectory "$releaseId.bundle"
    $backupPath = Join-Path $backupDirectory "pharmacy-20260828-120000-manual.dump"
    [IO.File]::WriteAllText($bundlePath, "verified interrupted bundle")
    [IO.File]::WriteAllText($backupPath, "verified interrupted database")
    $bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
    $backupHash = (Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash
    $manifestPath = Join-Path $releaseDirectory "manifest.json"
    $interruptedManifest = [ordered]@{
        schema_version = 1
        release_id = $releaseId
        release_tag = $releaseId
        source_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        previous_production_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        production_branch = "main"
        remote = "origin"
        remote_url = "https://github.com/AngusChik/FINAL-PHARM.git"
        production_worktree = $recoveryFixture.Production
        bundle_path = $bundlePath
        bundle_sha256 = $bundleHash
        production_backup_path = $backupPath
        checks = [ordered]@{
            repository = "passed"
            candidate = "passed"
            backup = "passed"
            production_health = "pending"
        }
        deployment = [ordered]@{
            status = "starting"
            completed_utc = $null
            rollback = $null
        }
        synchronization = [ordered]@{
            status = "not_started"
            completed_utc = $null
            error = $null
        }
    }
    [IO.File]::WriteAllText(
        $manifestPath,
        ($interruptedManifest | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    $recoveryPath = Join-Path `
        $recoveryFixture.Production ".runtime\production-recovery-required.json"
    $interruptedJournal = [ordered]@{
        schema_version = 1
        release_id = $releaseId
        failed_release_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        previous_production_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        production_backup_path = $backupPath
        failure = "Release deployment is in progress after the final backup."
        rollback = [ordered]@{
            status = "backup_verified"
            database_restore_required = $false
            completed_utc = $null
            notes = @("Verified rollback backup: $backupPath")
        }
        created_utc = "2026-08-28T12:00:00Z"
    }
    [IO.File]::WriteAllText(
        $recoveryPath,
        ($interruptedJournal | ConvertTo-Json -Depth 12) + [Environment]::NewLine
    )
    [IO.File]::WriteAllText(
        "$backupPath.sha256",
        ("0" * 64) + "  $(Split-Path -Leaf $backupPath)" + [Environment]::NewLine
    )
    $recoveryCalls = New-Object Collections.Generic.List[object]
    $recoveryInvoker = New-FakeInvoker `
        -Fixture $recoveryFixture `
        -Calls $recoveryCalls `
        -InitialProductionHead "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $corruptBackupRejected = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $recoveryFixture.Development `
            -WorkflowConfig $recoveryFixture.Configuration `
            -StateDirectory $recoveryFixture.State `
            -CommandInvoker $recoveryInvoker -ThrowOnError
    }
    catch {
        $corruptBackupRejected = $_.Exception.Message -match 'checksum verification failed'
    }
    Assert-True $corruptBackupRejected `
        "Interrupted recovery must fail closed on a corrupt final backup."
    Assert-True ($recoveryCalls.Count -eq 0) `
        "Corrupt recovery evidence must fail before Git or process commands."
    Assert-True (Test-Path -LiteralPath $recoveryPath -PathType Leaf) `
        "Rejected interrupted recovery must retain its durable journal."

    [IO.File]::WriteAllText(
        "$backupPath.sha256",
        "$backupHash  $(Split-Path -Leaf $backupPath)$([Environment]::NewLine)"
    )
    & $releaseScript -Action publish -ConfirmRelease `
        -DevelopmentWorktree $recoveryFixture.Development `
        -WorkflowConfig $recoveryFixture.Configuration `
        -StateDirectory $recoveryFixture.State `
        -CommandInvoker $recoveryInvoker `
        -Clock { [DateTimeOffset]::Parse("2026-08-28T12:37:30Z") } `
        -ThrowOnError

    Assert-True (-not (Test-Path -LiteralPath $recoveryPath)) `
        "Healthy interrupted recovery must clear the recovery journal."
    Assert-True (-not (Test-Path -LiteralPath `
        (Join-Path $recoveryFixture.State "sync-pending.json"))) `
        "Successful interrupted synchronization must clear pending state."
    $recoveredManifest = Get-Content -LiteralPath $manifestPath -Raw |
        ConvertFrom-Json
    Assert-True ($recoveredManifest.deployment.status -eq "healthy" -and
        $recoveredManifest.checks.production_health -eq "passed") `
        "Interrupted recovery must persist healthy deployment state before sync."
    Assert-True ($recoveredManifest.synchronization.status -eq "complete") `
        "Interrupted recovery must complete the exact atomic synchronization."
    $recoveryText = ($recoveryCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($recoveryText -match '-Action start' -and
        $recoveryText -match '-Action status' -and
        $recoveryText -match 'push --atomic') `
        "Interrupted recovery must restart, health-check, and atomically push."
    Assert-True ($recoveryText -notmatch
        'manage\.py|tag --annotate|bundle create|merge --ff-only|-Action backup') `
        "Interrupted recovery must not recreate checks, artifacts, backup, or deployment."
    $recoveryPushCall = $recoveryCalls | Where-Object {
        $_.Arguments -match '^push --atomic'
    } | Select-Object -First 1
    Assert-True $recoveryPushCall.ReleaseLockHeld `
        "Interrupted recovery push must remain inside the production release gate."
    Assert-True $recoveryPushCall.SyncPendingExists `
        "Interrupted recovery must persist sync intent before its atomic push."
    Assert-True (-not $recoveryPushCall.RecoveryBlockExists) `
        "Interrupted recovery must clear its block only after durable sync intent."

    # If the resumed candidate cannot become healthy, the recovery path must
    # restore the exact final backup and prior production commit, prove the
    # prior release healthy, and never synchronize the failed candidate.
    $recoveryRollbackFixture = New-ReleaseFixture "interrupted-rollback"
    $fixtures.Add($recoveryRollbackFixture.Root)
    $recoveryRollbackState = Initialize-InterruptedReleaseFixture `
        $recoveryRollbackFixture
    $recoveryRollbackCalls = New-Object Collections.Generic.List[object]
    $recoveryRollbackInvoker = New-FakeInvoker `
        -Fixture $recoveryRollbackFixture `
        -Calls $recoveryRollbackCalls `
        -FailFirstStart $true `
        -InitialProductionHead "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $interruptedStartFailed = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $recoveryRollbackFixture.Development `
            -WorkflowConfig $recoveryRollbackFixture.Configuration `
            -StateDirectory $recoveryRollbackFixture.State `
            -CommandInvoker $recoveryRollbackInvoker `
            -Clock { [DateTimeOffset]::Parse("2026-08-28T12:38:30Z") } `
            -ThrowOnError
    }
    catch {
        $interruptedStartFailed = `
            $_.Exception.Message -match 'Interrupted production recovery failed'
    }
    Assert-True $interruptedStartFailed `
        "A resumed startup failure must fail through interrupted recovery."
    $recoveryRollbackManifest = Get-Content `
        -LiteralPath $recoveryRollbackState.ManifestPath -Raw |
        ConvertFrom-Json
    Assert-True ($recoveryRollbackManifest.deployment.status -eq "failed" -and
        $recoveryRollbackManifest.deployment.rollback.status -eq "healthy") `
        "Interrupted startup failure must record a healthy completed rollback."
    Assert-True (
        [bool]$recoveryRollbackManifest.deployment.rollback.database_restore_required
    ) "Interrupted startup failure must restore the verified final backup."
    Assert-True (-not (Test-Path -LiteralPath $recoveryRollbackState.RecoveryPath)) `
        "Healthy interrupted rollback must clear the recovery block."
    $recoveryRollbackText = @(
        $recoveryRollbackCalls | ForEach-Object { $_.Arguments }
    ) -join "`n"
    Assert-True ($recoveryRollbackText -match 'reset --hard b{40}' -and
        $recoveryRollbackText -match 'database-restore\.ps1.*-BackupPath') `
        "Interrupted rollback must restore the prior code and exact backup."
    Assert-True ($recoveryRollbackText -notmatch 'push --atomic') `
        "Interrupted startup failure must never synchronize the candidate."

    # A candidate startup failure after deployment must restore both code and
    # the verified final backup, then prove the previous release healthy.
    $rollbackFixture = New-ReleaseFixture "rollback"
    $fixtures.Add($rollbackFixture.Root)
    $rollbackCalls = New-Object Collections.Generic.List[object]
    $rollbackInvoker = New-FakeInvoker `
        $rollbackFixture $rollbackCalls $false $false $true
    $releaseFailed = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $rollbackFixture.Development `
            -WorkflowConfig $rollbackFixture.Configuration `
            -StateDirectory $rollbackFixture.State `
            -CommandInvoker $rollbackInvoker `
            -Clock { [DateTimeOffset]::Parse("2026-08-28T12:35:56Z") } `
            -ThrowOnError
    }
    catch {
        $releaseFailed = $_.Exception.Message -match 'failed before synchronization'
    }
    Assert-True $releaseFailed "Candidate startup failure must fail the release."

    $rollbackManifestPath = Get-ChildItem `
        -LiteralPath (Join-Path $rollbackFixture.State "releases") `
        -Recurse -File -Filter "manifest.json" |
        Select-Object -ExpandProperty FullName -First 1
    $rollbackManifest = Get-Content -LiteralPath $rollbackManifestPath -Raw |
        ConvertFrom-Json
    Assert-True ($rollbackManifest.deployment.status -eq "failed") `
        "Failed candidate must be recorded as a failed deployment."
    Assert-True ($rollbackManifest.deployment.rollback.status -eq "healthy") `
        "Rollback must explicitly record restored production health."
    Assert-True ([bool]$rollbackManifest.deployment.rollback.database_restore_required) `
        "A startup-attempt failure must restore the verified database backup."
    Assert-True (($rollbackManifest.deployment.rollback.notes -join "`n") -match
        'previous production restored and healthy') `
        "Rollback notes must record the successful health proof."

    $rollbackText = ($rollbackCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($rollbackText -match 'reset --hard b{40}') `
        "Rollback must reset production to the previous main commit."
    Assert-True ($rollbackText -match 'database-restore\.ps1.*-BackupPath') `
        "Rollback must restore the exact verified final backup."
    Assert-True ($rollbackText -notmatch 'push --atomic') `
        "A failed deployment must never push main or its release tag."
    $rollbackRecoveryPath = Join-Path `
        $rollbackFixture.Production ".runtime\production-recovery-required.json"
    Assert-True (-not (Test-Path -LiteralPath $rollbackRecoveryPath)) `
        "A healthy rollback must clear the durable production recovery block."

    # If rollback cannot first prove that production stopped, it must leave the
    # candidate code/data untouched and retain a durable manual-recovery block.
    $blockedFixture = New-ReleaseFixture "blocked-rollback"
    $fixtures.Add($blockedFixture.Root)
    $blockedCalls = New-Object Collections.Generic.List[object]
    $blockedInvoker = New-FakeInvoker `
        -Fixture $blockedFixture `
        -Calls $blockedCalls `
        -FailFirstStart $true `
        -FailRollbackStop $true
    $blockedReleaseFailed = $false
    try {
        & $releaseScript -Action publish -PullRequest -ConfirmRelease `
            -DevelopmentWorktree $blockedFixture.Development `
            -WorkflowConfig $blockedFixture.Configuration `
            -StateDirectory $blockedFixture.State `
            -CommandInvoker $blockedInvoker `
            -Clock { [DateTimeOffset]::Parse("2026-08-28T12:36:56Z") } `
            -ThrowOnError
    }
    catch {
        $blockedReleaseFailed = `
            $_.Exception.Message -match 'failed before synchronization'
    }
    Assert-True $blockedReleaseFailed `
        "An unsafe rollback must fail the release before synchronization."

    $blockedManifestPath = Get-ChildItem `
        -LiteralPath (Join-Path $blockedFixture.State "releases") `
        -Recurse -File -Filter "manifest.json" |
        Select-Object -ExpandProperty FullName -First 1
    $blockedManifest = Get-Content -LiteralPath $blockedManifestPath -Raw |
        ConvertFrom-Json
    Assert-True ($blockedManifest.deployment.rollback.status -eq "failed") `
        "A rollback stop failure must be recorded as failed."
    Assert-True (($blockedManifest.deployment.rollback.notes -join "`n") -match
        'rollback aborted before code or database changes') `
        "Unsafe rollback must record that no code or database recovery ran."
    $blockedText = ($blockedCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($blockedText -notmatch 'reset --hard|database-restore\.ps1|push --atomic') `
        "Rollback must not reset, restore, or push after its stop proof fails."
    $blockedRecoveryPath = Join-Path `
        $blockedFixture.Production ".runtime\production-recovery-required.json"
    Assert-True (Test-Path -LiteralPath $blockedRecoveryPath -PathType Leaf) `
        "Unsafe rollback must retain a durable production recovery block."
    $blockedRecovery = Get-Content -LiteralPath $blockedRecoveryPath -Raw |
        ConvertFrom-Json
    Assert-True ($blockedRecovery.rollback.status -eq "failed") `
        "The durable recovery block must carry the failed rollback result."

    Write-Host "publish release script tests passed"
}
finally {
    foreach ($fixtureRoot in $fixtures) {
        $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
        $resolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if ($resolvedFixture.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase) -and
            (Split-Path $resolvedFixture -Leaf).StartsWith("pharmacy-release-")) {
            Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
