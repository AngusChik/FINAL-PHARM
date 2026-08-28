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
$requiredContracts = @(
    '\[ValidateSet\("check", "publish", "status"\)\]',
    '\[switch\]\$ConfirmRelease',
    '\[switch\]\$DryRun',
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
    'function Assert-SyncPendingState',
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
    'function Invoke-WithProductionMutationLocks'
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
    [bool]$FailRollbackStop = $false
) {
    $developmentPath = [IO.Path]::GetFullPath($Fixture.Development).TrimEnd('\', '/')
    $productionPath = [IO.Path]::GetFullPath($Fixture.Production).TrimEnd('\', '/')
    $developmentCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $productionCommit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    $sharedGitDirectory = Join-Path $Fixture.Root "shared.git"
    $markerDirectory = Join-Path $Fixture.Root "markers"
    $releaseLockPath = Join-Path $Fixture.Production ".runtime\production-release.lock"
    $releaseOwnerPath = Join-Path $Fixture.Production ".runtime\production-release.owner.json"
    $recoveryBlockPath = Join-Path `
        $Fixture.Production ".runtime\production-recovery-required.json"
    $syncPendingPath = Join-Path $Fixture.State "sync-pending.json"
    $fakeState = @{
        StartCalls = 0
        StopCalls = 0
        ProductionHead = if ($InitialProductionHead) {
            $InitialProductionHead
        }
        else { $productionCommit }
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
                $output = "https://github.com/AngusChik/FINAL-PHARM.git"
            }
            elseif ($argumentText -eq "rev-parse refs/remotes/origin/development") {
                $output = $developmentCommit
            }
            elseif ($argumentText -eq "rev-parse refs/remotes/origin/main") {
                $output = $productionCommit
            }
            elseif ($argumentText -like "rev-parse refs/tags/*^{}") {
                $output = $developmentCommit
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
                $output = Join-Path $Fixture.Root "verified-final-backup.dump"
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

$fixtures = New-Object Collections.Generic.List[string]
try {
    # `check` may fetch and run tests, but must not tag, bundle, deploy, or push.
    $checkFixture = New-ReleaseFixture "check"
    $fixtures.Add($checkFixture.Root)
    $checkCalls = New-Object Collections.Generic.List[object]
    $checkInvoker = New-FakeInvoker $checkFixture $checkCalls
    & $releaseScript -Action check `
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
    & $releaseScript -Action publish -DryRun `
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
        & $releaseScript -Action check `
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

    # Once production is healthy, push failure must not roll back; it must leave
    # a durable sync-pending record with the immutable release identity.
    $pendingFixture = New-ReleaseFixture "pending"
    $fixtures.Add($pendingFixture.Root)
    $pendingCalls = New-Object Collections.Generic.List[object]
    $pendingInvoker = New-FakeInvoker $pendingFixture $pendingCalls $false $true
    $pushFailed = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
            -DevelopmentWorktree $pendingFixture.Development `
            -WorkflowConfig $pendingFixture.Configuration `
            -StateDirectory $pendingFixture.State `
            -CommandInvoker $pendingInvoker `
            -Clock { [DateTimeOffset]::Parse("2026-08-28T12:34:56Z") } `
            -ThrowOnError
    }
    catch {
        $pushFailed = $_.Exception.Message -match 'synchronization is pending'
    }
    Assert-True $pushFailed "A failed atomic push must report synchronization pending."

    $syncPendingPath = Join-Path $pendingFixture.State "sync-pending.json"
    Assert-True (Test-Path -LiteralPath $syncPendingPath) `
        "Push failure must persist sync-pending.json."
    $pending = Get-Content -LiteralPath $syncPendingPath -Raw | ConvertFrom-Json
    Assert-True ($pending.commit -eq "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") `
        "Sync-pending must identify the deployed development commit."
    Assert-True ($pending.release_id -match '^pharmacy-release-20260828T123456Z-') `
        "Release identity must be deterministic from UTC time and commit."
    Assert-True (Test-Path -LiteralPath $pending.manifest_path) `
        "Sync-pending must point to a durable manifest."
    $manifest = Get-Content -LiteralPath $pending.manifest_path -Raw | ConvertFrom-Json
    Assert-True ($manifest.deployment.status -eq "healthy") `
        "Manifest must record healthy production before synchronization."
    Assert-True ($manifest.synchronization.status -eq "pending") `
        "Manifest must record pending synchronization."
    $pendingRecoveryPath = Join-Path `
        $pendingFixture.Production ".runtime\production-recovery-required.json"
    Assert-True (-not (Test-Path -LiteralPath $pendingRecoveryPath)) `
        "Healthy production must not remain blocked after a push-only failure."

    $pendingText = ($pendingCalls | ForEach-Object { $_.Arguments }) -join "`n"
    Assert-True ($pendingText -match 'tag --annotate') `
        "Publish must create a local annotated release tag."
    Assert-True ($pendingText -match 'bundle create') `
        "Publish must create a local release bundle."
    Assert-True ($pendingText -match '-Action backup') `
        "Publish must create a verified final production backup."
    Assert-True ($pendingText -match 'merge --ff-only') `
        "Publish must fast-forward the production worktree."
    Assert-True ($pendingText -match 'push --atomic') `
        "Publish must synchronize the main branch and tag atomically."
    $pushCommand = $pendingCalls | Where-Object { $_.Arguments -match '^push --atomic' } |
        Select-Object -ExpandProperty Arguments -First 1
    Assert-True ($pushCommand -notmatch 'development') `
        "Successful release must publish main and the tag, not the local development branch."
    Assert-True ($pendingText -notmatch 'reset --hard') `
        "A push-only failure must not roll healthy production back."
    $initialPushCall = $pendingCalls | Where-Object {
        $_.Arguments -match '^push --atomic'
    } | Select-Object -First 1
    Assert-True $initialPushCall.ReleaseLockHeld `
        "Initial atomic push must remain inside the production release gate."
    Assert-True $initialPushCall.SyncPendingExists `
        "Durable synchronization intent must exist before the initial push."
    Assert-True (-not $initialPushCall.RecoveryBlockExists) `
        "The recovery block must clear only after sync intent is durable and before push."
    $pendingStopCall = $pendingCalls | Where-Object {
        $_.Arguments -match '-Action stop(?: |$)'
    } | Select-Object -First 1
    $pendingBackupCall = $pendingCalls | Where-Object {
        $_.Arguments -match '-Action backup(?: |$)'
    } | Select-Object -First 1
    $pendingStartCall = $pendingCalls | Where-Object {
        $_.Arguments -match '-Action start(?: |$)'
    } | Select-Object -First 1
    $pendingStatusCall = $pendingCalls | Where-Object {
        $_.Arguments -match '-Action status(?: |$)'
    } | Select-Object -First 1
    foreach ($legacyCall in @($pendingStopCall, $pendingBackupCall)) {
        Assert-True ($legacyCall.Arguments -notmatch '-NoBrowser -NonInteractive') `
            "The pre-release controller must not receive an unknown -NonInteractive flag."
        Assert-True ($legacyCall.Arguments -notmatch '(?:^| )-ReleaseToken(?: |$)') `
            "The pre-release controller must not receive an unknown release token flag."
    }
    foreach ($candidateCall in @($pendingStartCall, $pendingStatusCall)) {
        $tokenMatch = [regex]::Match(
            $candidateCall.Arguments,
            '-NoBrowser -NonInteractive -ReleaseToken ([0-9a-f]{32})(?: |$)'
        )
        Assert-True ($tokenMatch.Success) `
            "The upgraded controller must receive the active release token."
        Assert-True ($candidateCall.ReleaseLockHeld) `
            "The publisher must retain the OS release lock during candidate control."
        Assert-True ($candidateCall.ReleaseOwnerToken -eq $tokenMatch.Groups[1].Value) `
            "The child controller token must match the active lock owner metadata."
    }
    foreach ($guardedCall in @(
        $pendingStopCall,
        $pendingBackupCall,
        ($pendingCalls | Where-Object {
            $_.Arguments -match '^merge --ff-only'
        } | Select-Object -First 1),
        $pendingStartCall,
        $pendingStatusCall,
        $initialPushCall
    )) {
        Assert-True ($guardedCall.ReleaseLockHeld) `
            "The shared production release lock must cover the entire deployment transaction."
        Assert-True ($guardedCall.ReleaseOwnerToken -match '^[0-9a-f]{32}$') `
            "The active release lock must publish GUID owner metadata."
    }
    $releaseOwnerPath = Join-Path `
        $pendingFixture.Production ".runtime\production-release.owner.json"
    Assert-True (-not (Test-Path -LiteralPath $releaseOwnerPath)) `
        "Publisher must remove release owner metadata after the transaction."
    $releaseLockPath = Join-Path `
        $pendingFixture.Production ".runtime\production-release.lock"
    $releasedProbe = $null
    try {
        $releasedProbe = [IO.File]::Open(
            $releaseLockPath,
            [IO.FileMode]::Open,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    }
    finally {
        if ($null -ne $releasedProbe) { $releasedProbe.Dispose() }
    }
    $pendingArguments = @($pendingCalls | ForEach-Object { $_.Arguments })
    $stopIndex = [Array]::FindIndex(
        $pendingArguments,
        [Predicate[string]]{ param($value) $value -match '-Action stop' }
    )
    $backupIndex = [Array]::FindIndex(
        $pendingArguments,
        [Predicate[string]]{ param($value) $value -match '-Action backup' }
    )
    $mergeIndex = [Array]::FindIndex(
        $pendingArguments,
        [Predicate[string]]{ param($value) $value -match 'merge --ff-only' }
    )
    Assert-True ($stopIndex -ge 0 -and $backupIndex -gt $stopIndex -and
        $mergeIndex -gt $backupIndex) `
        "Publish order must be stop, verified final backup, then code deployment."

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

    # A candidate startup failure after deployment must restore both code and
    # the verified final backup, then prove the previous release healthy.
    $rollbackFixture = New-ReleaseFixture "rollback"
    $fixtures.Add($rollbackFixture.Root)
    $rollbackCalls = New-Object Collections.Generic.List[object]
    $rollbackInvoker = New-FakeInvoker `
        $rollbackFixture $rollbackCalls $false $false $true
    $releaseFailed = $false
    try {
        & $releaseScript -Action publish -ConfirmRelease `
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
        & $releaseScript -Action publish -ConfirmRelease `
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
