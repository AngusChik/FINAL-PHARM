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

function Repair-DuplicatePathEnvironment {
    # Windows launch contexts can expose both Path and PATH. Start-Process
    # builds a case-insensitive environment dictionary and otherwise fails
    # before the hidden controller can start.
    $processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $processPath) {
        $processPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

function Invoke-ReleaseProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [switch]$Mutation,
        [switch]$AllowFailure,
        [switch]$ControllerProcess
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
    elseif ($ControllerProcess) {
        # A production controller starts long-lived service descendants. Both
        # a normal capture pipeline and Start-Process stream redirection create
        # inheritable handles that can keep the publisher/host attached until
        # production stops. Launch an unredirected hidden wrapper instead. The
        # wrapper executes production.ps1 in-process and writes its result to a
        # durable file; the publisher waits only on that direct wrapper handle.
        $controllerLogDirectory = Join-Path $WorkingDirectory "logs\release-controller"
        New-Item -ItemType Directory -Force -Path $controllerLogDirectory | Out-Null
        $captureId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([Guid]::NewGuid().ToString('N'))"
        $resultPath = Join-Path $controllerLogDirectory "$captureId.result.log"
        $fileIndex = [Array]::IndexOf($Arguments, "-File")
        if ($FilePath -notmatch '(?i)(?:^|\\)powershell(?:\.exe)?$' -or
            $fileIndex -lt 0 -or $fileIndex + 1 -ge $Arguments.Count) {
            throw "ControllerProcess requires a PowerShell -File invocation."
        }
        $scriptPath = [string]$Arguments[$fileIndex + 1]
        $scriptArguments = if ($fileIndex + 2 -lt $Arguments.Count) {
            [string[]]$Arguments[($fileIndex + 2)..($Arguments.Count - 1)]
        }
        else { [string[]]@() }
        $controllerParameters = [ordered]@{}
        for ($index = 0; $index -lt $scriptArguments.Count; $index++) {
            $token = [string]$scriptArguments[$index]
            if ($token -notmatch '^--?([A-Za-z][A-Za-z0-9_-]*)$') {
                throw "ControllerProcess received an invalid parameter token: $token"
            }
            $name = $Matches[1]
            if ($controllerParameters.Contains($name)) {
                throw "ControllerProcess received duplicate parameter: $name"
            }
            if ($index + 1 -lt $scriptArguments.Count -and
                [string]$scriptArguments[$index + 1] -notmatch '^--?[A-Za-z]') {
                $controllerParameters[$name] = [string]$scriptArguments[$index + 1]
                $index += 1
            }
            else {
                $controllerParameters[$name] = $true
            }
        }
        $payload = [ordered]@{
            script_path = $scriptPath
            parameters = $controllerParameters
            working_directory = $WorkingDirectory
            result_path = $resultPath
        } | ConvertTo-Json -Compress
        $payloadBase64 = [Convert]::ToBase64String(
            [Text.Encoding]::UTF8.GetBytes($payload)
        )
        $wrapperSource = @"
`$payloadJson = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String('$payloadBase64')
)
`$request = `$payloadJson | ConvertFrom-Json
`$controllerParameters = @{}
foreach (`$property in `$request.parameters.PSObject.Properties) {
    `$controllerParameters[`$property.Name] = `$property.Value
}
Set-Location -LiteralPath ([string]`$request.working_directory)
& ([string]`$request.script_path) @controllerParameters *> ([string]`$request.result_path)
`$controllerSucceeded = `$?
`$controllerExitCode = `$LASTEXITCODE
if (-not `$controllerSucceeded) {
    if (`$null -ne `$controllerExitCode -and [int]`$controllerExitCode -ne 0) {
        exit ([int]`$controllerExitCode)
    }
    exit 1
}
exit 0
"@
        $encodedCommand = [Convert]::ToBase64String(
            [Text.Encoding]::Unicode.GetBytes($wrapperSource)
        )
        Repair-DuplicatePathEnvironment
        $controller = Start-Process -FilePath $FilePath `
            -ArgumentList "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encodedCommand" `
            -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
        $controller.WaitForExit()
        $nativeOutput = @()
        if (Test-Path -LiteralPath $resultPath) {
            $nativeOutput += @(Get-Content -LiteralPath $resultPath)
        }
        $result = New-ProcessResult $controller.ExitCode $nativeOutput
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
        -AllowFailure:$AllowFailure -ControllerProcess
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

function Assert-ObjectProperties(
    [object]$Value,
    [string[]]$Names,
    [string]$Label
) {
    if ($null -eq $Value) { throw "$Label contains no object data." }
    foreach ($name in $Names) {
        if (-not ($Value.PSObject.Properties.Name -contains $name)) {
            throw "$Label is missing '$name'."
        }
    }
}

function Resolve-ValidatedChildFile(
    [string]$Path,
    [string]$Root,
    [string]$Label
) {
    Assert-ExistingDirectory $Root "$Label root"
    Assert-ExistingFile $Path $Label
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\', '/')
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolvedPath.StartsWith(
        $resolvedRoot + '\',
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label is outside its guarded directory: $resolvedPath"
    }
    return $resolvedPath
}

function Assert-FileChecksumSidecar([string]$Path, [string]$Label) {
    $checksumPath = "$Path.sha256"
    Assert-ExistingFile $checksumPath "$Label checksum"
    $checksumLine = ([string](Get-Content -LiteralPath $checksumPath -TotalCount 1)).Trim()
    $match = [regex]::Match($checksumLine, '^([a-fA-F0-9]{64})\s+(.+)$')
    if (-not $match.Success -or
        $match.Groups[2].Value.Trim() -cne (Split-Path -Leaf $Path)) {
        throw "$Label checksum sidecar is malformed or names a different file."
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -cne $match.Groups[1].Value.ToUpperInvariant()) {
        throw "$Label checksum verification failed."
    }
}

function Get-InterruptedDeploymentState {
    $recoveryPath = Join-Path `
        $ProductionWorktree ".runtime\production-recovery-required.json"
    if (-not (Test-Path -LiteralPath $recoveryPath)) { return $null }
    Assert-ExistingFile $recoveryPath "Production recovery journal"

    try { $recovery = Read-JsonFile $recoveryPath }
    catch { throw "Production recovery journal is invalid: $($_.Exception.Message)" }
    Assert-ObjectProperties $recovery @(
        "schema_version", "release_id", "failed_release_commit",
        "previous_production_commit", "production_backup_path",
        "failure", "rollback", "created_utc"
    ) "Production recovery journal"
    Assert-ObjectProperties $recovery.rollback @(
        "status", "database_restore_required", "completed_utc", "notes"
    ) "Production recovery rollback journal"

    $releaseId = [string]$recovery.release_id
    $releaseCommit = [string]$recovery.failed_release_commit
    $previousCommit = [string]$recovery.previous_production_commit
    if ([int]$recovery.schema_version -ne 1 -or
        $releaseId -notmatch '^pharmacy-release-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$' -or
        $releaseCommit -notmatch '^[a-f0-9]{40}$' -or
        $previousCommit -notmatch '^[a-f0-9]{40}$') {
        throw "Production recovery journal identity fields are invalid."
    }
    if ([string]$recovery.failure -cne
        "Release deployment is in progress after the final backup." -or
        [string]$recovery.rollback.status -cne "backup_verified" -or
        [bool]$recovery.rollback.database_restore_required -or
        $null -ne $recovery.rollback.completed_utc) {
        throw (
            "Production recovery journal does not describe a resumable " +
            "interrupted deployment. Use audited manual recovery."
        )
    }

    $releaseDirectory = Join-Path $releasesDirectory $releaseId
    $manifestPath = Join-Path $releaseDirectory "manifest.json"
    Assert-ExistingFile $manifestPath "Interrupted release manifest"
    try { $manifest = Read-JsonFile $manifestPath }
    catch { throw "Interrupted release manifest is invalid: $($_.Exception.Message)" }
    Assert-ObjectProperties $manifest @(
        "schema_version", "release_id", "release_tag", "source_commit",
        "previous_production_commit", "production_branch", "remote",
        "production_worktree", "bundle_path", "bundle_sha256",
        "production_backup_path", "checks", "deployment", "synchronization"
    ) "Interrupted release manifest"
    Assert-ObjectProperties $manifest.checks @(
        "repository", "candidate", "backup", "production_health"
    ) "Interrupted release checks"
    Assert-ObjectProperties $manifest.deployment @(
        "status", "completed_utc", "rollback"
    ) "Interrupted release deployment"
    Assert-ObjectProperties $manifest.synchronization @(
        "status", "completed_utc", "error"
    ) "Interrupted release synchronization"

    if ([int]$manifest.schema_version -ne 1 -or
        [string]$manifest.release_id -cne $releaseId -or
        [string]$manifest.release_tag -cne $releaseId -or
        [string]$manifest.source_commit -cne $releaseCommit -or
        [string]$manifest.previous_production_commit -cne $previousCommit -or
        [string]$manifest.production_branch -cne $ProductionBranch -or
        [string]$manifest.remote -cne $Remote -or
        (ConvertTo-NormalizedPath ([string]$manifest.production_worktree)) -ne
            (ConvertTo-NormalizedPath $ProductionWorktree) -or
        [string]$manifest.checks.repository -cne "passed" -or
        [string]$manifest.checks.candidate -cne "passed" -or
        [string]$manifest.checks.backup -cne "passed" -or
        [string]$manifest.checks.production_health -cne "pending" -or
        [string]$manifest.deployment.status -cne "starting" -or
        $null -ne $manifest.deployment.completed_utc -or
        $null -ne $manifest.deployment.rollback -or
        [string]$manifest.synchronization.status -cne "not_started" -or
        $null -ne $manifest.synchronization.completed_utc -or
        $null -ne $manifest.synchronization.error) {
        throw "Interrupted release journal and manifest are not a resumable exact match."
    }

    $expectedBundle = Join-Path $releaseDirectory "$releaseId.bundle"
    if ((ConvertTo-NormalizedPath ([string]$manifest.bundle_path)) -ne
        (ConvertTo-NormalizedPath $expectedBundle)) {
        throw "Interrupted release bundle path is outside its release directory."
    }
    $bundlePath = Resolve-ValidatedChildFile `
        ([string]$manifest.bundle_path) $releaseDirectory "Interrupted release bundle"
    $bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
    if ($bundleHash -cne ([string]$manifest.bundle_sha256).ToUpperInvariant()) {
        throw "Interrupted release bundle checksum verification failed."
    }

    if ([string]$manifest.production_backup_path -cne
        [string]$recovery.production_backup_path) {
        throw "Interrupted release journal and manifest name different backups."
    }
    $backupRoot = Join-Path $ProductionWorktree "backups\database"
    $backupPath = Resolve-ValidatedChildFile `
        ([string]$manifest.production_backup_path) $backupRoot `
        "Interrupted release production backup"
    Assert-FileChecksumSidecar $backupPath "Interrupted release production backup"

    $development = Get-CleanBranchSnapshot `
        $DevelopmentWorktree $DevelopmentBranch "Development"
    Assert-DevelopmentIsLocalOnly $development
    $production = Get-CleanBranchSnapshot `
        $ProductionWorktree $ProductionBranch "Production"
    if ($development.CommonGitDirectory -ne $production.CommonGitDirectory -or
        $production.Commit -cne $releaseCommit) {
        throw "Interrupted release production HEAD is not the exact deployed commit."
    }
    $containsRelease = Invoke-Git -Worktree $development.Path -AllowFailure `
        -Arguments @("merge-base", "--is-ancestor", $releaseCommit, $development.Commit)
    if ($containsRelease.ExitCode -ne 0) {
        throw "Development no longer contains the interrupted release commit."
    }
    $isForwardRelease = Invoke-Git -Worktree $development.Path -AllowFailure `
        -Arguments @("merge-base", "--is-ancestor", $previousCommit, $releaseCommit)
    if ($isForwardRelease.ExitCode -ne 0) {
        throw "Interrupted release commit does not descend from its recorded baseline."
    }
    Assert-ExpectedRemote $development.Path | Out-Null
    Assert-ExpectedRemote $production.Path | Out-Null
    $tagCommit = Get-GitValue $production.Path @(
        "rev-parse", "refs/tags/$releaseId^{}"
    )
    if ($tagCommit -cne $releaseCommit) {
        throw "Interrupted release tag does not resolve to production HEAD."
    }
    $cachedRemote = Get-GitValue $production.Path @(
        "rev-parse", "refs/remotes/${Remote}/${ProductionBranch}"
    )
    if ($cachedRemote -cne $previousCommit) {
        throw "Cached origin/main changed after the interrupted deployment."
    }

    foreach ($required in @(
        (Join-Path $production.Path "manage.py"),
        (Join-Path $production.Path "env\Scripts\python.exe"),
        (Join-Path $production.Path $productionScriptRelativePath),
        (Join-Path $production.Path ".env"),
        (Join-Path $production.Path ".runtime\production-role.json")
    )) {
        Assert-ExistingFile $required "Interrupted production prerequisite"
    }

    return [pscustomobject]@{
        RecoveryPath = $recoveryPath
        Recovery = $recovery
        ManifestPath = $manifestPath
        Manifest = $manifest
        ReleaseId = $releaseId
        ReleaseCommit = $releaseCommit
        PreviousCommit = $previousCommit
        BackupPath = $backupPath
        Development = $development
        Production = $production
    }
}

function Resume-InterruptedDeployment([object]$Interrupted) {
    Confirm-Publish $Interrupted.ReleaseCommit "RECOVER"
    if ($DryRun) {
        Write-ReleaseMessage (
            "DRY RUN complete: would restart and health-check interrupted " +
            "release $($Interrupted.ReleaseId), then synchronize it atomically."
        ) "Green"
        return
    }

    Update-ReleaseRefs $Interrupted.Development.Path
    $remoteCommit = Get-GitValue $Interrupted.Development.Path @(
        "rev-parse", "refs/remotes/${Remote}/${ProductionBranch}"
    )
    if ($remoteCommit -cne $Interrupted.PreviousCommit) {
        throw "origin/main changed after the interrupted deployment; manual review is required."
    }

    $productionReleaseLock = Enter-ProductionReleaseLock $Interrupted.Production
    $script:activeProductionReleaseToken = $productionReleaseLock.Token
    try {
        $production = Get-CleanBranchSnapshot `
            $Interrupted.Production.Path $ProductionBranch "Production"
        if ($production.Commit -cne $Interrupted.ReleaseCommit -or
            -not (Test-Path -LiteralPath $Interrupted.RecoveryPath -PathType Leaf)) {
            throw "Interrupted deployment state changed while its recovery lock was acquired."
        }
        $artifacts = [pscustomobject]@{
            Id = $Interrupted.ReleaseId
            Tag = $Interrupted.ReleaseId
            ManifestPath = $Interrupted.ManifestPath
            Manifest = $Interrupted.Manifest
        }
        $preflight = [pscustomobject]@{ Production = $production }

        try {
            Write-ReleaseMessage (
                "Resuming interrupted deployment $($Interrupted.ReleaseId)..."
            ) "Cyan"
            Invoke-ProductionControl $production "start" | Out-Null
            Assert-ProductionHealthy $production
            $artifacts.Manifest.checks.production_health = "passed"
            $artifacts.Manifest.deployment.status = "healthy"
            $artifacts.Manifest.deployment.completed_utc = (Get-UtcNow).ToString("o")
            Write-JsonAtomic $artifacts.ManifestPath $artifacts.Manifest
        }
        catch {
            $failure = $_.Exception.Message
            $rollback = Invoke-BestEffortRollback `
                $production $Interrupted.PreviousCommit `
                $Interrupted.BackupPath $true
            $artifacts.Manifest.checks.production_health = "failed"
            $artifacts.Manifest.deployment.status = "failed"
            $artifacts.Manifest.deployment.rollback = $rollback
            if ($rollback.status -eq "healthy") {
                Clear-ProductionRecoveryBlock $production
            }
            else {
                Set-ProductionRecoveryBlock $production $artifacts $failure $rollback
            }
            Write-JsonAtomic $artifacts.ManifestPath $artifacts.Manifest
            throw "Interrupted production recovery failed: $failure"
        }

        Save-SyncPending $artifacts "Awaiting interrupted-release Git synchronization."
        Clear-ProductionRecoveryBlock $production
        Write-ReleaseMessage (
            "Recovered production is healthy; synchronizing main and the release tag..."
        ) "Cyan"
        Push-Release $preflight $artifacts
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    Write-ReleaseMessage (
        "Interrupted release $($Interrupted.ReleaseId) recovered and published successfully."
    ) "Green"
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

    $recoveryPath = Join-Path `
        $ProductionWorktree ".runtime\production-recovery-required.json"
    if (Test-Path -LiteralPath $recoveryPath) {
        Write-ReleaseMessage "INTERRUPTED DEPLOYMENT RECOVERY REQUIRED" "Yellow"
        try {
            $recovery = Read-JsonFile $recoveryPath
            Write-Host "Release:    $($recovery.release_id)"
            Write-Host "Production: $ProductionWorktree"
            Write-Host "State:      $($recovery.rollback.status)"
        }
        catch { Write-Host "Recovery journal is invalid and needs manual review." }
        Write-Host ""
        Write-Host (
            "Rerun -Action publish to validate, restart, health-check, and " +
            "resume only this interrupted release."
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
                $interrupted = Get-InterruptedDeploymentState
                if ($interrupted) {
                    Resume-InterruptedDeployment $interrupted
                }
                else {
                    $preflight = Get-ReleasePreflight -Fetch -RunChecks
                    Publish-NewRelease $preflight
                }
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
