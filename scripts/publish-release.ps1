[CmdletBinding()]
param(
    [ValidateSet("check", "publish", "status", "register-pr", "finalize-pr")]
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
    [switch]$PullRequest,
    [string]$PullRequestUrl = "",

    # Test seams. Normal operators should not set these parameters.
    [scriptblock]$CommandInvoker,
    [scriptblock]$Clock,
    [string]$SecurityReviewPathForTests = "",
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
if ($PullRequest -and $Action -notin @("check", "publish")) {
    throw "-PullRequest is supported only with -Action check or publish."
}
if ($PullRequestUrl -and $Action -cne "register-pr") {
    throw "-PullRequestUrl is supported only with -Action register-pr."
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
$pullRequestPendingPath = Join-Path $StateDirectory "pull-request-pending.json"
$commonApplicationData = [Environment]::GetFolderPath(
    [Environment+SpecialFolder]::CommonApplicationData
)
if (-not $commonApplicationData) {
    throw "Windows did not report the machine-wide application data directory."
}
$securityReviewRequiredPath = Join-Path $commonApplicationData `
    "FINAL-PHARM\release-security\security-review-required.json"
if ($CommandInvoker) {
    # An injected command runner is a test-only execution mode. Keep every
    # associated path inside the OS temporary directory so this seam cannot be
    # pointed at the real development or production checkout.
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')
    foreach ($testPath in @(
        $normalizedDevelopmentRoot,
        $normalizedProductionRoot,
        $normalizedStateDirectory
    )) {
        if (-not $testPath.StartsWith(
            $temporaryRoot + '\',
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "The injected command runner is restricted to temporary test worktrees."
        }
    }
    if (-not $SecurityReviewPathForTests) {
        $SecurityReviewPathForTests = Join-Path `
            $StateDirectory "security-review-required.test.json"
    }
    $normalizedTestSecurityPath = [IO.Path]::GetFullPath(
        $SecurityReviewPathForTests
    )
    if (-not $normalizedTestSecurityPath.StartsWith(
        $temporaryRoot + '\',
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "The test security-review path must stay inside the OS temporary directory."
    }
    $securityReviewRequiredPath = $normalizedTestSecurityPath
}
elseif ($SecurityReviewPathForTests) {
    throw "-SecurityReviewPathForTests requires the temporary injected-command test seam."
}
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

function Get-GitHubRepositorySlug {
    $normalized = ConvertTo-NormalizedRemote $ExpectedOriginUrl
    $match = [regex]::Match(
        $normalized,
        '^https://github\.com/([a-z0-9_.-]+/[a-z0-9_.-]+)$',
        [Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $match.Success) {
        throw "Pull-request releases require the configured GitHub HTTPS remote."
    }
    return $match.Groups[1].Value
}

function Get-ReleaseReviewBranch([string]$ReleaseId) {
    if ($ReleaseId -notmatch '^pharmacy-release-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$') {
        throw "Cannot derive a review branch from an invalid release ID."
    }
    return "release/$ReleaseId"
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

function Invoke-GitHub(
    [string[]]$Arguments,
    [string]$WorkingDirectory = $DevelopmentWorktree,
    [switch]$AllowFailure
) {
    return Invoke-ReleaseProcess -FilePath "gh.exe" `
        -Arguments $Arguments -WorkingDirectory $WorkingDirectory `
        -AllowFailure:$AllowFailure
}

function Assert-GitHubPrerequisites([string]$Worktree) {
    $repository = Get-GitHubRepositorySlug
    Write-ReleaseMessage "Checking GitHub pull-request access..." "Cyan"
    Invoke-GitHub -WorkingDirectory $Worktree -Arguments @(
        "auth", "status", "--hostname", "github.com"
    ) | Out-Null
    $repositoryResult = Invoke-GitHub -WorkingDirectory $Worktree -Arguments @(
        "repo", "view", $repository,
        "--json", "nameWithOwner", "--jq", ".nameWithOwner"
    )
    if ($repositoryResult.Output.Trim() -ine $repository) {
        throw "GitHub CLI resolved a different repository than $repository."
    }
}

function Get-RemoteRefCommit(
    [string]$Worktree,
    [string]$Ref,
    [switch]$PeeledTag
) {
    if ($Ref -notmatch '^refs/(heads|tags)/[A-Za-z0-9._/-]+$' -or
        $Ref -match '(?:^|/)\.\.?(?:/|$)|\.\.|@\{|[~^:?*\[\\]') {
        throw "Remote ref is not safe to inspect: $Ref"
    }
    $queryRef = if ($PeeledTag) { "$Ref^{}" } else { $Ref }
    $result = Invoke-Git -Worktree $Worktree -Arguments @(
        "ls-remote", $Remote, $queryRef
    )
    $lines = @($result.Output -split "`r?`n" | Where-Object { $_.Trim() })
    if ($lines.Count -eq 0) { return "" }
    if ($lines.Count -ne 1) {
        throw "Remote ref inspection returned multiple matches for $queryRef."
    }
    $match = [regex]::Match(
        $lines[0],
        '^([a-f0-9]{40})\s+(.+)$',
        [Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $match.Success -or $match.Groups[2].Value -cne $queryRef) {
        throw "Remote ref inspection returned malformed data for $queryRef."
    }
    return $match.Groups[1].Value.ToLowerInvariant()
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

function Assert-PendingRemoteIdentity([object]$Manifest) {
    $recordedRemote = ConvertTo-NormalizedRemote ([string]$Manifest.remote_url)
    $expectedRemote = ConvertTo-NormalizedRemote $ExpectedOriginUrl
    if ([string]$Manifest.remote -cne $Remote -or
        [string]$Manifest.production_branch -cne $ProductionBranch -or
        $recordedRemote -cne $expectedRemote) {
        throw "Pending release remote identity does not match its guarded configuration."
    }
    $developmentRemote = Assert-ExpectedRemote $DevelopmentWorktree
    $productionRemote = Assert-ExpectedRemote $ProductionWorktree
    if ((ConvertTo-NormalizedRemote $developmentRemote) -cne $recordedRemote -or
        (ConvertTo-NormalizedRemote $productionRemote) -cne $recordedRemote) {
        throw "Live development or production remote does not match the pending release."
    }
}

function Assert-RunningPublisherMatchesRelease([object]$Pending) {
    $production = Get-CleanBranchSnapshot `
        $ProductionWorktree $ProductionBranch "Production"
    if ($production.Commit -cne [string]$Pending.commit) {
        throw "Pending release production HEAD no longer matches its exact commit."
    }
    $publisherWorktree = (Resolve-Path (Join-Path $scriptRoot "..")).Path
    if ((Resolve-GitCommonDirectory $publisherWorktree) -ne
        $production.CommonGitDirectory) {
        throw "The running publisher is not from the pending release repository."
    }
    $scriptDiff = Invoke-Git -Worktree $publisherWorktree -AllowFailure `
        -Arguments @(
            "diff", "--quiet", [string]$Pending.commit, "--",
            "scripts/publish-release.ps1"
        )
    if ($scriptDiff.ExitCode -ne 0) {
        throw (
            "The running publisher differs from the production release commit. " +
            "Use that release's production copy for pending actions."
        )
    }
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

function Get-ReleasePreflight(
    [switch]$Fetch,
    [switch]$RunChecks,
    [switch]$PullRequestMode
) {
    if (Test-Path -LiteralPath $pendingPath) {
        throw "A previous release is sync-pending. Run this script with -Action status."
    }
    if (Test-Path -LiteralPath $pullRequestPendingPath) {
        throw (
            "A production-first pull request is pending. Run -Action status; " +
            "another release is blocked until it is finalized."
        )
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
    if ($RunChecks) {
        if ($PullRequestMode) { Assert-GitHubPrerequisites $development.Path }
        Invoke-CandidateChecks $development $production
    }

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

function Clear-MatchingProductionRecoveryBlock(
    [object]$Production,
    [object]$Pending,
    [object]$Manifest
) {
    $path = Join-Path `
        $Production.Path ".runtime\production-recovery-required.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return }
    try { $recovery = Read-JsonFile $path }
    catch { throw "Production recovery journal is invalid: $($_.Exception.Message)" }
    Assert-ObjectProperties $recovery @(
        "schema_version", "release_id", "failed_release_commit",
        "previous_production_commit", "production_backup_path", "failure",
        "rollback"
    ) "Production recovery journal"
    Assert-ObjectProperties $recovery.rollback @(
        "status", "database_restore_required", "completed_utc", "notes"
    ) "Production recovery rollback journal"
    if ([int]$recovery.schema_version -ne 1 -or
        [string]$recovery.release_id -cne [string]$Pending.release_id -or
        [string]$recovery.failed_release_commit -cne [string]$Pending.commit -or
        [string]$recovery.previous_production_commit -cne
            [string]$Pending.previous_main_commit -or
        [string]$recovery.production_backup_path -cne
            [string]$Manifest.production_backup_path -or
        [string]$recovery.failure -cne
            "Release deployment is in progress after the final backup." -or
        [string]$recovery.rollback.status -cne "backup_verified" -or
        [bool]$recovery.rollback.database_restore_required -or
        $null -ne $recovery.rollback.completed_utc) {
        throw "Production recovery journal does not match the healthy pending release."
    }
    Clear-ProductionRecoveryBlock $Production
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
    $reviewBranch = if ($PullRequest) {
        Get-ReleaseReviewBranch $releaseId
    }
    else { $null }
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
            mode = if ($PullRequest) { "pull_request" } else { "direct" }
            status = "not_started"
            completed_utc = $null
            error = $null
            review_branch = $reviewBranch
            pull_request_number = $null
            pull_request_url = $null
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
        remote_url = [string]$manifest.remote_url
        production_worktree = $manifest.production_worktree
        manifest_path = $Artifacts.ManifestPath
        created_utc = (Get-UtcNow).ToString("o")
        last_error = $ErrorMessage
    }
    Write-JsonAtomic $pendingPath $pending
}

function Save-PullRequestPending(
    [object]$Artifacts,
    [string]$Phase,
    [string]$ErrorMessage = "",
    [object]$Existing = $null
) {
    $allowedPhases = @(
        "candidate_branch_pending", "awaiting_pull_request",
        "awaiting_exact_main", "main_sync_pending", "remote_main_mismatch",
        "main_synced_pr_status_pending"
    )
    if ($Phase -notin $allowedPhases) {
        throw "Pull-request release phase is invalid: $Phase"
    }
    $manifest = $Artifacts.Manifest
    $reviewBranch = Get-ReleaseReviewBranch $Artifacts.Id
    $manifest.synchronization.mode = "pull_request"
    $manifest.synchronization.status = "pull_request_pending"
    $manifest.synchronization.error = if ($ErrorMessage) { $ErrorMessage } else { $null }
    $manifest.synchronization.review_branch = $reviewBranch
    $createdUtc = if ($Existing -and
        $Existing.PSObject.Properties.Name -contains "created_utc") {
        [string]$Existing.created_utc
    }
    else { (Get-UtcNow).ToString("o") }
    $pullRequestNumber = if ($Existing -and
        $Existing.PSObject.Properties.Name -contains "pull_request_number") {
        $Existing.pull_request_number
    }
    else { $null }
    $pullRequestUrl = if ($Existing -and
        $Existing.PSObject.Properties.Name -contains "pull_request_url") {
        $Existing.pull_request_url
    }
    else { $null }

    $manifest.synchronization.pull_request_number = $pullRequestNumber
    $manifest.synchronization.pull_request_url = $pullRequestUrl
    Write-JsonAtomic $Artifacts.ManifestPath $manifest

    $pending = [pscustomobject][ordered]@{
        schema_version = 1
        release_id = $Artifacts.Id
        tag = $Artifacts.Tag
        commit = [string]$manifest.source_commit
        previous_main_commit = [string]$manifest.previous_production_commit
        production_branch = $ProductionBranch
        remote = $Remote
        remote_url = [string]$manifest.remote_url
        production_worktree = [string]$manifest.production_worktree
        manifest_path = $Artifacts.ManifestPath
        review_branch = $reviewBranch
        phase = $Phase
        pull_request_number = $pullRequestNumber
        pull_request_url = $pullRequestUrl
        created_utc = $createdUtc
        updated_utc = (Get-UtcNow).ToString("o")
        last_error = if ($ErrorMessage) { $ErrorMessage } else { $null }
    }
    Write-JsonAtomic $pullRequestPendingPath $pending
    return $pending
}

function Assert-PullRequestPendingState([object]$Pending) {
    Assert-ObjectProperties $Pending @(
        "schema_version", "release_id", "tag", "commit",
        "previous_main_commit", "production_branch", "remote",
        "remote_url", "production_worktree", "manifest_path", "review_branch", "phase",
        "pull_request_number", "pull_request_url", "created_utc",
        "updated_utc", "last_error"
    ) "Pull-request pending state"
    if ([int]$Pending.schema_version -ne 1) {
        throw "Pull-request pending state must use schema_version 1."
    }
    $releaseId = [string]$Pending.release_id
    $commit = [string]$Pending.commit
    $previousCommit = [string]$Pending.previous_main_commit
    if ($releaseId -notmatch '^pharmacy-release-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$' -or
        [string]$Pending.tag -cne $releaseId -or
        $commit -notmatch '^[a-f0-9]{40}$' -or
        $previousCommit -notmatch '^[a-f0-9]{40}$') {
        throw "Pull-request pending release identity is invalid."
    }
    $expectedReviewBranch = Get-ReleaseReviewBranch $releaseId
    if ([string]$Pending.review_branch -cne $expectedReviewBranch) {
        throw "Pull-request pending review branch is not derived from its release ID."
    }
    if ([string]$Pending.production_branch -cne $ProductionBranch -or
        [string]$Pending.remote -cne $Remote) {
        throw "Pull-request pending branch or remote does not match origin/main."
    }
    if ((ConvertTo-NormalizedPath ([string]$Pending.production_worktree)) -ne
        (ConvertTo-NormalizedPath $ProductionWorktree)) {
        throw "Pull-request pending production worktree does not match configuration."
    }
    $allowedPhases = @(
        "candidate_branch_pending", "awaiting_pull_request",
        "awaiting_exact_main", "main_sync_pending", "remote_main_mismatch",
        "main_synced_pr_status_pending"
    )
    if ([string]$Pending.phase -notin $allowedPhases) {
        throw "Pull-request pending phase is invalid."
    }

    $expectedManifest = Join-Path `
        (Join-Path $releasesDirectory $releaseId) "manifest.json"
    if ((ConvertTo-NormalizedPath ([string]$Pending.manifest_path)) -ne
        (ConvertTo-NormalizedPath $expectedManifest)) {
        throw "Pull-request pending manifest path is outside the release directory."
    }
    Assert-ExistingFile $expectedManifest "Pull-request release manifest"
    $manifest = Read-JsonFile $expectedManifest
    Assert-ObjectProperties $manifest @(
        "schema_version", "release_id", "release_tag", "source_commit",
        "previous_production_commit", "production_branch", "remote", "remote_url",
        "production_worktree", "bundle_path", "bundle_sha256",
        "production_backup_path", "checks", "deployment", "synchronization"
    ) "Pull-request release manifest"
    Assert-ObjectProperties $manifest.checks @(
        "repository", "candidate", "backup", "production_health"
    ) "Pull-request release checks"
    Assert-ObjectProperties $manifest.deployment @(
        "status", "completed_utc", "rollback"
    ) "Pull-request release deployment"
    Assert-ObjectProperties $manifest.synchronization @(
        "mode", "status", "review_branch", "pull_request_number",
        "pull_request_url"
    ) "Pull-request manifest synchronization"
    $manifestSyncStatus = [string]$manifest.synchronization.status
    $allowsCompletedManifest = (
        [string]$Pending.phase -eq "main_synced_pr_status_pending" -and
        $manifestSyncStatus -eq "complete"
    )
    if ([int]$manifest.schema_version -ne 1 -or
        [string]$manifest.release_id -cne $releaseId -or
        [string]$manifest.release_tag -cne $releaseId -or
        [string]$manifest.source_commit -cne $commit -or
        [string]$manifest.previous_production_commit -cne $previousCommit -or
        [string]$manifest.production_branch -cne $ProductionBranch -or
        [string]$manifest.remote -cne $Remote -or
        (ConvertTo-NormalizedRemote ([string]$manifest.remote_url)) -cne
            (ConvertTo-NormalizedRemote $ExpectedOriginUrl) -or
        (ConvertTo-NormalizedRemote ([string]$Pending.remote_url)) -cne
            (ConvertTo-NormalizedRemote ([string]$manifest.remote_url)) -or
        [string]$manifest.synchronization.mode -cne "pull_request" -or
        ($manifestSyncStatus -cne "pull_request_pending" -and
            -not $allowsCompletedManifest) -or
        [string]$manifest.synchronization.review_branch -cne $expectedReviewBranch -or
        [string]$manifest.checks.repository -cne "passed" -or
        [string]$manifest.checks.candidate -cne "passed" -or
        [string]$manifest.checks.backup -cne "passed" -or
        [string]$manifest.checks.production_health -cne "passed" -or
        [string]$manifest.deployment.status -cne "healthy" -or
        (ConvertTo-NormalizedPath ([string]$manifest.production_worktree)) -ne
            (ConvertTo-NormalizedPath $ProductionWorktree)) {
        throw "Pull-request pending state does not match its release manifest."
    }
    $releaseDirectory = Join-Path $releasesDirectory $releaseId
    $expectedBundle = Join-Path $releaseDirectory "$releaseId.bundle"
    if ((ConvertTo-NormalizedPath ([string]$manifest.bundle_path)) -ne
        (ConvertTo-NormalizedPath $expectedBundle) -or
        [string]$manifest.bundle_sha256 -notmatch '^[A-Fa-f0-9]{64}$') {
        throw "Pull-request release bundle identity is invalid."
    }
    $bundlePath = Resolve-ValidatedChildFile `
        ([string]$manifest.bundle_path) $releaseDirectory `
        "Pull-request release bundle"
    $bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
    if ($bundleHash -cne ([string]$manifest.bundle_sha256).ToUpperInvariant()) {
        throw "Pull-request release bundle checksum verification failed."
    }
    $backupRoot = Join-Path $ProductionWorktree "backups\database"
    $backupPath = Resolve-ValidatedChildFile `
        ([string]$manifest.production_backup_path) $backupRoot `
        "Pull-request release production backup"
    Assert-FileChecksumSidecar `
        $backupPath "Pull-request release production backup"

    $requiresPullRequest = [string]$Pending.phase -in @(
        "awaiting_exact_main", "main_sync_pending", "remote_main_mismatch",
        "main_synced_pr_status_pending"
    )
    if ($requiresPullRequest) {
        $number = 0
        if (-not [int]::TryParse(
            [string]$Pending.pull_request_number,
            [ref]$number
        ) -or $number -lt 1) {
            throw "Pull-request pending state has no valid pull request number."
        }
        $repository = Get-GitHubRepositorySlug
        $expectedUrl = "https://github.com/$repository/pull/$number"
        if ([string]$Pending.pull_request_url -ine $expectedUrl -or
            [string]$manifest.synchronization.pull_request_url -ine $expectedUrl -or
            [int]$manifest.synchronization.pull_request_number -ne $number) {
            throw "Pull-request pending URL or number does not match the configured repository."
        }
    }
    elseif ($null -ne $Pending.pull_request_number -or
        $null -ne $Pending.pull_request_url) {
        throw "Pull-request identity was recorded before the registration phase."
    }

    foreach ($dateProperty in @("created_utc", "updated_utc")) {
        $parsed = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse(
            [string]$Pending.$dateProperty,
            [ref]$parsed
        )) {
            throw "Pull-request pending $dateProperty is invalid."
        }
    }
    return $manifest
}

function Get-ValidatedGitHubPullRequest(
    [object]$Pending,
    [string]$PullRequestUrl,
    [string[]]$AllowedStates = @("OPEN")
) {
    $repository = Get-GitHubRepositorySlug
    $urlMatch = [regex]::Match(
        $PullRequestUrl,
        '^https://github\.com/' + [regex]::Escape($repository) + '/pull/([1-9][0-9]*)$',
        [Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $urlMatch.Success) {
        throw "Pull request URL must belong to https://github.com/$repository."
    }
    $result = Invoke-GitHub -WorkingDirectory $DevelopmentWorktree -Arguments @(
        "pr", "view", $PullRequestUrl, "--repo", $repository,
        "--json",
        "number,url,state,isDraft,headRefName,headRefOid,baseRefName,reviewDecision,statusCheckRollup"
    )
    try { $pullRequest = $result.Output | ConvertFrom-Json }
    catch { throw "GitHub returned invalid pull-request metadata." }
    Assert-ObjectProperties $pullRequest @(
        "number", "url", "state", "isDraft", "headRefName",
        "headRefOid", "baseRefName", "reviewDecision", "statusCheckRollup"
    ) "GitHub pull request"
    $number = [int]$urlMatch.Groups[1].Value
    if ([int]$pullRequest.number -ne $number -or
        [string]$pullRequest.url -ine $PullRequestUrl -or
        [string]$pullRequest.baseRefName -cne $ProductionBranch -or
        [string]$pullRequest.headRefName -cne [string]$Pending.review_branch -or
        [string]$pullRequest.headRefOid -cne [string]$Pending.commit) {
        throw "GitHub pull request does not match the pending release identity."
    }
    if ([bool]$pullRequest.isDraft) {
        throw "The release pull request is still a draft."
    }
    if ([string]$pullRequest.state -notin $AllowedStates) {
        throw "The release pull request is $($pullRequest.state), not $($AllowedStates -join ' or ')."
    }
    return $pullRequest
}

function Assert-PullRequestChecksReady([object]$PullRequest) {
    # GitHub does not let an author approve their own PR. A blank decision is
    # therefore allowed only because FINALIZE is a separate exact-SHA operator
    # confirmation; every nonblank decision must be the positive GitHub state.
    if ([string]$PullRequest.reviewDecision -notin @("", "APPROVED")) {
        throw "The release pull request has not been approved for finalization."
    }
    foreach ($check in @($PullRequest.statusCheckRollup)) {
        $status = if ($check.PSObject.Properties.Name -contains "status") {
            [string]$check.status
        }
        else { "" }
        $conclusion = if ($check.PSObject.Properties.Name -contains "conclusion") {
            [string]$check.conclusion
        }
        else { "" }
        $legacyState = if ($check.PSObject.Properties.Name -contains "state") {
            [string]$check.state
        }
        else { "" }
        if ($legacyState) {
            if ($legacyState -cne "SUCCESS") {
                throw "A pull-request status is not successful: $legacyState"
            }
            continue
        }
        if ($status -or $conclusion) {
            if ($status -cne "COMPLETED") {
                throw "A pull-request check is still pending."
            }
            if ($conclusion -notin @("SUCCESS", "NEUTRAL", "SKIPPED")) {
                throw "A pull-request check did not pass: $conclusion"
            }
            continue
        }
        throw "GitHub returned an unrecognized pull-request check result."
    }
}

function Push-PullRequestCandidate(
    [object]$Production,
    [object]$Artifacts,
    [object]$Pending
) {
    $commit = [string]$Artifacts.Manifest.source_commit
    $branch = [string]$Pending.review_branch
    $ref = "refs/heads/$branch"
    try {
        $remoteCommit = Get-RemoteRefCommit $Production.Path $ref
        if ($remoteCommit -and $remoteCommit -cne $commit) {
            throw "Remote release branch $branch points to another commit."
        }
        if (-not $remoteCommit) {
            Assert-SecurityReviewComplete
            Invoke-Git -Worktree $Production.Path -Mutation -Arguments @(
                "push", $Remote, "${commit}:$ref"
            ) | Out-Null
        }
        $verified = Get-RemoteRefCommit $Production.Path $ref
        if ($verified -cne $commit) {
            throw "Remote release branch did not resolve to the deployed commit."
        }
        return Save-PullRequestPending `
            $Artifacts "awaiting_pull_request" "" $Pending
    }
    catch {
        $detail = $_.Exception.Message
        Save-PullRequestPending `
            $Artifacts "candidate_branch_pending" $detail $Pending | Out-Null
        throw (
            "Production is healthy, but its review branch is pending. " +
            "Rerun publish with -PullRequest to retry.`n$detail"
        )
    }
}

function New-PendingArtifacts([object]$Pending, [object]$Manifest) {
    return [pscustomobject]@{
        Id = [string]$Pending.release_id
        Tag = [string]$Pending.tag
        ManifestPath = [string]$Pending.manifest_path
        Manifest = $Manifest
    }
}

function Resume-PendingPullRequest([object]$Pending) {
    $manifest = Assert-PullRequestPendingState $Pending
    Assert-RunningPublisherMatchesRelease $Pending
    Assert-PendingRemoteIdentity $manifest
    $retryBranch = [string]$Pending.phase -eq "candidate_branch_pending"
    if ($retryBranch) {
        Confirm-Publish ([string]$Pending.commit) "PR"
    }
    if ($DryRun) {
        Write-ReleaseMessage (
            "DRY RUN complete: would revalidate production and its pending PR state."
        ) "Green"
        return
    }

    $production = Get-CleanBranchSnapshot `
        $ProductionWorktree $ProductionBranch "Production"
    $productionReleaseLock = Enter-ProductionReleaseLock $production
    $script:activeProductionReleaseToken = $productionReleaseLock.Token
    try {
        Assert-SecurityReviewComplete
        $production = Get-CleanBranchSnapshot `
            $ProductionWorktree $ProductionBranch "Production"
        if ($production.Commit -cne [string]$Pending.commit) {
            throw "Pending pull-request commit does not match production HEAD."
        }
        Assert-ExpectedRemote $production.Path | Out-Null
        $tagCommit = Get-GitValue $production.Path @(
            "rev-parse", "refs/tags/$($Pending.tag)^{}"
        )
        if ($tagCommit -cne [string]$Pending.commit) {
            throw "Pending pull-request tag does not resolve to production HEAD."
        }
        Assert-ProductionHealthy $production
        Clear-MatchingProductionRecoveryBlock `
            $production $Pending $manifest
        if ($retryBranch) {
            $artifacts = New-PendingArtifacts $Pending $manifest
            $Pending = Push-PullRequestCandidate $production $artifacts $Pending
        }
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }
    if ($retryBranch) {
        Write-ReleaseMessage "Review branch is ready for pull-request creation." "Green"
    }
    else {
        Write-ReleaseMessage "Production-first pull request remains safely pending." "Yellow"
        Write-Host "Phase:         $($Pending.phase)"
        Write-Host "Review branch: $($Pending.review_branch)"
        if ($Pending.pull_request_url) {
            Write-Host "Pull request:  $($Pending.pull_request_url)"
        }
    }
}

function Register-PullRequest([object]$Pending, [string]$Url) {
    $manifest = Assert-PullRequestPendingState $Pending
    Assert-RunningPublisherMatchesRelease $Pending
    Assert-PendingRemoteIdentity $manifest
    if ([string]$Pending.phase -notin @("awaiting_pull_request", "awaiting_exact_main")) {
        throw "The pending release is not ready to register a pull request."
    }
    if (-not $Url) {
        throw "-PullRequestUrl is required with -Action register-pr."
    }
    $remoteCommit = Get-RemoteRefCommit `
        $DevelopmentWorktree "refs/heads/$($Pending.review_branch)"
    if ($remoteCommit -cne [string]$Pending.commit) {
        throw "The remote review branch no longer matches the deployed commit."
    }
    $pullRequest = Get-ValidatedGitHubPullRequest $Pending $Url @("OPEN")
    if ([string]$Pending.phase -eq "awaiting_exact_main" -and
        ([int]$Pending.pull_request_number -ne [int]$pullRequest.number -or
        [string]$Pending.pull_request_url -cne [string]$pullRequest.url)) {
        throw "A different pull request is already registered for this release."
    }
    if ($DryRun) {
        Write-ReleaseMessage (
            "DRY RUN complete: pull request identity is valid; no registration was saved."
        ) "Green"
        return
    }
    $Pending.pull_request_number = [int]$pullRequest.number
    $Pending.pull_request_url = [string]$pullRequest.url
    $artifacts = New-PendingArtifacts $Pending $manifest
    Save-PullRequestPending `
        $artifacts "awaiting_exact_main" "" $Pending | Out-Null
    Write-ReleaseMessage "Pull request registered; further releases remain blocked." "Green"
    Write-Host "Pull request: $($pullRequest.url)"
    Write-Host "After review approval, run -Action finalize-pr."
    Write-Host "Do not use GitHub Merge, Squash, or Rebase."
}

function Finalize-PullRequest([object]$Pending) {
    $manifest = Assert-PullRequestPendingState $Pending
    Assert-RunningPublisherMatchesRelease $Pending
    Assert-PendingRemoteIdentity $manifest
    if ([string]$Pending.phase -notin @(
        "awaiting_exact_main", "main_sync_pending", "remote_main_mismatch",
        "main_synced_pr_status_pending"
    )) {
        throw "The pending release has no registered pull request to finalize."
    }
    Confirm-Publish ([string]$Pending.commit) "FINALIZE"
    if ($DryRun) {
        Write-ReleaseMessage (
            "DRY RUN complete: would validate the approved PR and fast-forward origin/main."
        ) "Green"
        return
    }

    $production = Get-CleanBranchSnapshot `
        $ProductionWorktree $ProductionBranch "Production"
    $productionReleaseLock = Enter-ProductionReleaseLock $production
    $script:activeProductionReleaseToken = $productionReleaseLock.Token
    try {
        Assert-SecurityReviewComplete
        $production = Get-CleanBranchSnapshot `
            $ProductionWorktree $ProductionBranch "Production"
        if ($production.Commit -cne [string]$Pending.commit) {
            throw "Finalization requires production HEAD to remain on the deployed commit."
        }
        Assert-ExpectedRemote $production.Path | Out-Null
        Assert-GitHubPrerequisites $production.Path
        Assert-ProductionHealthy $production
        $tagCommit = Get-GitValue $production.Path @(
            "rev-parse", "refs/tags/$($Pending.tag)^{}"
        )
        if ($tagCommit -cne [string]$Pending.commit) {
            throw "The local release tag no longer resolves to the deployed commit."
        }
        $localTagObject = Get-GitValue $production.Path @(
            "rev-parse", "refs/tags/$($Pending.tag)"
        )

        $remoteMain = Get-RemoteRefCommit `
            $production.Path "refs/heads/$ProductionBranch"
        $remoteReview = Get-RemoteRefCommit `
            $production.Path "refs/heads/$($Pending.review_branch)"
        $remoteTagObject = Get-RemoteRefCommit `
            $production.Path "refs/tags/$($Pending.tag)"
        $remoteTag = Get-RemoteRefCommit `
            $production.Path "refs/tags/$($Pending.tag)" -PeeledTag
        if (($remoteTag -and $remoteTag -cne [string]$Pending.commit) -or
            ($remoteTagObject -and $remoteTagObject -cne $localTagObject)) {
            throw "The remote release tag points to another commit."
        }
        if ($remoteMain -cne [string]$Pending.commit -and
            $remoteReview -cne [string]$Pending.commit) {
            throw "The remote review branch no longer resolves to the deployed commit."
        }
        $allowedPullRequestStates = if ($remoteMain -ceq [string]$Pending.commit) {
            @("OPEN", "MERGED")
        }
        else { @("OPEN") }
        $pullRequest = Get-ValidatedGitHubPullRequest `
            $Pending ([string]$Pending.pull_request_url) $allowedPullRequestStates
        Assert-PullRequestChecksReady $pullRequest

        if ($remoteMain -cne [string]$Pending.commit) {
            if ($remoteMain -cne [string]$Pending.previous_main_commit) {
                $failure = (
                    "origin/main changed from the recorded pre-release baseline. " +
                    "No synchronization was attempted."
                )
                $artifacts = New-PendingArtifacts $Pending $manifest
                Save-PullRequestPending `
                    $artifacts "remote_main_mismatch" $failure $Pending | Out-Null
                throw $failure
            }
            $isFastForward = Invoke-Git -Worktree $production.Path -AllowFailure `
                -Arguments @(
                    "merge-base", "--is-ancestor",
                    [string]$Pending.previous_main_commit,
                    [string]$Pending.commit
                )
            if ($isFastForward.ExitCode -ne 0) {
                throw "The deployed commit is not a fast-forward from the recorded main baseline."
            }
            $artifacts = New-PendingArtifacts $Pending $manifest
            $Pending = Save-PullRequestPending `
                $artifacts "main_sync_pending" `
                "Awaiting exact origin/main and release-tag synchronization." $Pending
            Assert-SecurityReviewComplete
            $push = Invoke-Git -Worktree $production.Path -Mutation -AllowFailure `
                -Arguments @(
                    "push", "--atomic",
                    "--force-with-lease=refs/heads/${ProductionBranch}:$($Pending.previous_main_commit)",
                    "--force-with-lease=refs/heads/$($Pending.review_branch):$($Pending.commit)",
                    $Remote,
                    "$($Pending.commit):refs/heads/${ProductionBranch}",
                    "$($Pending.commit):refs/heads/$($Pending.review_branch)",
                    "$($localTagObject):refs/tags/$($Pending.tag)"
                )
            if ($push.ExitCode -ne 0) {
                $detail = if ($push.Output) { $push.Output } else { "Git push failed." }
                Save-PullRequestPending `
                    $artifacts "main_sync_pending" $detail $Pending | Out-Null
                throw "Production remains healthy, but exact Git synchronization failed.`n$detail"
            }
        }
        elseif (-not $remoteTag) {
            if ($remoteReview -cne [string]$Pending.commit) {
                throw "Release-tag recovery requires the exact review branch to remain present."
            }
            $artifacts = New-PendingArtifacts $Pending $manifest
            $Pending = Save-PullRequestPending `
                $artifacts "main_sync_pending" `
                "origin/main is exact; awaiting release-tag synchronization." $Pending
            Assert-SecurityReviewComplete
            $tagPush = Invoke-Git -Worktree $production.Path -Mutation -AllowFailure `
                -Arguments @(
                    "push", "--atomic",
                    "--force-with-lease=refs/heads/$($Pending.review_branch):$($Pending.commit)",
                    $Remote,
                    "$($Pending.commit):refs/heads/$($Pending.review_branch)",
                    "$($localTagObject):refs/tags/$($Pending.tag)"
                )
            if ($tagPush.ExitCode -ne 0) {
                $detail = if ($tagPush.Output) { $tagPush.Output } else { "Git tag push failed." }
                Save-PullRequestPending `
                    $artifacts "main_sync_pending" $detail $Pending | Out-Null
                throw "Production remains healthy, but release-tag synchronization failed.`n$detail"
            }
        }

        $verifiedMain = Get-RemoteRefCommit `
            $production.Path "refs/heads/$ProductionBranch"
        $verifiedTagObject = Get-RemoteRefCommit `
            $production.Path "refs/tags/$($Pending.tag)"
        $verifiedTag = Get-RemoteRefCommit `
            $production.Path "refs/tags/$($Pending.tag)" -PeeledTag
        if ($verifiedMain -cne [string]$Pending.commit -or
            $verifiedTag -cne [string]$Pending.commit -or
            $verifiedTagObject -cne $localTagObject) {
            $detail = "Remote main or the release tag did not verify at the deployed commit."
            $artifacts = New-PendingArtifacts $Pending $manifest
            Save-PullRequestPending `
                $artifacts "main_sync_pending" $detail $Pending | Out-Null
            throw $detail
        }

        # GitHub normally marks the PR merged after its exact head commit lands
        # on main. Persist a distinct phase before querying that eventual state
        # so a crash or GitHub delay can be resumed without another ref update.
        $artifacts = New-PendingArtifacts $Pending $manifest
        $Pending = Save-PullRequestPending `
            $artifacts "main_synced_pr_status_pending" `
            "Exact main and tag are synchronized; awaiting GitHub PR status." $Pending
        $postSyncPullRequest = Get-ValidatedGitHubPullRequest `
            $Pending ([string]$Pending.pull_request_url) @("OPEN", "MERGED")
        Assert-PullRequestChecksReady $postSyncPullRequest
        if ([string]$postSyncPullRequest.state -cne "MERGED") {
            throw (
                "origin/main and the release tag are exact, but GitHub has not " +
                "yet marked the pull request merged. Rerun -Action finalize-pr."
            )
        }
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    Assert-SecurityReviewComplete
    $manifest.synchronization.status = "complete"
    $manifest.synchronization.completed_utc = (Get-UtcNow).ToString("o")
    $manifest.synchronization.error = $null
    Write-JsonAtomic ([string]$Pending.manifest_path) $manifest
    Remove-StateFile $pullRequestPendingPath
    Write-ReleaseMessage "Approved pull request synchronized to the exact production commit." "Green"
}

function Push-Release([object]$Preflight, [object]$Artifacts) {
    Assert-SecurityReviewComplete
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
        $destination = if ($PullRequest) {
            "prepare a review-only pull request"
        }
        else { "push main and the release tag atomically" }
        Write-ReleaseMessage (
            "DRY RUN complete: would publish $shortCommit to " +
            "$($Preflight.Production.Path), verify health, then $destination."
        ) "Green"
        return
    }

    # Full tests can take several minutes and confirmation can add an arbitrary
    # pause. Refresh both remote refs and all clean-worktree invariants before
    # creating a tag, stopping production, or changing main.
    $checkedDevelopmentCommit = $Preflight.Relationship.DevelopmentCommit
    $checkedProductionCommit = $Preflight.Relationship.ProductionCommit
    $Preflight = Get-ReleasePreflight -Fetch -PullRequestMode:$PullRequest
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
            Assert-SecurityReviewComplete
            $lockedProduction = Get-CleanBranchSnapshot `
                $Preflight.Production.Path $ProductionBranch "Production"
            if ($lockedProduction.Commit -ne $Preflight.Relationship.ProductionCommit) {
                throw "Production HEAD changed after release validation; no deployment was attempted."
            }

            Assert-SecurityReviewComplete
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

            Assert-SecurityReviewComplete
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
        # A crash at any later point can therefore resume only the remote phase
        # without rerunning deployment or restoring a now-live database.
        if ($PullRequest) {
            $pullPending = Save-PullRequestPending `
                $artifacts "candidate_branch_pending" `
                "Awaiting the immutable review branch push."
            Clear-ProductionRecoveryBlock $Preflight.Production
            Write-ReleaseMessage (
                "Production is healthy; publishing the immutable review branch..."
            ) "Cyan"
            $pullPending = Push-PullRequestCandidate `
                $Preflight.Production $artifacts $pullPending
        }
        else {
            Save-SyncPending $artifacts "Awaiting initial atomic Git synchronization."
            Clear-ProductionRecoveryBlock $Preflight.Production
            Write-ReleaseMessage "Production is healthy; synchronizing main and the release tag..." "Cyan"
            Push-Release $Preflight $artifacts
        }
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    if ($PullRequest) {
        Write-ReleaseMessage (
            "Production release $($artifacts.Id) is healthy and ready for its review PR."
        ) "Green"
        Write-Host "Review branch: $($pullPending.review_branch)"
        Write-Host "Next: create the PR, then run -Action register-pr with its GitHub URL."
        Write-Host "Do not use GitHub Merge, Squash, or Rebase; use -Action finalize-pr after approval."
    }
    else {
        Write-ReleaseMessage "Release $($artifacts.Id) published successfully." "Green"
    }
}

function Read-JsonFile([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-SecurityReviewBlock {
    $parent = Split-Path $securityReviewRequiredPath -Parent
    $leaf = Split-Path $securityReviewRequiredPath -Leaf
    try {
        $parentItem = Get-Item -LiteralPath $parent -Force -ErrorAction Stop
    }
    catch [Management.Automation.ItemNotFoundException] {
        return $null
    }
    catch {
        throw (
            "Security review state cannot be inspected; release remains blocked. " +
            $_.Exception.Message
        )
    }
    if (-not $parentItem.PSIsContainer -or
        ($parentItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Security review state directory is not a regular local directory; release remains blocked."
    }

    try {
        $entry = @(
            Get-ChildItem -LiteralPath $parent -Force -ErrorAction Stop |
                Where-Object { $_.Name -ieq $leaf }
        )
    }
    catch {
        throw (
            "Security review state cannot be enumerated; release remains blocked. " +
            $_.Exception.Message
        )
    }
    if ($entry.Count -eq 0) { return $null }
    if ($entry.Count -ne 1 -or
        $entry[0].PSIsContainer -or
        -not ($entry[0] -is [IO.FileInfo]) -or
        ($entry[0].Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Security review state is not a regular local file; release remains blocked."
    }
    if ($entry[0].Length -gt 65536) {
        throw "Security review state is unexpectedly large; release remains blocked."
    }
    try {
        $block = Read-JsonFile $securityReviewRequiredPath
        Assert-ObjectProperties $block @("schema_version", "status", "reason") `
            "Security review block"
        $schemaVersion = 0
        if (-not [int]::TryParse(
            [string]$block.schema_version,
            [ref]$schemaVersion
        ) -or
            $schemaVersion -ne 1 -or
            [string]$block.status -cne "review_required" -or
            [string]::IsNullOrWhiteSpace([string]$block.reason)) {
            throw "Security review block has invalid content."
        }
        return $block
    }
    catch {
        throw (
            "Security review state is invalid; release remains blocked. " +
            $_.Exception.Message
        )
    }
}

function Assert-SecurityReviewComplete {
    $block = Get-SecurityReviewBlock
    if ($block) {
        throw (
            "Production and GitHub release are blocked for security review: " +
            [string]$block.reason
        )
    }
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
    $syncMode = if ($manifest.synchronization.PSObject.Properties.Name -contains "mode") {
        [string]$manifest.synchronization.mode
    }
    else { "direct" }
    if ($syncMode -notin @("direct", "pull_request")) {
        throw "Interrupted release synchronization mode is invalid."
    }
    if ($syncMode -eq "pull_request") {
        Assert-ObjectProperties $manifest.synchronization @(
            "review_branch", "pull_request_number", "pull_request_url"
        ) "Interrupted pull-request synchronization"
        if ([string]$manifest.synchronization.review_branch -cne
            (Get-ReleaseReviewBranch $releaseId)) {
            throw "Interrupted release review branch is invalid."
        }
    }

    $preHealthState = (
        [string]$manifest.checks.production_health -ceq "pending" -and
        [string]$manifest.deployment.status -ceq "starting" -and
        $null -eq $manifest.deployment.completed_utc -and
        $null -eq $manifest.deployment.rollback -and
        [string]$manifest.synchronization.status -ceq "not_started" -and
        $null -eq $manifest.synchronization.completed_utc -and
        $null -eq $manifest.synchronization.error
    )
    $postHealthPullRequestState = (
        $syncMode -ceq "pull_request" -and
        [string]$manifest.checks.production_health -ceq "passed" -and
        [string]$manifest.deployment.status -ceq "healthy" -and
        $null -ne $manifest.deployment.completed_utc -and
        $null -eq $manifest.deployment.rollback -and
        [string]$manifest.synchronization.status -in @(
            "not_started", "pull_request_pending"
        ) -and
        $null -eq $manifest.synchronization.completed_utc -and
        $null -eq $manifest.synchronization.pull_request_number -and
        $null -eq $manifest.synchronization.pull_request_url
    )

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
        (-not $preHealthState -and -not $postHealthPullRequestState)) {
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
        SynchronizationMode = $syncMode
        ProductionAlreadyHealthy = $postHealthPullRequestState
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
        Assert-SecurityReviewComplete
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
            if ($Interrupted.ProductionAlreadyHealthy) {
                Write-ReleaseMessage (
                    "Production was already verified; rechecking health without redeploying..."
                ) "Cyan"
                Assert-ProductionHealthy $production
            }
            else {
                Assert-SecurityReviewComplete
                Invoke-ProductionControl $production "start" | Out-Null
                Assert-ProductionHealthy $production
                $artifacts.Manifest.checks.production_health = "passed"
                $artifacts.Manifest.deployment.status = "healthy"
                $artifacts.Manifest.deployment.completed_utc = (Get-UtcNow).ToString("o")
                Write-JsonAtomic $artifacts.ManifestPath $artifacts.Manifest
            }
        }
        catch {
            $failure = $_.Exception.Message
            if ($Interrupted.ProductionAlreadyHealthy) {
                throw (
                    "The previously verified production release is now unhealthy. " +
                    "No automatic database rollback was attempted because it may " +
                    "have served live traffic. Use audited manual recovery. $failure"
                )
            }
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

        if ($Interrupted.SynchronizationMode -eq "pull_request") {
            $pullPending = Save-PullRequestPending `
                $artifacts "candidate_branch_pending" `
                "Awaiting interrupted-release review branch synchronization."
            Clear-ProductionRecoveryBlock $production
            Write-ReleaseMessage (
                "Recovered production is healthy; publishing its review branch..."
            ) "Cyan"
            Push-PullRequestCandidate `
                $production $artifacts $pullPending | Out-Null
        }
        else {
            Save-SyncPending $artifacts "Awaiting interrupted-release Git synchronization."
            Clear-ProductionRecoveryBlock $production
            Write-ReleaseMessage (
                "Recovered production is healthy; synchronizing main and the release tag..."
            ) "Cyan"
            Push-Release $preflight $artifacts
        }
    }
    finally {
        Exit-ProductionReleaseLock $productionReleaseLock
    }

    Write-ReleaseMessage (
        "Interrupted release $($Interrupted.ReleaseId) recovered successfully."
    ) "Green"
}

function Assert-SyncPendingState([object]$Pending) {
    $required = @(
        "schema_version", "release_id", "tag", "commit",
        "production_branch", "remote", "remote_url", "production_worktree",
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
        [string]$Pending.remote -cne $Remote -or
        (ConvertTo-NormalizedRemote ([string]$Pending.remote_url)) -cne
            (ConvertTo-NormalizedRemote $ExpectedOriginUrl)) {
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
    Assert-ObjectProperties $manifest @(
        "release_id", "release_tag", "source_commit", "production_branch",
        "remote", "remote_url", "production_worktree"
    ) "Sync-pending release manifest"
    if ([string]$manifest.release_id -cne $releaseId -or
        [string]$manifest.release_tag -cne $releaseId -or
        [string]$manifest.source_commit -cne $commit -or
        [string]$manifest.production_branch -cne $ProductionBranch -or
        [string]$manifest.remote -cne $Remote -or
        (ConvertTo-NormalizedRemote ([string]$manifest.remote_url)) -cne
            (ConvertTo-NormalizedRemote ([string]$Pending.remote_url)) -or
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
    Assert-RunningPublisherMatchesRelease $Pending
    Assert-PendingRemoteIdentity $validatedManifest
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
        Assert-SecurityReviewComplete
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
        Assert-SecurityReviewComplete
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
    $securityBlock = Get-SecurityReviewBlock
    if ($securityBlock) {
        Write-ReleaseMessage "SECURITY REVIEW REQUIRED" "Red"
        Write-Host ([string]$securityBlock.reason)
        Write-Host "Production and GitHub release actions remain blocked."
        Write-Host ""
    }
    if (Test-Path -LiteralPath $pullRequestPendingPath) {
        try {
            $pending = Read-JsonFile $pullRequestPendingPath
            Assert-PullRequestPendingState $pending | Out-Null
        }
        catch {
            Write-ReleaseMessage "PULL REQUEST PENDING STATE IS INVALID" "Red"
            Write-Host $_.Exception.Message
            Write-Host "Further releases remain blocked until audited manual recovery."
            return
        }
        Write-ReleaseMessage "PULL REQUEST PENDING" "Yellow"
        Write-Host "Release:      $($pending.release_id)"
        Write-Host "Commit:       $($pending.commit)"
        Write-Host "Production:   $($pending.production_worktree)"
        Write-Host "Phase:        $($pending.phase)"
        Write-Host "Review branch: $($pending.review_branch)"
        if ($pending.pull_request_url) {
            Write-Host "Pull request: $($pending.pull_request_url)"
        }
        if ($pending.last_error) {
            Write-Host "Last error:   $($pending.last_error)"
        }
        Write-Host ""
        switch ([string]$pending.phase) {
            "candidate_branch_pending" {
                Write-Host "Rerun -Action publish -PullRequest to retry only the review branch push."
            }
            "awaiting_pull_request" {
                Write-Host "Create the GitHub PR, then run -Action register-pr -PullRequestUrl <URL>."
            }
            "main_synced_pr_status_pending" {
                Write-Host (
                    "Main and the release tag are exact. Rerun -Action finalize-pr " +
                    "to recheck only GitHub's merged status; refs will not be pushed again."
                )
            }
            default {
                Write-Host "After explicit approval, run -Action finalize-pr."
            }
        }
        Write-Host "Do not use GitHub Merge, Squash, or Rebase."
        return
    }
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
    $hasDirectPending = Test-Path -LiteralPath $pendingPath -PathType Leaf
    $hasPullRequestPending = Test-Path `
        -LiteralPath $pullRequestPendingPath -PathType Leaf
    if ($hasDirectPending -and $hasPullRequestPending) {
        throw (
            "Both direct and pull-request pending states exist; no release " +
            "action is safe until this conflict receives audited manual review."
        )
    }
    switch ($Action) {
        "status" {
            Show-ReleaseStatus
        }
        "check" {
            if (-not $PullRequest) {
                throw (
                    "New release checks require -PullRequest. Direct main publication " +
                    "is retained only for recovery of an existing legacy pending release."
                )
            }
            $preflight = Get-ReleasePreflight `
                -Fetch -RunChecks -PullRequestMode:$PullRequest
            $shortCommit = $preflight.Relationship.DevelopmentCommit.Substring(0, 12)
            Write-ReleaseMessage (
                "Release check passed: $DevelopmentBranch at $shortCommit is " +
                "$($preflight.Relationship.CommitsAhead) commit(s) ahead of " +
                "$Remote/$ProductionBranch."
            ) "Green"
        }
        "publish" {
            $publishLock = Enter-PublishLock
            Assert-SecurityReviewComplete
            if (Test-Path -LiteralPath $pullRequestPendingPath) {
                if (-not $PullRequest) {
                    throw (
                        "A production-first pull request is pending. " +
                        "Use -Action status or rerun publish with -PullRequest."
                    )
                }
                Resume-PendingPullRequest (Read-JsonFile $pullRequestPendingPath)
            }
            elseif (Test-Path -LiteralPath $pendingPath) {
                if ($PullRequest) {
                    throw "A direct Git synchronization is pending; -PullRequest cannot replace it."
                }
                Resume-PendingSynchronization (Read-JsonFile $pendingPath)
            }
            else {
                $interrupted = Get-InterruptedDeploymentState
                if ($interrupted) {
                    Resume-InterruptedDeployment $interrupted
                }
                else {
                    if (-not $PullRequest) {
                        throw (
                            "New production releases require -PullRequest. Direct main " +
                            "publication is retained only for legacy pending recovery."
                        )
                    }
                    $preflight = Get-ReleasePreflight `
                        -Fetch -RunChecks -PullRequestMode:$PullRequest
                    Publish-NewRelease $preflight
                }
            }
        }
        "register-pr" {
            $publishLock = Enter-PublishLock
            Assert-SecurityReviewComplete
            if (-not (Test-Path -LiteralPath $pullRequestPendingPath)) {
                throw "No production-first pull request is awaiting registration."
            }
            Register-PullRequest `
                (Read-JsonFile $pullRequestPendingPath) $PullRequestUrl
        }
        "finalize-pr" {
            $publishLock = Enter-PublishLock
            Assert-SecurityReviewComplete
            if (-not (Test-Path -LiteralPath $pullRequestPendingPath)) {
                throw "No approved production-first pull request is awaiting finalization."
            }
            Finalize-PullRequest (Read-JsonFile $pullRequestPendingPath)
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
