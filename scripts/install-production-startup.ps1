param(
    [switch]$EnableAutoStart,
    [switch]$DisableAutoStart,
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($EnableAutoStart -and $DisableAutoStart) {
    throw "Choose either -EnableAutoStart or -DisableAutoStart, not both."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$hiddenRunnerScript = Join-Path $PSScriptRoot "start-production-hidden.vbs"
$productionBatch = Join-Path $projectRoot "production.bat"
$wscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
$cscriptExe = Join-Path $env:SystemRoot "System32\cscript.exe"
$taskName = "Pharmacy Production Startup"
$productionRoleFile = Join-Path $projectRoot ".runtime\production-role.json"

function ConvertTo-NormalizedProductionPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
}

function Assert-ProductionRole {
    if (-not (Test-Path -LiteralPath $productionRoleFile -PathType Leaf)) {
        throw "This checkout is not authorized for production startup installation."
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
    if ([string]$roleMarker.role -cne "production" -or
        [string]$roleMarker.branch -cne "main" -or
        [string]$roleMarker.remote -cne "origin") {
        throw "The production role marker has unexpected role, branch, or remote values."
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
        throw "Production startup installation requires branch 'main'."
    }
    $dirty = @(& $git.Source -C $projectRoot status --porcelain=v1 `
        --untracked-files=all 2>&1)
    if ($LASTEXITCODE -ne 0 -or
        (($dirty | ForEach-Object { [string]$_ }) -join "").Trim()) {
        throw "Production startup installation requires a clean main worktree."
    }
}

function New-PharmacyShortcut(
    [string]$Path,
    [string]$TargetPath,
    [string]$Arguments,
    [string]$Description
) {
    $shortcutDirectory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $shortcutDirectory | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = $Description
    $shortcut.Save()
}

function Install-ProductionShortcuts {
    $desktopDirectory = [Environment]::GetFolderPath("Desktop")
    $programsDirectory = [Environment]::GetFolderPath("Programs")
    if (-not $desktopDirectory -or -not $programsDirectory) {
        throw "Windows Desktop or Start Menu could not be located for this user."
    }

    $locations = @($desktopDirectory, $programsDirectory)
    foreach ($location in $locations) {
        $legacyAdminShortcut = Join-Path $location "Pharmacy Control.lnk"
        if (Test-Path -LiteralPath $legacyAdminShortcut -PathType Leaf) {
            Remove-Item -LiteralPath $legacyAdminShortcut -Force
        }
        New-PharmacyShortcut `
            -Path (Join-Path $location "Pharmacy.lnk") `
            -TargetPath $wscriptExe `
            -Arguments "//B //NoLogo `"$hiddenRunnerScript`" --user-requested" `
            -Description "Start the pharmacy without a command window"
        New-PharmacyShortcut `
            -Path (Join-Path $location "Pharmacy Admin Control.lnk") `
            -TargetPath $productionBatch `
            -Arguments "" `
            -Description "Open production administration controls"
    }
}

function Enable-ProductionAutoStart {
    if (-not $RunAsUser) {
        $script:RunAsUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    }
    if (-not $RunAsUser -or
        $RunAsUser -match '(^|\\)(SYSTEM|LOCAL SERVICE|NETWORK SERVICE)$') {
        throw "Automatic startup must run as the signed-in pharmacy server user, not a service account."
    }

    $action = New-ScheduledTaskAction `
        -Execute $wscriptExe `
        -Argument "//B //NoLogo `"$hiddenRunnerScript`" --no-browser --quiet"
    $logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $RunAsUser
    $logonTrigger.Delay = "PT30S"
    $recoveryTrigger = New-ScheduledTaskTrigger `
        -Once `
        -At ((Get-Date).AddMinutes(5)) `
        -RepetitionInterval (New-TimeSpan -Minutes 5)
    $triggers = @($logonTrigger, $recoveryTrigger)
    $principal = New-ScheduledTaskPrincipal `
        -UserId $RunAsUser -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $triggers `
        -Principal $principal `
        -Settings $settings `
        -Description "Start Pharmacy 30 seconds after sign-in and check every 5 minutes for recovery" `
        -Force | Out-Null

    $verified = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $verifiedAction = @($verified.Actions)[0]
    $verifiedTriggers = @($verified.Triggers)
    $verifiedLogonTrigger = (
        $verifiedTriggers | Where-Object {
            $_.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger"
        } | Select-Object -First 1
    )
    $verifiedRecoveryTrigger = (
        $verifiedTriggers | Where-Object {
            $_.CimClass.CimClassName -eq "MSFT_TaskTimeTrigger"
        } | Select-Object -First 1
    )
    if ($verifiedAction.Execute -ine $wscriptExe -or
        $verifiedAction.Arguments -notmatch [regex]::Escape($hiddenRunnerScript) -or
        $verifiedAction.Arguments -notmatch "--no-browser --quiet" -or
        $verified.Principal.LogonType -ne "Interactive" -or
        $verified.Principal.RunLevel -ne "Limited" -or
        $null -eq $verifiedLogonTrigger -or
        $verifiedLogonTrigger.Delay -ne "PT30S" -or
        $null -eq $verifiedRecoveryTrigger -or
        $verifiedRecoveryTrigger.Repetition.Interval -ne "PT5M") {
        throw "The production startup task was created with unexpected action, trigger, or user settings."
    }
}

function Disable-ProductionAutoStart {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Disabled automatic Pharmacy startup." -ForegroundColor Yellow
    }
    else {
        Write-Host "Automatic Pharmacy startup was already disabled." -ForegroundColor Yellow
    }
}

if (-not (Test-Path -LiteralPath $hiddenRunnerScript) -or
    -not (Test-Path -LiteralPath $productionBatch) -or
    -not (Test-Path -LiteralPath $wscriptExe) -or
    -not (Test-Path -LiteralPath $cscriptExe)) {
    throw "A required Pharmacy or Windows launcher file is missing."
}

Assert-ProductionRole

# Parse and dependency-check the VBS without starting production.
& $cscriptExe @("//B", "//NoLogo", $hiddenRunnerScript, "--probe")
if ($LASTEXITCODE -ne 0) {
    throw "The hidden production launcher probe returned code $LASTEXITCODE."
}

Install-ProductionShortcuts
Write-Host "Installed Pharmacy and Pharmacy Admin Control shortcuts on the Desktop and Start Menu." -ForegroundColor Green

if ($EnableAutoStart) {
    Enable-ProductionAutoStart
    Write-Host (
        "Pharmacy will start 30 seconds after $RunAsUser signs in and " +
        "will be checked every 5 minutes for recovery."
    ) -ForegroundColor Green
}
elseif ($DisableAutoStart) {
    Disable-ProductionAutoStart
}
else {
    Write-Host "Automatic startup was left unchanged. Use -EnableAutoStart to enable it." -ForegroundColor Cyan
}
