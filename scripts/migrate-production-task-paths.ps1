param(
    [string]$DevelopmentWorktree = "",
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$roleMarkerPath = Join-Path $projectRoot ".runtime\production-role.json"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "task-path-migration.log"
$powershellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$wscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$changedTasks = New-Object Collections.Generic.List[object]

function Write-MigrationLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-ProductionRole {
    if (-not (Test-Path -LiteralPath $roleMarkerPath -PathType Leaf)) {
        throw "Production role marker is missing: $roleMarkerPath"
    }
    $marker = Get-Content -LiteralPath $roleMarkerPath -Raw | ConvertFrom-Json
    $markedRoot = [IO.Path]::GetFullPath([string]$marker.worktree).TrimEnd('\')
    if ([int]$marker.schema_version -ne 1 -or
        [string]$marker.role -cne "production" -or
        [string]$marker.branch -cne "main" -or
        [string]$marker.remote -cne "origin" -or
        $markedRoot -ine $projectRoot.TrimEnd('\')) {
        throw "The production role marker does not authorize this worktree."
    }
    $branch = (& git -C $projectRoot symbolic-ref --quiet --short HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -cne "main") {
        throw "Scheduled task paths may be migrated only from production main."
    }
    $dirty = @(& git -C $projectRoot status --porcelain=v1 `
        --untracked-files=all 2>&1)
    if ($LASTEXITCODE -ne 0 -or
        (($dirty | ForEach-Object { [string]$_ }) -join "").Trim()) {
        throw "Scheduled task paths require a clean production main worktree."
    }
}

if (-not (Test-IsAdministrator)) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    if (-not $RunAsUser) { $RunAsUser = $identity }
    $arguments = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$installerScript`" " +
        "-DevelopmentWorktree `"$DevelopmentWorktree`" " +
        "-RunAsUser `"$RunAsUser`""
    )
    $elevated = Start-Process powershell.exe -Verb RunAs `
        -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    exit $elevated.ExitCode
}

try {
    Assert-ProductionRole
    $mappings = @(
        [pscustomobject]@{
            Name = "Pharmacy Supplier Ordering"
            Execute = $powershellExe
            Arguments = (
                "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass " +
                "-File `"$(Join-Path $PSScriptRoot 'run-supplier-orders.ps1')`""
            )
        },
        [pscustomobject]@{
            Name = "Pharmacy Scheduled Jobs"
            Execute = $wscriptExe
            Arguments = (
                "//B //NoLogo `"$(Join-Path $PSScriptRoot 'run-scheduled-jobs-hidden.vbs')`""
            )
        },
        [pscustomobject]@{
            Name = "Pharmacy Production Startup"
            Execute = $wscriptExe
            Arguments = (
                "//B //NoLogo `"$(Join-Path $PSScriptRoot 'start-production-hidden.vbs')`" " +
                "--no-browser --quiet"
            )
        }
    )

    $migrated = 0
    foreach ($mapping in $mappings) {
        $task = Get-ScheduledTask -TaskName $mapping.Name -ErrorAction SilentlyContinue
        if ($null -eq $task) {
            Write-MigrationLog "Task not installed; left unchanged: $($mapping.Name)."
            continue
        }
        if (@($task.Actions).Count -ne 1) {
            throw "Task '$($mapping.Name)' must have exactly one action."
        }
        $oldAction = @($task.Actions)[0]
        $principalFingerprint = (
            "$($task.Principal.UserId)|$($task.Principal.LogonType)|" +
            "$($task.Principal.RunLevel)"
        )
        $triggerFingerprint = @($task.Triggers) |
            ConvertTo-Json -Depth 8 -Compress
        $action = New-ScheduledTaskAction `
            -Execute $mapping.Execute -Argument $mapping.Arguments
        Set-ScheduledTask -TaskName $mapping.Name -Action $action | Out-Null
        $changedTasks.Add([pscustomobject]@{
            Name = $mapping.Name
            Action = $oldAction
        })

        $verified = Get-ScheduledTask -TaskName $mapping.Name -ErrorAction Stop
        $verifiedAction = @($verified.Actions)[0]
        if ($verifiedAction.Execute -ine $mapping.Execute -or
            $verifiedAction.Arguments -cne $mapping.Arguments) {
            throw "Task '$($mapping.Name)' did not retain the production action."
        }
        $verifiedPrincipalFingerprint = (
            "$($verified.Principal.UserId)|$($verified.Principal.LogonType)|" +
            "$($verified.Principal.RunLevel)"
        )
        $verifiedTriggerFingerprint = @($verified.Triggers) |
            ConvertTo-Json -Depth 8 -Compress
        if ($verifiedPrincipalFingerprint -cne $principalFingerprint -or
            $verifiedTriggerFingerprint -cne $triggerFingerprint) {
            throw "Task '$($mapping.Name)' principal or triggers changed unexpectedly."
        }
        $migrated++
        Write-MigrationLog "Migrated task action to production: $($mapping.Name)."
    }

    if ($DevelopmentWorktree) {
        $developmentRoot = [IO.Path]::GetFullPath($DevelopmentWorktree).TrimEnd('\')
        foreach ($taskName in @(
            "Pharmacy Supplier Ordering",
            "Pharmacy Scheduled Jobs",
            "Pharmacy Production Startup"
        )) {
            $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
            if ($null -eq $task) { continue }
            foreach ($action in @($task.Actions)) {
                $actionText = "$($action.Execute) $($action.Arguments)"
                if ($actionText.IndexOf(
                    $developmentRoot,
                    [StringComparison]::OrdinalIgnoreCase
                ) -ge 0) {
                    throw "Task '$taskName' still executes from development."
                }
            }
        }
    }

    Write-Host "Existing pharmacy task paths now point at production." -ForegroundColor Green
    Write-Host "Tasks migrated: $migrated; missing tasks were left unchanged."
    exit 0
}
catch {
    $failure = $_.Exception.Message
    foreach ($changed in @($changedTasks)) {
        try {
            Set-ScheduledTask -TaskName $changed.Name -Action $changed.Action |
                Out-Null
            Write-MigrationLog "Rolled back task action: $($changed.Name)."
        }
        catch {
            Write-MigrationLog "CRITICAL Could not roll back task '$($changed.Name)': $($_.Exception.Message)"
        }
    }
    Write-MigrationLog "FAILED $failure"
    Write-Host "Scheduled task path migration failed: $failure" -ForegroundColor Red
    exit 1
}
