param(
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$runnerScript = Join-Path $PSScriptRoot "run-scheduled-jobs.ps1"
$taskName = "Pharmacy Scheduled Jobs"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "automation-install.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-InstallLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Host "Administrator access is required to install pharmacy automation." -ForegroundColor Yellow
    Write-InstallLog "Requesting administrator elevation."
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $arguments = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$installerScript`" " +
        "-RunAsUser `"$currentUser`""
    )
    $elevated = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    Write-InstallLog "Elevated installer exited with code $($elevated.ExitCode)."
    exit $elevated.ExitCode
}

try {
    Write-InstallLog "Elevated installer started."
    if (-not $RunAsUser) {
        $RunAsUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    }
    if (-not (Test-Path -LiteralPath $runnerScript)) {
        throw "Scheduled-job runner is missing: $runnerScript"
    }
    # Keep the interactive-user task (it needs that user's Google credentials),
    # but prevent its hourly dispatcher window from flashing on the desktop.
    $taskCommand = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runnerScript`""
    & schtasks.exe @(
        "/Create", "/TN", $taskName, "/TR", $taskCommand,
        # Run at half past the hour so the pull still occurs exactly 30 minutes
        # before the configured whole-hour closing times.
        "/SC", "HOURLY", "/MO", "1", "/ST", "00:30",
        "/RU", $RunAsUser, "/IT", "/RL", "HIGHEST", "/F"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Windows Task Scheduler returned exit code $LASTEXITCODE."
    }

    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 25)
    Set-ScheduledTask -TaskName $taskName -Settings $settings | Out-Null

    $verified = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    if ($verified.Principal.UserId -eq 'SYSTEM') {
        throw "The task exists but is still configured as SYSTEM."
    }

    # Exercise the real scheduled action once so installation cannot report a
    # false success when the selected Windows account cannot launch Python.
    Start-ScheduledTask -TaskName $taskName
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $verified = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    } while ($verified.State -eq 'Running' -and (Get-Date) -lt $deadline)
    if ($verified.State -eq 'Running') {
        throw "The scheduled-task self-test did not finish within 30 seconds."
    }
    $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction Stop
    if ($taskInfo.LastTaskResult -ne 0) {
        throw "The scheduled-task self-test returned code $($taskInfo.LastTaskResult)."
    }

    Write-InstallLog "Task installed and self-tested for $RunAsUser."
    Write-Host "Pharmacy automation installed successfully." -ForegroundColor Green
    Write-Host "The hidden dispatcher checks once per hour at :30 and runs each database-backed job once when due."
    Write-Host "Project: $projectRoot"
    exit 0
}
catch {
    Write-InstallLog "FAILED $($_.Exception.Message)"
    Write-Host "Could not install pharmacy automation: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
