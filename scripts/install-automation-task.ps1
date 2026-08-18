param(
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$runnerScript = Join-Path $PSScriptRoot "run-scheduled-jobs.ps1"
$hiddenRunnerScript = Join-Path $PSScriptRoot "run-scheduled-jobs-hidden.vbs"
$supplierRunnerScript = Join-Path $PSScriptRoot "run-supplier-orders.ps1"
$wscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
$cscriptExe = Join-Path $env:SystemRoot "System32\cscript.exe"
$taskName = "Pharmacy Scheduled Jobs"
$supplierTaskName = "Pharmacy Supplier Ordering"
$legacyBackupTaskName = "Pharmacy Database Backup"
$runtimeDirectory = Join-Path $projectRoot ".runtime"
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

function Get-SignedInInteractiveUser {
    try {
        $computerSystem = Get-CimInstance -ClassName Win32_ComputerSystem `
            -ErrorAction Stop
        if ($computerSystem.UserName) {
            return [string]$computerSystem.UserName
        }
    }
    catch {
        Write-InstallLog "Could not query the signed-in desktop user: $($_.Exception.Message)"
    }

    $identityName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    if ($identityName -notmatch '(^|\\)(SYSTEM|LOCAL SERVICE|NETWORK SERVICE)$') {
        return $identityName
    }
    return ""
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
        $RunAsUser = Get-SignedInInteractiveUser
    }
    if (-not $RunAsUser -or
        $RunAsUser -match '(^|\\)(SYSTEM|LOCAL SERVICE|NETWORK SERVICE)$') {
        throw "A signed-in Windows user is required for supplier ordering. Sign in, then rerun the installer."
    }
    if (-not (Test-Path -LiteralPath $runnerScript)) {
        throw "Scheduled-job runner is missing: $runnerScript"
    }
    if (-not (Test-Path -LiteralPath $hiddenRunnerScript)) {
        throw "Windowless scheduled-job launcher is missing: $hiddenRunnerScript"
    }
    if (-not (Test-Path -LiteralPath $supplierRunnerScript)) {
        throw "Supplier-order runner is missing: $supplierRunnerScript"
    }
    if (-not (Test-Path -LiteralPath $wscriptExe) -or
        -not (Test-Path -LiteralPath $cscriptExe)) {
        throw "Windows Script Host is required for the windowless scheduler launcher."
    }

    # Parse/probe the VBS without invoking the PowerShell runner or any job.
    & $cscriptExe @("//B", "//NoLogo", $hiddenRunnerScript, "--probe")
    if ($LASTEXITCODE -ne 0) {
        throw "Windowless scheduled-job launcher probe returned code $LASTEXITCODE."
    }

    # Run under the pharmacy user's interactive token because the project tree
    # is user-writable; executing it as SYSTEM would create an avoidable local
    # privilege-escalation path. GUI-subsystem Windows Script Host still makes
    # the hourly launch fully windowless while that user is signed in.
    $taskCommand = "`"$wscriptExe`" //B //NoLogo `"$hiddenRunnerScript`""
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
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 25)
    Set-ScheduledTask -TaskName $taskName -Settings $settings | Out-Null

    $verified = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    if ($verified.Principal.UserId -match '(?i)(^|\\)(SYSTEM)$' -or
        $verified.Principal.UserId -eq 'S-1-5-18') {
        throw "The hourly task must not execute user-writable project code as SYSTEM."
    }
    $verifiedAction = @($verified.Actions)[0]
    if ((Split-Path -Leaf $verifiedAction.Execute) -ine "wscript.exe" -or
        $verifiedAction.Arguments -notmatch [regex]::Escape($hiddenRunnerScript)) {
        throw "The task exists but is not using the windowless launcher."
    }

    # Temporarily replace the registered action with its no-work self-test and
    # start the task itself. This proves the registered principal can launch
    # the hidden runner, connect to Django's database/configuration, write logs,
    # lock the backup directory, and find pg_dump/pg_restore without making a
    # dump or evaluating any due job.
    $normalAction = New-ScheduledTaskAction `
        -Execute $wscriptExe `
        -Argument "//B //NoLogo `"$hiddenRunnerScript`""
    $selfTestAction = New-ScheduledTaskAction `
        -Execute $wscriptExe `
        -Argument "//B //NoLogo `"$hiddenRunnerScript`" --self-test"
    $selfTestResult = $null
    try {
        $previousTaskInfo = Get-ScheduledTaskInfo `
            -TaskName $taskName -ErrorAction Stop
        Set-ScheduledTask -TaskName $taskName -Action $selfTestAction | Out-Null
        Start-ScheduledTask -TaskName $taskName
        $deadline = (Get-Date).AddSeconds(90)
        $selfTestObserved = $false
        do {
            Start-Sleep -Milliseconds 500
            $verified = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
            $currentTaskInfo = Get-ScheduledTaskInfo `
                -TaskName $taskName -ErrorAction Stop
            $selfTestObserved = (
                $currentTaskInfo.LastRunTime -gt $previousTaskInfo.LastRunTime
            )
        } while ((
            $verified.State -eq 'Running' -or -not $selfTestObserved
        ) -and (Get-Date) -lt $deadline)
        if ($verified.State -eq 'Running') {
            throw "The registered-principal self-test did not finish within 90 seconds."
        }
        if (-not $selfTestObserved) {
            throw "Task Scheduler did not start the registered-principal self-test within 90 seconds."
        }
        $selfTestResult = $currentTaskInfo.LastTaskResult
    }
    finally {
        $selfTestTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($null -ne $selfTestTask -and
            $selfTestTask.State -in @('Running', 'Queued')) {
            Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        }
        Set-ScheduledTask -TaskName $taskName -Action $normalAction | Out-Null
    }
    if ($selfTestResult -ne 0) {
        throw "The registered-principal self-test returned code $selfTestResult."
    }
    $restoredTask = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $restoredAction = @($restoredTask.Actions)[0]
    if ($restoredAction.Arguments -match '(?i)--self-test') {
        throw "The hourly task action was not restored after its self-test."
    }

    # Remove the old 02:00 task immediately after its replacement has passed.
    # An unrelated supplier-browser smoke failure cannot leave duplicate backup
    # schedules behind.
    $legacyBackupTask = Get-ScheduledTask `
        -TaskName $legacyBackupTaskName -ErrorAction SilentlyContinue
    if ($null -ne $legacyBackupTask) {
        Unregister-ScheduledTask `
            -TaskName $legacyBackupTaskName -Confirm:$false
        Write-InstallLog "Removed legacy task: $legacyBackupTaskName."
    }

    # Supplier browsers do require the signed-in user's interactive desktop and
    # must not inherit Waitress's restrictive Job Object. Keep this separate,
    # on-demand process broker interactive while keeping its longer run limit.
    $supplierAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument (
            "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass " +
            "-File `"$supplierRunnerScript`""
        )
    $supplierPrincipal = New-ScheduledTaskPrincipal `
        -UserId $RunAsUser -LogonType Interactive -RunLevel Highest
    $supplierSettings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances Parallel `
        -ExecutionTimeLimit (New-TimeSpan -Hours 12)
    Register-ScheduledTask `
        -TaskName $supplierTaskName `
        -Action $supplierAction `
        -Principal $supplierPrincipal `
        -Settings $supplierSettings `
        -Force | Out-Null

    # With no launch marker this opens only a local headless about:blank page.
    # It proves Playwright can create processes outside Waitress's restrictive
    # job without initiating a supplier order or making external traffic.
    $pendingSupplierMarkers = @(
        Get-ChildItem -LiteralPath $runtimeDirectory `
            -Filter "supplier-order-*.launch" -File -ErrorAction SilentlyContinue
    )
    if ($pendingSupplierMarkers.Count -eq 0) {
        Start-ScheduledTask -TaskName $supplierTaskName
        $deadline = (Get-Date).AddSeconds(30)
        do {
            Start-Sleep -Milliseconds 500
            $verifiedSupplier = Get-ScheduledTask -TaskName $supplierTaskName -ErrorAction Stop
        } while ($verifiedSupplier.State -eq 'Running' -and (Get-Date) -lt $deadline)
        if ($verifiedSupplier.State -eq 'Running') {
            throw "The supplier-launcher browser smoke did not finish within 30 seconds."
        }
        $supplierTaskInfo = Get-ScheduledTaskInfo -TaskName $supplierTaskName -ErrorAction Stop
        if ($supplierTaskInfo.LastTaskResult -ne 0) {
            throw "The supplier-launcher browser smoke returned code $($supplierTaskInfo.LastTaskResult)."
        }
    }
    else {
        Write-InstallLog "Skipped supplier browser smoke because a launch request is pending."
    }

    Write-InstallLog "Windowless scheduled jobs and supplier launcher installed and self-tested for $RunAsUser."
    Write-Host "Pharmacy automation installed successfully." -ForegroundColor Green
    Write-Host "The hidden dispatcher checks once per hour at :30 and runs each database-backed job once when due."
    Write-Host "The on-demand supplier launcher is ready for job-constrained production starts."
    Write-Host "Project: $projectRoot"
    exit 0
}
catch {
    Write-InstallLog "FAILED $($_.Exception.Message)"
    Write-Host "Could not install pharmacy automation: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
