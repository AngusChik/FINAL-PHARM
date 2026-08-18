param(
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$runnerScript = Join-Path $PSScriptRoot "run-scheduled-jobs.ps1"
$supplierRunnerScript = Join-Path $PSScriptRoot "run-supplier-orders.ps1"
$taskName = "Pharmacy Scheduled Jobs"
$supplierTaskName = "Pharmacy Supplier Ordering"
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
    if (-not (Test-Path -LiteralPath $supplierRunnerScript)) {
        throw "Supplier-order runner is missing: $supplierRunnerScript"
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

    # Supplier browsers need an interactive desktop, but must not inherit a
    # restrictive Job Object from Waitress. This on-demand task is the trusted
    # process broker used only when the web server detects that constraint.
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

    Write-InstallLog "Scheduled jobs and supplier launcher installed and self-tested for $RunAsUser."
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
