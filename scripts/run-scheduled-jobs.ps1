$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "scheduled-jobs.log"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-RunLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

Push-Location $projectRoot
try {
    $failures = New-Object System.Collections.Generic.List[string]
    $runAt = Get-Date

    # The dispatcher runs hourly. From 02:00 local time onward, ask the backup
    # script for today's scheduled backup. That script re-validates an existing
    # artifact before skipping, so a failed/corrupt attempt is retried on the
    # next hourly invocation without producing duplicate valid backups.
    if ($runAt.TimeOfDay -ge [TimeSpan]::FromHours(2)) {
        if (-not (Test-Path -LiteralPath $backupScript)) {
            $failures.Add("Scheduled database backup script not found: $backupScript")
        }
        else {
            $backupOutput = @()
            $backupExitCode = 1
            $previousErrorPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $backupOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
                    -File $backupScript -Reason scheduled 2>&1
                $backupExitCode = $LASTEXITCODE
            }
            catch {
                $backupOutput = @($_)
                $backupExitCode = 1
            }
            finally {
                $ErrorActionPreference = $previousErrorPreference
            }
            foreach ($line in $backupOutput) {
                Write-RunLog "database backup: $line"
            }
            if ($backupExitCode -ne 0) {
                $failures.Add("Scheduled database backup returned exit code $backupExitCode.")
            }
        }
    }

    if (-not (Test-Path -LiteralPath $python)) {
        $failures.Add("Python environment not found: $python")
    }
    else {
        $env:DJANGO_SETTINGS_MODULE = "inventory.settings_production"
        # Windows PowerShell turns native stderr lines into PowerShell error
        # records. Django and Axes write harmless startup diagnostics there, so
        # temporarily allow those records through and trust the process exit code.
        $previousErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $python "manage.py" "run_scheduled_jobs" 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }
        foreach ($line in $output) {
            Write-RunLog "$line"
        }
        if ($exitCode -ne 0) {
            $failures.Add("Scheduled jobs returned exit code $exitCode.")
        }
    }

    if ($failures.Count -gt 0) {
        throw ($failures -join " ")
    }
}
catch {
    Write-RunLog "ERROR $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
