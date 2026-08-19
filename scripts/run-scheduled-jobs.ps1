param(
    [switch]$SelfTest
)

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
            $djangoArguments = @("manage.py", "run_scheduled_jobs")
            if ($SelfTest) { $djangoArguments += "--self-test" }
            $output = & $python @djangoArguments 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }
        foreach ($line in $output) {
            Write-RunLog "$line"
        }
        if ($exitCode -ne 0) {
            $label = if ($SelfTest) { "Scheduled-job self-test" } else { "Scheduled jobs" }
            $failures.Add("$label returned exit code $exitCode.")
        }
    }

    if ($SelfTest) {
        if (-not (Test-Path -LiteralPath $backupScript)) {
            $failures.Add("Database backup script not found: $backupScript")
        }
        else {
            $backupOutput = @()
            $backupExitCode = 1
            $previousErrorPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $backupOutput = & powershell.exe -NoProfile -NonInteractive `
                    -ExecutionPolicy Bypass -File $backupScript -SelfTest 2>&1
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
                Write-RunLog "backup self-test: $line"
            }
            if ($backupExitCode -ne 0) {
                $failures.Add("Database backup self-test returned exit code $backupExitCode.")
            }
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
