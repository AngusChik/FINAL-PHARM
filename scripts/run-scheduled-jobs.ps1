$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "scheduled-jobs.log"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
if (-not (Test-Path -LiteralPath $python)) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) ERROR Python environment not found: $python"
    throw "Python environment not found: $python"
}

Push-Location $projectRoot
try {
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
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $line"
    }
    if ($exitCode -ne 0) {
        throw "Scheduled jobs returned exit code $exitCode."
    }
}
catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) ERROR $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
