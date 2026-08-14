$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$backupScript = Join-Path $PSScriptRoot "database-backup.ps1"
$taskName = "Pharmacy Database Backup"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Host "Administrator access is required to install the reliable daily backup task." -ForegroundColor Yellow
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$installerScript`""
    $elevated = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
    exit $elevated.ExitCode
}

try {
    if (-not (Test-Path -LiteralPath $backupScript)) {
        throw "Database backup script is missing: $backupScript"
    }
    $taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$backupScript`" -Reason scheduled"
    & schtasks.exe @(
        "/Create", "/TN", $taskName, "/TR", $taskCommand,
        "/SC", "DAILY", "/ST", "02:00", "/RU", "SYSTEM", "/RL", "HIGHEST", "/F"
    )
    if ($LASTEXITCODE -ne 0) { throw "Windows Task Scheduler returned exit code $LASTEXITCODE." }

    Write-Host "Daily database backup installed for 2:00 AM." -ForegroundColor Green
    Write-Host "Backups are also created before every production start/update."
    Write-Host "Project: $projectRoot"
    exit 0
}
catch {
    Write-Host "Could not install the daily backup task: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
