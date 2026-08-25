param(
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerScript = $PSCommandPath
$automationInstaller = Join-Path $PSScriptRoot "install-automation-task.ps1"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Host "Administrator access is required to install pharmacy automation." -ForegroundColor Yellow
    if (-not $RunAsUser) {
        $RunAsUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    }
    $arguments = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$installerScript`" " +
        "-RunAsUser `"$RunAsUser`""
    )
    $elevated = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
    exit $elevated.ExitCode
}

try {
    if (-not (Test-Path -LiteralPath $automationInstaller)) {
        throw "Pharmacy automation installer is missing: $automationInstaller"
    }
    $automationArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $automationInstaller
    )
    if ($RunAsUser) {
        $automationArguments += @("-RunAsUser", $RunAsUser)
    }
    & powershell.exe @automationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pharmacy automation installer returned exit code $LASTEXITCODE."
    }

    Write-Host "Pre-closing database backup automation installed." -ForegroundColor Green
    Write-Host "The backup runs once per open business day, one hour before closing."
    Write-Host "Backups are also created before every production start/update."
    Write-Host "Project: $projectRoot"
    exit 0
}
catch {
    Write-Host "Could not install pre-closing backup automation: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
