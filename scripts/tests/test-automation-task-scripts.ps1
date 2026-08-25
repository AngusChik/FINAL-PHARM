$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$scriptsDirectory = Join-Path $projectRoot "scripts"
$parseTargets = @(
    "run-scheduled-jobs.ps1",
    "database-backup.ps1",
    "production.ps1",
    "install-automation-task.ps1",
    "install-database-backup-task.ps1",
    "setup-main-computer.ps1"
)

foreach ($name in $parseTargets) {
    $path = Join-Path $scriptsDirectory $name
    $tokens = $null
    $errors = $null
    [Management.Automation.Language.Parser]::ParseFile(
        $path, [ref]$tokens, [ref]$errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        throw "$name has a PowerShell parse error: $($errors[0])"
    }
}

$hiddenLauncher = Join-Path $scriptsDirectory "run-scheduled-jobs-hidden.vbs"
$cscript = Join-Path $env:SystemRoot "System32\cscript.exe"
& $cscript @("//B", "//NoLogo", $hiddenLauncher, "--probe")
if ($LASTEXITCODE -ne 0) {
    throw "Windowless launcher probe returned exit code $LASTEXITCODE."
}
$hiddenLauncherSource = Get-Content -LiteralPath $hiddenLauncher -Raw
if ($hiddenLauncherSource -notmatch '%SystemRoot%\\System32\\WindowsPowerShell' -or
    $hiddenLauncherSource -notmatch 'shell\.Run\(command, 0, True\)' -or
    $hiddenLauncherSource -notmatch 'runnerArguments = " -SelfTest"') {
    throw "The VBS launcher must use the trusted PowerShell path and window style 0."
}

$runnerSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "run-scheduled-jobs.ps1") -Raw
if ($runnerSource -match '-Reason\s+scheduled') {
    throw "The hourly runner must leave backup timing to ScheduledJobRun."
}
if ($runnerSource -notmatch '\$SelfTest' -or
    $runnerSource -notmatch '"run_scheduled_jobs"' -or
    $runnerSource -notmatch '"--self-test"' -or
    $runnerSource -notmatch 'database-backup\.ps1' -or
    $runnerSource -notmatch '-SelfTest') {
    throw "The hourly runner self-test must validate Django and backup prerequisites."
}

$installerSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "install-automation-task.ps1") -Raw
if ($installerSource -notmatch 'run-scheduled-jobs-hidden\.vbs' -or
    $installerSource -notmatch 'Pharmacy Database Backup' -or
    $installerSource -notmatch '--self-test' -or
    $installerSource -notmatch '-AllowStartIfOnBatteries' -or
    $installerSource -notmatch '-DontStopIfGoingOnBatteries' -or
    $installerSource -notmatch '"/ST", "00:00"' -or
    $installerSource -notmatch '"/RU", \$RunAsUser' -or
    $installerSource -notmatch '"/IT"' -or
    $installerSource -notmatch 'Start-ScheduledTask -TaskName \$taskName') {
    throw "The installer must use the windowless launcher and retire the legacy backup task."
}
$legacyRemovalIndex = $installerSource.IndexOf('Unregister-ScheduledTask')
$supplierSmokeIndex = $installerSource.IndexOf('Start-ScheduledTask -TaskName $supplierTaskName')
if ($legacyRemovalIndex -lt 0 -or $supplierSmokeIndex -lt 0 -or
    $legacyRemovalIndex -gt $supplierSmokeIndex) {
    throw "The legacy backup task must be retired before supplier smoke testing."
}
if ($installerSource -notmatch '-LogonType Interactive' -or
    $installerSource -notmatch 'Get-SignedInInteractiveUser' -or
    $installerSource -notmatch 'System32\\WindowsPowerShell\\v1\.0\\powershell\.exe' -or
    $installerSource -notmatch '-DontStopOnIdleEnd' -or
    $installerSource -notmatch '\$verifiedSupplier\.Principal\.LogonType') {
    throw "Supplier ordering must preserve the signed-in interactive user."
}

$backupSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "database-backup.ps1") -Raw
if ($backupSource -notmatch '\$BusinessDate' -or
    $backupSource -notmatch '\$NotBefore' -or
    $backupSource -notmatch '\$ForceNew' -or
    $backupSource -notmatch '\$SelfTest' -or
    $backupSource -notmatch 'Database backup prerequisites are available' -or
    $backupSource -notmatch '\[IO\.FileShare\]::None' -or
    $backupSource -notmatch 'PHARMACY_BACKUP_COMMAND_TIMEOUT_SECONDS' -or
    $backupSource -notmatch 'PHARMACY_BACKUP_LOCK_WAIT_SECONDS' -or
    $backupSource -notmatch 'System\.Diagnostics\.ProcessStartInfo' -or
    $backupSource -match 'Start-Process -FilePath \$FilePath' -or
    $backupSource -notmatch '\.WaitForExit\(' -or
    $backupSource -notmatch 'is still running:' -or
    $backupSource -notmatch '"--no-password"' -or
    $backupSource -notmatch 'taskkill\.exe' -or
    $backupSource -notmatch 'pharmacy-backup\.owner\.json') {
    throw "The backup script must enforce pre-closing candidates and prevent overlapping runs."
}
$backupSelfTestIndex = $backupSource.IndexOf('if ($SelfTest)')
$pgDumpInvocationIndex = $backupSource.IndexOf('-FilePath $pgDump')
if ($backupSelfTestIndex -lt 0 -or $pgDumpInvocationIndex -lt 0 -or
    $backupSelfTestIndex -gt $pgDumpInvocationIndex) {
    throw "Backup self-test must exit before pg_dump can create an archive."
}

$legacyInstallerSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "install-database-backup-task.ps1") -Raw
if ($legacyInstallerSource -match '02:00' -or
    $legacyInstallerSource -notmatch 'install-automation-task\.ps1' -or
    $legacyInstallerSource -notmatch '"-RunAsUser", \$RunAsUser') {
    throw "The legacy backup entry point must delegate to pre-closing automation."
}

$setupSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "setup-main-computer.ps1") -Raw
if ($setupSource -notmatch '\[string\]\$RunAsUser' -or
    $setupSource -notmatch '\$configuration\.ContainsKey\("DB_PASSWORD"\)' -or
    $setupSource -match 'if \(\$createdEnv\) \{' -or
    $setupSource -notmatch '"-RunAsUser", \$RunAsUser') {
    throw "Main-computer setup must preserve the supplier user's identity across elevation."
}

$productionSource = Get-Content `
    -LiteralPath (Join-Path $scriptsDirectory "production.ps1") -Raw
if ($productionSource -notmatch 'function Ensure-DatabaseLogin' -or
    $productionSource -notmatch 'function Read-DatabasePassword' -or
    $productionSource -notmatch '-AsSecureString' -or
    $productionSource -notmatch 'Database password verified and saved in \.env' -or
    $productionSource -notmatch 'Ensure-DatabaseLogin \$config') {
    throw "Production startup must repair and verify a missing database password interactively."
}

Write-Host "automation task script tests passed"
