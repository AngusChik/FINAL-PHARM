$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "env\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "supplier-launcher.log"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-RunLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Python environment not found: $python"
    }

    $env:DJANGO_SETTINGS_MODULE = "inventory.settings_production"
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $python "manage.py" "launch_supplier_orders" "--browser-smoke-if-idle" 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    foreach ($line in $output) {
        Write-RunLog "$line"
    }
    if ($exitCode -ne 0) {
        throw "Supplier launcher returned exit code $exitCode."
    }
}
catch {
    Write-RunLog "ERROR $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
