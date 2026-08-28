param(
    [string]$ServerUrl = "",
    [string]$CertificatePath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ServerUrl) {
    $serverUrlFile = Join-Path $projectRoot "server-url.txt"
    if (Test-Path -LiteralPath $serverUrlFile -PathType Leaf) {
        $ServerUrl = (Get-Content -LiteralPath $serverUrlFile -Raw).Trim()
    }
    else {
        $ServerUrl = Read-Host "Enter the Pharmacy server URL (for example https://192.168.0.15)"
    }
}
$ServerUrl = $ServerUrl.Trim().TrimEnd('/')

$uri = $null
if (-not [Uri]::TryCreate($ServerUrl, [UriKind]::Absolute, [ref]$uri) -or
    $uri.Scheme -ne "https" -or
    -not $uri.Host -or
    $uri.UserInfo -or
    $uri.Query -or
    $uri.Fragment -or
    $uri.AbsolutePath -ne "/") {
    throw "ServerUrl must be an HTTPS origin such as https://192.168.0.15 with no path, query, or credentials."
}

if (-not $CertificatePath) {
    $CertificatePath = Join-Path $projectRoot "Pharmacy-Root-Certificate.crt"
}
if (Test-Path -LiteralPath $CertificatePath) {
    $resolvedCertificate = (Resolve-Path -LiteralPath $CertificatePath).Path
    $certificate = New-Object Security.Cryptography.X509Certificates.X509Certificate2(
        $resolvedCertificate
    )
    & certutil.exe -user -addstore Root $resolvedCertificate | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "The Pharmacy root certificate could not be installed."
    }
    $installed = Get-ChildItem -LiteralPath "Cert:\CurrentUser\Root\$($certificate.Thumbprint)" `
        -ErrorAction SilentlyContinue
    if (-not $installed) {
        throw "The Pharmacy root certificate was not found after installation."
    }
    Write-Host "Trusted the Pharmacy server certificate for this Windows user." -ForegroundColor Green
}
else {
    Write-Host (
        "Certificate not found at '$CertificatePath'. The shortcut will be installed, " +
        "but HTTPS may show a trust warning until the public root certificate is installed."
    ) -ForegroundColor Yellow
}

$shortcutContent = @(
    "[InternetShortcut]",
    "URL=$ServerUrl",
    "IconFile=$env:SystemRoot\System32\SHELL32.dll",
    "IconIndex=220"
)
$shortcutLocations = @(
    [Environment]::GetFolderPath("Desktop"),
    [Environment]::GetFolderPath("Programs")
)
foreach ($location in $shortcutLocations) {
    if (-not $location) { throw "Windows Desktop or Start Menu could not be located." }
    New-Item -ItemType Directory -Force -Path $location | Out-Null
    Set-Content -LiteralPath (Join-Path $location "Pharmacy.url") `
        -Value $shortcutContent -Encoding ASCII
}

Write-Host "Pharmacy workstation shortcut installed for $ServerUrl" -ForegroundColor Green
Write-Host "This workstation contains only the HTTPS shortcut and public certificate."
