param([Parameter(Mandatory=$true)][ValidatePattern('^v[0-9]+$')][string]$Version,
      [string]$Remote, [string]$RemoteRoot)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$configPath = Join-Path $PSScriptRoot 'remote.local.json'
if (Test-Path -LiteralPath $configPath) { $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json }
if (-not $Remote) { $Remote = $config.Remote }
if (-not $RemoteRoot) { $RemoteRoot = $config.RemoteRoot }
if ($RemoteRoot -notmatch '^/[A-Za-z0-9/_.-]+$' -or -not $Remote) { throw 'Configure a valid Remote and RemoteRoot.' }
$package = (& ssh -o StrictHostKeyChecking=yes $Remote "cat '$RemoteRoot/packages/$Version.latest'")
if ($LASTEXITCODE -ne 0) { throw 'No package pointer; run package.sh on the server first.' }
$package = ($package -join '').Trim()
if ($package -notmatch "^CONV-($Version-[0-9TZ]+-[0-9]+)\.zip$") { throw 'Unexpected package name' }
$runId = $Matches[1]
$packageDir = Join-Path $root 'packages'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $packageDir,$logDir | Out-Null
$remoteFiles = @("${Remote}:$RemoteRoot/packages/$package", "${Remote}:$RemoteRoot/packages/$package.sha256")
& scp -o StrictHostKeyChecking=yes @remoteFiles $packageDir
if ($LASTEXITCODE -ne 0) { throw 'Package download failed' }
$expected = ((Get-Content -Raw (Join-Path $packageDir "$package.sha256")) -split '\s+')[0]
$actual = (Get-FileHash (Join-Path $packageDir $package) -Algorithm SHA256).Hash
if ($actual -ine $expected) { throw 'ZIP SHA256 mismatch' }
$localRun = Join-Path $logDir $runId
$completeMarker = Join-Path $localRun '.download-complete'
if (Test-Path -LiteralPath $completeMarker) {
    Write-Host 'This immutable run directory was already downloaded; leaving it in place.'
} else {
    & scp -r -o StrictHostKeyChecking=yes "${Remote}:$RemoteRoot/logs/$runId" $logDir
    if ($LASTEXITCODE -ne 0) { throw 'Log download failed' }
    Set-Content -LiteralPath $completeMarker -Value $runId
}
& scp -o StrictHostKeyChecking=yes "${Remote}:$RemoteRoot/scores.md" $root
if ($LASTEXITCODE -ne 0) { throw 'Score download failed' }
Write-Host "Verified ZIP: $(Join-Path $packageDir $package)"
Write-Host 'Downloaded logs and scores.md. Submit the ZIP on the website yourself.'
