param([string]$Remote, [string]$RemoteRoot)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$configPath = Join-Path $PSScriptRoot 'remote.local.json'
if (Test-Path -LiteralPath $configPath) { $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json }
if (-not $Remote) { $Remote = $config.Remote }
if (-not $RemoteRoot) { $RemoteRoot = $config.RemoteRoot }
if (-not $Remote -or -not $RemoteRoot) { throw 'Set tools/remote.local.json or pass -Remote and -RemoteRoot.' }
$source = Join-Path $root 'CONV/conv2d.c'
& scp -o StrictHostKeyChecking=yes $source "${Remote}:$RemoteRoot/CONV/conv2d.c"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
Write-Host 'Uploaded only CONV/conv2d.c. Next on server: bash tools/test.sh v01'
