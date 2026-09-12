param([Parameter(Mandatory=$true)][ValidatePattern('^v[0-9]+$')][string]$Version,
      [string]$Remote, [string]$RemoteRoot)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$configPath = Join-Path $PSScriptRoot 'remote.local.json'
if (Test-Path -LiteralPath $configPath) { $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json }
if (-not $Remote) { $Remote = $config.Remote }
if (-not $RemoteRoot) { $RemoteRoot = $config.RemoteRoot }
if ($RemoteRoot -notmatch '^/[A-Za-z0-9/_.-]+$' -or $Remote -notmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$') { throw 'Configure a valid Remote and RemoteRoot.' }
# Transfer all files over one SSH connection, as text to support Windows PowerShell.
$remoteCode = @'
import sys,re,json,base64
from pathlib import Path
root=Path(sys.argv[1]); version=sys.argv[2]
pkg=(root/'packages'/(version+'.latest')).read_text().strip()
m=re.fullmatch(r'CONV-('+re.escape(version)+r'-[0-9]{8}T[0-9]{6}-[0-9]+)\.zip',pkg)
if not m: sys.exit('Unexpected package name')
run=root/'logs'/m[1]
if not (run/'run.log').is_file(): sys.exit('Missing run log')
paths=[root/'packages'/pkg,root/'packages'/(pkg+'.sha256'),root/'scores.md']
paths+=sorted(p for p in run.rglob('*') if p.is_file())
files={}
for p in paths:
    if p.is_symlink() or root.resolve() not in p.resolve().parents: sys.exit('Unexpected path')
    files[p.relative_to(root).as_posix()]=base64.b64encode(p.read_bytes()).decode('ascii')
print(json.dumps(dict(package=pkg,run=m[1],files=files)))
'@
$code64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteCode))
$command = "python3 -c 'import base64; exec(base64.b64decode(""$code64""))' '$RemoteRoot' '$Version'"
$payload = & ssh -o StrictHostKeyChecking=yes $Remote $command
if ($LASTEXITCODE -ne 0) { throw 'Download failed; check connection and package.sh output.' }
$data = ($payload -join "`n") | ConvertFrom-Json
$package = $data.package
if ($package -notmatch "^CONV-($Version-[0-9]{8}T[0-9]{6}-[0-9]+)\.zip$" -or $data.run -ne $Matches[1]) { throw 'Unexpected package name' }
$runId = $data.run
$decoded = @{}
foreach ($item in $data.files.PSObject.Properties) {
    $name = $item.Name
    if ($name -match '\\|(^|/)\.\.?(/|$)' -or ($name -ne "packages/$package" -and $name -ne "packages/$package.sha256" -and $name -ne 'scores.md' -and -not $name.StartsWith("logs/$runId/"))) { throw 'Unexpected file path' }
    $target = [IO.Path]::GetFullPath((Join-Path $root $name))
    if (-not $target.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe path' }
    $decoded[$name] = [Convert]::FromBase64String($item.Value)
}
foreach ($required in @("packages/$package","packages/$package.sha256",'scores.md',"logs/$runId/run.log")) {
    if (-not $decoded.ContainsKey($required)) { throw "Missing $required" }
}
$sha = [Security.Cryptography.SHA256]::Create()
try { $actual = ([BitConverter]::ToString($sha.ComputeHash($decoded["packages/$package"]))).Replace('-','') } finally { $sha.Dispose() }
$expected = ([Text.Encoding]::UTF8.GetString($decoded["packages/$package.sha256"]) -split '\s+')[0]
if ($actual -ine $expected) { throw 'ZIP SHA256 mismatch' }
$marker = Join-Path $root "logs/$runId/.download-complete"
$alreadyDownloaded = Test-Path -LiteralPath $marker
foreach ($name in $decoded.Keys) {
    if ($alreadyDownloaded -and $name.StartsWith("logs/$runId/")) { continue }
    $target = Join-Path $root $name
    New-Item -ItemType Directory -Force (Split-Path $target -Parent) | Out-Null
    [IO.File]::WriteAllBytes($target, $decoded[$name])
}
Set-Content -LiteralPath $marker -Value $runId
Write-Host "Verified ZIP: $(Join-Path $root "packages/$package")"
Write-Host 'Downloaded package, logs and scores through one SSH connection.'
