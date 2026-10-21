param([string]$Session='giada-task40b-primary')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE='primary'
$identity=& "$PSScriptRoot/colab.ps1" whoami
if ($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch 'Email:\s*bellialessandro39@gmail\.com(\s|$)') {throw 'Primary identity not verified'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if ($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE') {throw 'Session is not verified IDLE'}
$request=Get-Content experiments/results/task40b_protocol_v1/dispatch_request.json -Raw | ConvertFrom-Json
$sha=[System.Security.Cryptography.SHA256]::Create()
try {$managerHash=([System.BitConverter]::ToString($sha.ComputeHash([System.IO.File]::ReadAllBytes("$PSScriptRoot/task39e_console_manager.py")))).Replace('-','').ToLowerInvariant()} finally {$sha.Dispose()}
if ($managerHash -ne $request.manager_sha256) {throw 'Manager changed'}
foreach ($pair in @(@('scripts/task39e_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@('experiments/results/task40b_protocol_v1/dispatch_request.json','/content/giada_task40b_launch.json'))) {
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) {throw 'Upload failed'}
}
$python='C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python scripts/task39e_local_supervisor.py --task 40b --session $Session --revision $request.revision --destination artifacts/task40b_primary_external
exit $LASTEXITCODE
