param([string]$Session='giada-task41-primary')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE='primary'
$identity=& "$PSScriptRoot/colab.ps1" whoami
if ($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch 'Email:\s*bellialessandro39@gmail\.com(\s|$)') {throw 'Primary identity not verified'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if ($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE') {throw 'Session not IDLE; inspect instead of duplicate launch'}
$request=Get-Content experiments/results/task41_protocol_v1/dispatch_request.json -Raw | ConvertFrom-Json
if ((Get-FileHash "$PSScriptRoot/task39e_console_manager.py" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $request.manager_sha256) {throw 'Manager changed'}
foreach ($pair in @(@('scripts/task39e_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@('experiments/results/task41_protocol_v1/dispatch_request.json','/content/giada_task41_launch.json'))) {
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) {throw 'Upload failed'}
}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task39e_local_supervisor.py --task 41 --session $Session --revision $request.revision --destination artifacts/task41_primary_external
exit $LASTEXITCODE
