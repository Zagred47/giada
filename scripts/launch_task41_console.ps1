param([string]$Session='giada-task41-primary',[string]$Request='experiments/results/task41_protocol_v1/dispatch_request.json',[string]$Destination='artifacts/task41_primary_external')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE='primary'
$identity=& "$PSScriptRoot/colab.ps1" whoami
if ($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch 'Email:\s*bellialessandro39@gmail\.com(\s|$)') {throw 'Primary identity not verified'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if ($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE') {throw 'Session not IDLE; inspect instead of duplicate launch'}
$request=Get-Content -LiteralPath $Request -Raw | ConvertFrom-Json
$managerBytes=[IO.File]::ReadAllBytes("$PSScriptRoot/task39e_console_manager.py")
$managerSha=[Security.Cryptography.SHA256]::Create()
try {$managerHash=([BitConverter]::ToString($managerSha.ComputeHash($managerBytes))).Replace('-','').ToLowerInvariant()} finally {$managerSha.Dispose()}
if ($managerHash -ne $request.manager_sha256) {throw 'Manager changed'}
foreach ($pair in @(@('scripts/task39e_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@($Request,'/content/giada_task41_launch.json'))) {
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) {throw 'Upload failed'}
}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task39e_local_supervisor.py --task 41 --session $Session --revision $request.revision --destination $Destination
exit $LASTEXITCODE
