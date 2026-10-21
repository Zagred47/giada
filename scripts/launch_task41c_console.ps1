param([string]$Session='giada-task41c-primary')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE='primary'
$identity=& "$PSScriptRoot/colab.ps1" whoami
if ($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch 'Email:\s*bellialessandro39@gmail\.com(\s|$)') {throw 'Primary identity not verified'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if ($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE') {throw 'Session not IDLE; inspect before dispatch'}
$requestPath='experiments/results/task41c_protocol_v1/dispatch_request.json'
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
$hash=(Get-FileHash -LiteralPath "$PSScriptRoot/task41b_console_manager.py" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $request.manager_sha256) {throw 'Manager changed'}
foreach ($pair in @(@('scripts/task41b_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@($requestPath,'/content/giada_task41c_launch.json'))) {
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) {throw 'Upload failed'}
}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task41b_local_supervisor.py --task 41c --session $Session --revision $request.revision --destination artifacts/task41c_primary_external
exit $LASTEXITCODE
