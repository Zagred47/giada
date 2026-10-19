# Explicit single dispatch through official Colab CLI; no automatic retry loop.
param([string]$Session='giada-task39h',[ValidateSet('primary','reserve')][string]$Profile='primary',[string]$ExpectedEmail='',[switch]$ReuseSession,[string]$Destination='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE=$Profile
if (-not $ExpectedEmail) {
    if ($Profile -eq 'primary') {$ExpectedEmail='bellialessandro39@gmail.com'}
    else {throw 'Provide the verified reserve account email before dispatch'}
}
$requestPath=Join-Path $repo 'experiments/results/task39h_protocol_v1/dispatch_request.json'
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
$identity=& "$PSScriptRoot/colab.ps1" whoami
if ($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch ('Email:\s*'+[regex]::Escape($ExpectedEmail)+'(\s|$)')) {throw 'Requested Colab identity not verified'}
if ((Get-FileHash "$PSScriptRoot/run_task39h_relaxation_identification.py" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $request.worker_sha256) {throw 'Worker hash changed'}
if ((Get-FileHash "$PSScriptRoot/task39e_console_manager.py" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $request.manager_sha256) {throw 'Manager hash changed'}
& "$PSScriptRoot/colab.ps1" sessions
if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect Colab assignments'}
$existing=& "$PSScriptRoot/colab.ps1" status -s $Session 2>$null
$missingPattern="Session '"+[regex]::Escape($Session)+"' not found"
if ($ReuseSession) {
    if (($existing -join "`n") -notmatch 'Status:\s*IDLE') {throw 'Reuse requires a verified IDLE session'}
} else {
    if (($existing -join "`n") -notmatch $missingPattern) {throw 'Named session exists or status is ambiguous; inspect before dispatch'}
    & "$PSScriptRoot/colab.ps1" new -s $Session --gpu T4
    if ($LASTEXITCODE -ne 0) {throw 'Colab T4 allocation failed; no experiment started'}
}
& "$PSScriptRoot/colab.ps1" exec -s $Session -f scripts/colab_smoke_test.py
if ($LASTEXITCODE -ne 0) {throw 'GPU console smoke failed'}
$transfers=@(
    @('scripts/task39e_console_manager.py','/content/task39e_console_manager.py'),
    @('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),
    @('experiments/results/task39h_protocol_v1/dispatch_request.json','/content/giada_task39h_launch.json')
)
foreach ($pair in $transfers) {
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) {throw "Upload failed: $($pair[0])"}
}
$python='C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not $Destination) {$Destination="artifacts/task39h_${Profile}_external"}
& $python scripts/task39e_local_supervisor.py --task 39h --session $Session --revision $request.revision --destination $Destination
exit $LASTEXITCODE
