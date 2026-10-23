param([string]$Session='giada-task41h-eval-primary')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE='primary'
$identity=& "$PSScriptRoot/colab.ps1" whoami
if($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch 'Email:\s*bellialessandro39@gmail.com'){throw 'Wrong primary account'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE'){throw 'Session must be idle'}
& "$PSScriptRoot/colab.ps1" exec -s $Session -f scripts/colab_smoke_test.py --timeout 60
if($LASTEXITCODE -ne 0){throw 'GPU smoke failed'}
$requestPath='experiments/results/task41h_recovery_v1/dispatch_request.json'
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
foreach($pair in @(@('scripts/task41b_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@($requestPath,'/content/giada_task41h_launch.json'))){
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if($LASTEXITCODE -ne 0){throw 'Upload failed'}
}
foreach($inputFile in $request.inputs){
    & "$PSScriptRoot/colab.ps1" upload -s $Session $inputFile.local $inputFile.remote
    if($LASTEXITCODE -ne 0){throw 'Input upload failed'}
}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task41b_local_supervisor.py --task 41h --session $Session --revision $request.revision --destination artifacts/task41h_evaluation_primary_external
exit $LASTEXITCODE
