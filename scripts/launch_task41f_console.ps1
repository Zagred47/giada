param([string]$Session='giada-task41f-reserve',[ValidateSet('primary','reserve')][string]$Profile='reserve')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$env:GIADA_COLAB_PROFILE=$Profile
$expectedEmail=if ($Profile -eq 'primary') {'bellialessandro39@gmail.com'} else {'cinziamrg48@gmail.com'}
$identity=& "$PSScriptRoot/colab.ps1" whoami
if($LASTEXITCODE -ne 0 -or ($identity -join "`n") -notmatch ('Email:\s*'+[regex]::Escape($expectedEmail)+'(\s|$)')){throw 'Account not verified'}
$status=& "$PSScriptRoot/colab.ps1" status -s $Session
if($LASTEXITCODE -ne 0 -or ($status -join "`n") -notmatch 'Status:\s*IDLE'){throw 'Session must be verified idle before launch'}
$requestPath='experiments/results/task41f_protocol_v1/dispatch_request.json'
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
if($request.profile -ne $Profile){throw 'Profile differs from GPU preflight'}
foreach($pair in @(@('scripts/task41b_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@($requestPath,'/content/giada_task41f_launch.json'),@('artifacts/task41b_opened_inputs.zip','/content/task41b_opened_inputs.zip'),@('artifacts/task41d_opened41c.zip','/content/task41d_opened41c.zip'))){
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if($LASTEXITCODE -ne 0){throw 'Upload failed'}
}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task41b_local_supervisor.py --task 41f --session $Session --revision $request.revision --destination "artifacts/task41f_${Profile}_external"
exit $LASTEXITCODE
