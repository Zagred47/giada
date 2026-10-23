param([string]$Session='giada-task41h-frozen-resume')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$env:GIADA_COLAB_PROFILE='primary'
$requestPath='artifacts/task41h_frozen_resume/dispatch_request.json'
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
foreach($pair in @(@('scripts/task41b_console_manager.py','/content/task39e_console_manager.py'),@('scripts/colab_task39e_driver.py','/content/colab_task39e_driver.py'),@($requestPath,'/content/giada_task41h_launch.json'))){
    & "$PSScriptRoot/colab.ps1" upload -s $Session $pair[0] $pair[1]
    if($LASTEXITCODE -ne 0){throw 'Control file upload failed'}
}
foreach($inputFile in $request.inputs){
    foreach($part in $inputFile.parts){
        & "$PSScriptRoot/colab.ps1" upload -s $Session $part.local $part.remote
        if($LASTEXITCODE -ne 0){throw 'Input part upload failed'}
    }
}
& "$PSScriptRoot/colab.ps1" exec -s $Session -f scripts/assemble_task41h_frozen_resume.py --timeout 120
if($LASTEXITCODE -ne 0){throw 'Input assembly/verification failed'}
& 'C:/Users/belli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/task41b_local_supervisor.py --task 41h --session $Session --revision $request.revision --destination artifacts/task41h_frozen_evaluation_primary_external
exit $LASTEXITCODE
