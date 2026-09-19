param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
$runtimePython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $runtimePython = (Get-Command python -ErrorAction Stop).Source
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $runtimePython -m research_memory @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Research memory command failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
