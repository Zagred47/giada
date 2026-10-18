# Official Colab CLI through its supported Linux platform (Ubuntu WSL2).
# No credentials or session tokens belong in this script/repository.
# Keep a simple script: advanced-function common parameters would intercept
# CLI flags such as -o (OutVariable/OutBuffer) before they reach Colab.
$ColabArguments = @($args)
$ErrorActionPreference = 'Stop'
function Convert-ColabLocalPath([string]$Value) {
    if ($Value -match '^([A-Za-z]):[\\/](.*)$') {
        return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\','/')
    }
    return $Value
}
$linuxPython = '/home/belli/.local/share/giada-colab-cli/venv/bin/python'
$linuxAdapter = Convert-ColabLocalPath (Join-Path $PSScriptRoot 'colab_checked_exec.py')
$linuxCwd = Convert-ColabLocalPath (Get-Location).Path
$forwarded = @($ColabArguments | ForEach-Object {
    if ($_ -match '^\.\.?\\') { $_.Replace('\','/') }
    else { Convert-ColabLocalPath $_ }
})
if ($forwarded.Count -eq 0) { $forwarded = @('--help') }
# Explicit OAuth2 avoids silently selecting unrelated ambient GCP credentials.
& wsl.exe --distribution Ubuntu --cd $linuxCwd --exec $linuxPython $linuxAdapter @forwarded
exit $LASTEXITCODE
