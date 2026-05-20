$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE "Source\dz5"
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\gpu-jupyter\python.exe"
$Runner = Join-Path $Root "scripts\run_llm_group.py"
$LogDir = Join-Path $Root "artifacts\logs"
$PidPath = Join-Path $LogDir "llm_group_launcher_pid.txt"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @($Runner, "--group", "core_top5") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -Path $PidPath -Encoding ASCII
Write-Output $process.Id
