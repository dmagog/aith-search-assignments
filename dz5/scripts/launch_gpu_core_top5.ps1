$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE "Source\dz5"
$Runner = Join-Path $Root "scripts\run_gpu_core_top5.ps1"
$PidPath = Join-Path $Root "artifacts\logs\gpu_core_top5_launcher_pid.txt"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PidPath) | Out-Null

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -Path $PidPath -Encoding ASCII
Write-Output $process.Id
