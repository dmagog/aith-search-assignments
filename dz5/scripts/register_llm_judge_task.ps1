$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE "Source\dz5"
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\gpu-jupyter\python.exe"
$Runner = Join-Path $Root "scripts\run_llm_judge.py"
$TaskName = "DZ5LLMJudge"

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$Runner`" --device cuda --batch-size 8" `
    -WorkingDirectory $Root

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
