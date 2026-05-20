$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE "Source\dz5"
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\gpu-jupyter\python.exe"
$Runner = Join-Path $Root "scripts\run_t5gemma_pipeline.py"
$TaskName = "DZ5T5Gemma"

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$Runner`" --max-eval-samples 256 --num-train-epochs 1 --train-batch-size 1 --eval-batch-size 2 --gradient-accumulation-steps 16 --save-steps 250 --eval-steps 1000 --generation-batch-size 2 --skip-bertscore" `
    -WorkingDirectory $Root

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
