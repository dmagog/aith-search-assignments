$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE "Source\dz5"
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\gpu-jupyter\python.exe"
$Runner = Join-Path $Root "scripts\run_t5gemma_pipeline.py"
$TaskName = "DZ5T5GemmaSmoke"

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$Runner`" --output-dir `"$Root\artifacts\models\t5gemma_squad_lora_smoke`" --max-train-samples 128 --max-eval-samples 32 --num-train-epochs 1 --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 --save-steps 50 --eval-steps 50 --generation-batch-size 1 --skip-eval" `
    -WorkingDirectory $Root

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
