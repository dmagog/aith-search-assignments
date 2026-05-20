$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\gpu-jupyter\python.exe"
$InputPath = Join-Path $Root "artifacts\data\mirage_eval_sample_1000.jsonl"
$PredDir = Join-Path $Root "artifacts\predictions"
$LogDir = Join-Path $Root "artifacts\logs"
$MetricsDir = Join-Path $Root "artifacts\metrics"
$RunStatePath = Join-Path $LogDir "gpu_core_top5_run_state.json"

New-Item -ItemType Directory -Force -Path $PredDir, $LogDir, $MetricsDir | Out-Null

$Model = "Qwen/Qwen2.5-1.5B-Instruct"
$Experiments = @(
    "qwen2_5_1_5b_instruct_closed_book",
    "qwen2_5_1_5b_instruct_oracle",
    "qwen2_5_1_5b_instruct_top5_mixture",
    "qwen2_5_1_5b_instruct_top5_dense",
    "qwen2_5_1_5b_instruct_mirage_mixed"
)

function Write-RunState {
    param(
        [string] $Status,
        [string] $Experiment,
        [int] $Index,
        [int] $Total,
        [int] $ExitCode = 0
    )
    $payload = [ordered]@{
        status = $Status
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        model = $Model
        experiment = $Experiment
        experiment_index = $Index
        experiment_total = $Total
        exit_code = $ExitCode
        root = $Root
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -Path $RunStatePath -Encoding UTF8
}

Write-RunState -Status "starting" -Experiment "" -Index 0 -Total $Experiments.Count

for ($i = 0; $i -lt $Experiments.Count; $i++) {
    $Experiment = $Experiments[$i]
    $OutPath = Join-Path $PredDir "$Experiment.jsonl"
    $LogPath = Join-Path $LogDir "$Experiment.log"
    $ProgressPath = Join-Path $PredDir "$Experiment.progress.json"

    Write-RunState -Status "running" -Experiment $Experiment -Index ($i + 1) -Total $Experiments.Count
    Add-Content -Path $LogPath -Encoding UTF8 -Value ""
    Add-Content -Path $LogPath -Encoding UTF8 -Value "==== $(Get-Date -Format o) START $Experiment ===="

    & $Python `
        (Join-Path $Root "scripts\run_llm_qa.py") `
        --experiment $Experiment `
        --input $InputPath `
        --output $OutPath `
        --progress $ProgressPath `
        --model $Model `
        --device cuda `
        --min-expected-rows 1000 `
        --batch-size 4 `
        --max-new-tokens 16 `
        --max-input-tokens 3072 `
        --context-budget-chars 6000 `
        2>&1 | Tee-Object -FilePath $LogPath -Append

    $RunExitCode = $LASTEXITCODE
    if ($RunExitCode -ne 0) {
        Write-RunState -Status "failed" -Experiment $Experiment -Index ($i + 1) -Total $Experiments.Count -ExitCode $RunExitCode
        throw "Experiment $Experiment failed with exit code $RunExitCode"
    }

    $MetricPath = Join-Path $MetricsDir "$($Experiment)_metrics.json"
    & $Python `
        (Join-Path $Root "scripts\evaluate_predictions.py") `
        --predictions $OutPath `
        --output $MetricPath `
        --skip-bertscore `
        2>&1 | Tee-Object -FilePath $LogPath -Append

    $EvalExitCode = $LASTEXITCODE
    if ($EvalExitCode -ne 0) {
        Write-RunState -Status "failed_evaluation" -Experiment $Experiment -Index ($i + 1) -Total $Experiments.Count -ExitCode $EvalExitCode
        throw "Evaluation for $Experiment failed with exit code $EvalExitCode"
    }

    Add-Content -Path $LogPath -Encoding UTF8 -Value "==== $(Get-Date -Format o) DONE $Experiment ===="
}

Write-RunState -Status "complete" -Experiment "" -Index $Experiments.Count -Total $Experiments.Count
