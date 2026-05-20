from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
INPUT = ROOT / "artifacts" / "data" / "mirage_eval_sample_1000.jsonl"
PRED_DIR = ROOT / "artifacts" / "predictions"
METRICS_DIR = ROOT / "artifacts" / "metrics"
LOG_DIR = ROOT / "artifacts" / "logs"

GROUPS = {
    "closed_oracle": [
        "qwen2_5_1_5b_instruct_closed_book",
        "qwen2_5_1_5b_instruct_oracle",
    ],
    "core_top5": [
        "qwen2_5_1_5b_instruct_closed_book",
        "qwen2_5_1_5b_instruct_oracle",
        "qwen2_5_1_5b_instruct_top5_mixture",
        "qwen2_5_1_5b_instruct_top5_dense",
        "qwen2_5_1_5b_instruct_mirage_mixed",
    ],
    "top1_optional": [
        "qwen2_5_1_5b_instruct_top1_mixture",
        "qwen2_5_1_5b_instruct_top1_dense",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HA5 LLM experiment group with logging and resume.")
    parser.add_argument("--group", default="core_top5", choices=sorted(GROUPS))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--context-budget-chars", type=int, default=6000)
    parser.add_argument("--min-expected-rows", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_and_log(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n==== {utc_now()} COMMAND ====\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    args = parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    experiments = GROUPS[args.group]
    state_path = LOG_DIR / f"{args.group}_run_state.json"
    write_state(
        state_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "group": args.group,
            "model": args.model,
            "experiment_total": len(experiments),
            "root": str(ROOT),
            "python": str(PYTHON),
        },
    )

    for index, experiment in enumerate(experiments, start=1):
        pred_path = PRED_DIR / f"{experiment}.jsonl"
        progress_path = PRED_DIR / f"{experiment}.progress.json"
        metric_path = METRICS_DIR / f"{experiment}_metrics.json"
        log_path = LOG_DIR / f"{experiment}.log"

        write_state(
            state_path,
            {
                "status": "running",
                "updated_at_utc": utc_now(),
                "group": args.group,
                "model": args.model,
                "experiment": experiment,
                "experiment_index": index,
                "experiment_total": len(experiments),
                "prediction_file": str(pred_path),
                "progress_file": str(progress_path),
                "log_file": str(log_path),
            },
        )

        run_cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_llm_qa.py"),
            "--experiment",
            experiment,
            "--input",
            str(INPUT),
            "--output",
            str(pred_path),
            "--progress",
            str(progress_path),
            "--model",
            args.model,
            "--device",
            args.device,
            "--min-expected-rows",
            str(args.min_expected_rows),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-input-tokens",
            str(args.max_input_tokens),
            "--context-budget-chars",
            str(args.context_budget_chars),
        ]
        exit_code = run_and_log(run_cmd, log_path)
        if exit_code:
            write_state(
                state_path,
                {
                    "status": "failed",
                    "updated_at_utc": utc_now(),
                    "group": args.group,
                    "model": args.model,
                    "experiment": experiment,
                    "experiment_index": index,
                    "experiment_total": len(experiments),
                    "exit_code": exit_code,
                    "log_file": str(log_path),
                },
            )
            raise SystemExit(exit_code)

        eval_cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "evaluate_predictions.py"),
            "--predictions",
            str(pred_path),
            "--output",
            str(metric_path),
            "--skip-bertscore",
        ]
        exit_code = run_and_log(eval_cmd, log_path)
        if exit_code:
            write_state(
                state_path,
                {
                    "status": "failed_evaluation",
                    "updated_at_utc": utc_now(),
                    "group": args.group,
                    "model": args.model,
                    "experiment": experiment,
                    "experiment_index": index,
                    "experiment_total": len(experiments),
                    "exit_code": exit_code,
                    "log_file": str(log_path),
                },
            )
            raise SystemExit(exit_code)

    write_state(
        state_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "group": args.group,
            "model": args.model,
            "experiment_total": len(experiments),
        },
    )


if __name__ == "__main__":
    main()
