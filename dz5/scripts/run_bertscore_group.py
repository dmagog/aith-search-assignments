from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PRED_DIR = ROOT / "artifacts" / "predictions"
METRICS_DIR = ROOT / "artifacts" / "metrics"
LOG_DIR = ROOT / "artifacts" / "logs"

EXPERIMENTS = [
    "roberta_oracle_full",
    "roberta_top1_mixture_full",
    "roberta_top1_dense_full",
    "qwen2_5_1_5b_instruct_closed_book",
    "qwen2_5_1_5b_instruct_oracle",
    "qwen2_5_1_5b_instruct_top1_mixture",
    "qwen2_5_1_5b_instruct_top1_dense",
    "qwen2_5_1_5b_instruct_top5_mixture",
    "qwen2_5_1_5b_instruct_top5_dense",
    "qwen2_5_1_5b_instruct_mirage_mixed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute HA5 metrics with BERTScore.")
    parser.add_argument("--experiments", nargs="*", default=EXPERIMENTS)
    parser.add_argument("--bertscore-model", default="bert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default="cuda")
    parser.add_argument("--bertscore-local-files-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "bertscore_group_run_state.json"
    log_path = LOG_DIR / "bertscore_group.log"

    write_state(
        state_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "experiment_total": len(args.experiments),
            "bertscore_model": args.bertscore_model,
            "bertscore_device": args.bertscore_device,
            "python": str(PYTHON),
            "root": str(ROOT),
        },
    )

    for index, experiment in enumerate(args.experiments, start=1):
        pred_path = PRED_DIR / f"{experiment}.jsonl"
        metric_path = METRICS_DIR / f"{experiment}_metrics.json"
        if not pred_path.exists():
            write_state(
                state_path,
                {
                    "status": "failed_missing_predictions",
                    "updated_at_utc": utc_now(),
                    "experiment": experiment,
                    "experiment_index": index,
                    "experiment_total": len(args.experiments),
                    "prediction_file": str(pred_path),
                },
            )
            raise SystemExit(f"missing predictions: {pred_path}")

        write_state(
            state_path,
            {
                "status": "running",
                "updated_at_utc": utc_now(),
                "experiment": experiment,
                "experiment_index": index,
                "experiment_total": len(args.experiments),
                "prediction_file": str(pred_path),
                "metrics_file": str(metric_path),
                "bertscore_model": args.bertscore_model,
                "bertscore_device": args.bertscore_device,
                "log_file": str(log_path),
            },
        )

        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "evaluate_predictions.py"),
            "--predictions",
            str(pred_path),
            "--output",
            str(metric_path),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
            "--bertscore-model",
            args.bertscore_model,
            "--bertscore-batch-size",
            str(args.bertscore_batch_size),
            "--bertscore-device",
            args.bertscore_device,
        ]
        if args.bertscore_local_files_only:
            command.append("--bertscore-local-files-only")

        exit_code = run_and_log(command, log_path)
        if exit_code:
            write_state(
                state_path,
                {
                    "status": "failed",
                    "updated_at_utc": utc_now(),
                    "experiment": experiment,
                    "experiment_index": index,
                    "experiment_total": len(args.experiments),
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
            "experiment_total": len(args.experiments),
            "bertscore_model": args.bertscore_model,
            "bertscore_device": args.bertscore_device,
        },
    )


if __name__ == "__main__":
    main()
