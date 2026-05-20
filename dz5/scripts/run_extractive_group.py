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

EXPERIMENTS = [
    "roberta_oracle_full",
    "roberta_top1_mixture_full",
    "roberta_top1_dense_full",
]

EXPERIMENT_ARG = {
    "roberta_oracle_full": "roberta_oracle",
    "roberta_top1_mixture_full": "roberta_top1_mixture",
    "roberta_top1_dense_full": "roberta_top1_dense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full extractive QA experiments with logging and resume.")
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-expected-rows", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
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
    state_path = LOG_DIR / "extractive_full_run_state.json"

    write_state(
        state_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "experiment_total": len(EXPERIMENTS),
            "root": str(ROOT),
            "python": str(PYTHON),
        },
    )

    for index, output_stem in enumerate(EXPERIMENTS, start=1):
        experiment = EXPERIMENT_ARG[output_stem]
        pred_path = PRED_DIR / f"{output_stem}.jsonl"
        progress_path = PRED_DIR / f"{output_stem}.progress.json"
        metric_path = METRICS_DIR / f"{output_stem}_metrics.json"
        log_path = LOG_DIR / f"{output_stem}.log"

        write_state(
            state_path,
            {
                "status": "running",
                "updated_at_utc": utc_now(),
                "model": args.model,
                "experiment": output_stem,
                "experiment_index": index,
                "experiment_total": len(EXPERIMENTS),
                "prediction_file": str(pred_path),
                "progress_file": str(progress_path),
                "log_file": str(log_path),
            },
        )

        run_cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_extractive_qa.py"),
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
            "--max-seq-len",
            str(args.max_seq_len),
            "--doc-stride",
            str(args.doc_stride),
        ]
        exit_code = run_and_log(run_cmd, log_path)
        if exit_code:
            write_state(
                state_path,
                {
                    "status": "failed",
                    "updated_at_utc": utc_now(),
                    "model": args.model,
                    "experiment": output_stem,
                    "experiment_index": index,
                    "experiment_total": len(EXPERIMENTS),
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
                    "model": args.model,
                    "experiment": output_stem,
                    "experiment_index": index,
                    "experiment_total": len(EXPERIMENTS),
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
            "model": args.model,
            "experiment_total": len(EXPERIMENTS),
        },
    )


if __name__ == "__main__":
    main()
