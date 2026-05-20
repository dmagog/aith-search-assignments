from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LOG_DIR = ROOT / "artifacts" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T5Gemma fine-tuning and MIRAGE evaluation.")
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "models" / "t5gemma_squad_lora")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "t5gemma_pipeline_state.json"
    log_path = LOG_DIR / "t5gemma_pipeline.log"
    write_state(
        state_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
        },
    )

    if not args.skip_train:
        train_cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_t5gemma_finetune.py"),
            "--model",
            args.model,
            "--output-dir",
            str(args.output_dir),
            "--max-eval-samples",
            str(args.max_eval_samples),
            "--num-train-epochs",
            str(args.num_train_epochs),
            "--per-device-train-batch-size",
            str(args.train_batch_size),
            "--per-device-eval-batch-size",
            str(args.eval_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--save-steps",
            str(args.save_steps),
            "--eval-steps",
            str(args.eval_steps),
        ]
        if args.max_train_samples is not None:
            train_cmd.extend(["--max-train-samples", str(args.max_train_samples)])
        if args.local_files_only:
            train_cmd.append("--local-files-only")

        write_state(state_path, {"status": "training", "updated_at_utc": utc_now(), "command": train_cmd})
        exit_code = run_and_log(train_cmd, log_path)
        if exit_code:
            write_state(state_path, {"status": "failed_training", "updated_at_utc": utc_now(), "exit_code": exit_code})
            raise SystemExit(exit_code)

    if not args.skip_eval:
        eval_cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_t5gemma_eval.py"),
            "--base-model",
            args.model,
            "--adapter-dir",
            str(args.output_dir),
            "--batch-size",
            str(args.generation_batch_size),
        ]
        if args.skip_bertscore:
            eval_cmd.append("--skip-bertscore")
        if args.local_files_only:
            eval_cmd.append("--local-files-only")

        write_state(state_path, {"status": "evaluating", "updated_at_utc": utc_now(), "command": eval_cmd})
        exit_code = run_and_log(eval_cmd, log_path)
        if exit_code:
            write_state(state_path, {"status": "failed_evaluation", "updated_at_utc": utc_now(), "exit_code": exit_code})
            raise SystemExit(exit_code)

    write_state(
        state_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
        },
    )


if __name__ == "__main__":
    main()
