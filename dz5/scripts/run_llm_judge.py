from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "artifacts" / "predictions"
JUDGE_DIR = ROOT / "artifacts" / "judge"

LABELS = {
    "correct",
    "partially_correct",
    "incorrect",
    "unanswerable_or_bad_context",
}

LETTER_TO_LABEL = {
    "A": "correct",
    "B": "partially_correct",
    "C": "incorrect",
    "D": "unanswerable_or_bad_context",
}

DEFAULT_EXPERIMENTS = [
    "qwen2_5_1_5b_instruct_closed_book",
    "qwen2_5_1_5b_instruct_oracle",
    "qwen2_5_1_5b_instruct_top1_mixture",
    "qwen2_5_1_5b_instruct_top1_dense",
    "qwen2_5_1_5b_instruct_top5_mixture",
    "qwen2_5_1_5b_instruct_top5_dense",
    "qwen2_5_1_5b_instruct_mirage_mixed",
    "roberta_oracle_full",
    "roberta_top1_mixture_full",
    "roberta_top1_dense_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge over HA5 predictions.")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-per-experiment", type=int)
    parser.add_argument("--output", type=Path, default=JUDGE_DIR / "llm_judge_qwen2_5_1_5b_v2.jsonl")
    parser.add_argument("--summary", type=Path, default=JUDGE_DIR / "llm_judge_qwen2_5_1_5b_v2_summary.json")
    parser.add_argument("--progress", type=Path, default=JUDGE_DIR / "llm_judge_qwen2_5_1_5b_v2.progress.json")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def row_key(row: dict[str, Any]) -> str:
    return f"{row['experiment_id']}::{row['query_id']}"


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            experiment_id = row.get("experiment_id")
            query_id = row.get("query_id")
            label = row.get("judge_label")
            if experiment_id and query_id and label in LABELS:
                keys.add(f"{experiment_id}::{query_id}")
    return keys


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_prompt(row: dict[str, Any]) -> str:
    golds = row.get("gold_answers", [])
    if isinstance(golds, str):
        golds = [golds]
    gold_text = "; ".join(str(gold) for gold in golds[:8])
    prediction = str(row.get("prediction", "")).strip()
    question = str(row.get("question", "")).strip()
    return (
        "Grade a question-answering prediction against gold answers.\n"
        "Return one letter only.\n\n"
        "A = prediction is semantically equivalent to a gold answer.\n"
        "B = prediction contains the gold answer but has extra words, or is incomplete but useful.\n"
        "C = prediction is a different/wrong answer.\n"
        "D = prediction is empty, refuses, says unknown, or is not an answer.\n\n"
        "Examples:\n"
        "Question: Capital of France?\nGold answers: Paris\nPrediction: Paris\nLabel: A\n\n"
        "Question: Who wrote Hamlet?\nGold answers: William Shakespeare\nPrediction: Shakespeare\nLabel: A\n\n"
        "Question: Who wrote Hamlet?\nGold answers: William Shakespeare\nPrediction: Shakespeare wrote plays\nLabel: B\n\n"
        "Question: Capital of France?\nGold answers: Paris\nPrediction: London\nLabel: C\n\n"
        "Question: Capital of France?\nGold answers: Paris\nPrediction:\nLabel: D\n\n"
        "Now grade this case.\n"
        f"Question: {question}\n"
        f"Gold answers: {gold_text}\n"
        f"Prediction: {prediction}\n\n"
        "Label:"
    )


def apply_chat_template(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def parse_label(text: str) -> str:
    compact = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", compact)
    if match:
        return LETTER_TO_LABEL[match.group(1)]
    normalized = text.strip().lower()
    if not normalized:
        return "unanswerable_or_bad_context"
    normalized = re.sub(r"[^a-z_]+", " ", normalized)
    for label in LABELS:
        if label in normalized:
            return label
    return "incorrect"


def load_tasks(experiments: list[str], limit_per_experiment: int | None) -> list[dict[str, Any]]:
    tasks = []
    missing = []
    for experiment in experiments:
        path = PRED_DIR / f"{experiment}.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        tasks.extend(read_jsonl(path, limit=limit_per_experiment))
    if missing:
        print("missing prediction files:", *missing, sep="\n", flush=True)
    return tasks


def summarize(path: Path, summary_path: Path) -> None:
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row["experiment_id"])][str(row["judge_label"])] += 1
            total += 1

    payload = {
        "judge_file": str(path),
        "total": total,
        "by_experiment": {
            experiment: {
                "count": sum(counter.values()),
                **{label: counter.get(label, 0) for label in sorted(LABELS)},
                "correct_rate": counter.get("correct", 0) / max(sum(counter.values()), 1),
                "correct_or_partial_rate": (
                    counter.get("correct", 0) + counter.get("partially_correct", 0)
                )
                / max(sum(counter.values()), 1),
            }
            for experiment, counter in sorted(counts.items())
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.experiments, args.limit_per_experiment)
    done = existing_keys(args.output)
    todo = [row for row in tasks if row_key(row) not in done]
    print(f"judge: tasks={len(tasks)} done={len(done)} todo={len(todo)} output={args.output}", flush=True)
    write_progress(
        args.progress,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "device": device,
            "tasks_total": len(tasks),
            "tasks_done_before_start": len(done),
            "tasks_remaining": len(todo),
            "output": str(args.output),
        },
    )
    if not todo:
        summarize(args.output, args.summary)
        write_progress(
            args.progress,
            {
                "status": "complete",
                "updated_at_utc": utc_now(),
                "model": args.model,
                "device": device,
                "tasks_total": len(tasks),
                "tasks_completed": len(done),
                "tasks_remaining": 0,
                "percent": 100.0,
            },
        )
        return

    print(f"judge: loading model {args.model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    print("judge: model loaded", flush=True)

    processed = 0
    started_at = time.monotonic()
    with args.output.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batched(todo, args.batch_size), start=1):
            prompts = [apply_chat_template(tokenizer, build_prompt(row)) for row in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)
            input_len = encoded["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row, sequence in zip(batch, outputs, strict=True):
                raw = tokenizer.decode(sequence[input_len:], skip_special_tokens=True)
                label = parse_label(raw)
                record = {
                    "experiment_id": row["experiment_id"],
                    "query_id": row["query_id"],
                    "source": row.get("source"),
                    "question": row.get("question"),
                    "gold_answers": row.get("gold_answers", row.get("answers", [])),
                    "prediction": row.get("prediction", ""),
                    "context_mode": row.get("context_mode"),
                    "context_source": row.get("context_source"),
                    "judge_label": label,
                    "judge_raw": raw,
                    "judge_model": args.model,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
            handle.flush()
            os.fsync(handle.fileno())

            completed = len(done) + processed
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = processed / elapsed
            remaining = max(len(tasks) - completed, 0)
            percent = 100.0 * completed / len(tasks) if tasks else 100.0
            write_progress(
                args.progress,
                {
                    "status": "running",
                    "updated_at_utc": utc_now(),
                    "model": args.model,
                    "device": device,
                    "tasks_total": len(tasks),
                    "tasks_done_before_start": len(done),
                    "tasks_processed_this_run": processed,
                    "tasks_completed": completed,
                    "tasks_remaining": remaining,
                    "percent": percent,
                    "batch_index": batch_index,
                    "batch_size": args.batch_size,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": rate,
                    "eta_seconds": remaining / rate if rate > 0 else None,
                },
            )
            if batch_index == 1 or processed % 200 == 0 or completed == len(tasks):
                print(f"judge: completed={completed}/{len(tasks)} ({percent:.1f}%) rate={rate:.3f}/s", flush=True)

    summarize(args.output, args.summary)
    final_done = existing_keys(args.output)
    write_progress(
        args.progress,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "device": device,
            "tasks_total": len(tasks),
            "tasks_completed": len(final_done),
            "tasks_remaining": max(len(tasks) - len(final_done), 0),
            "percent": 100.0 * len(final_done) / len(tasks) if tasks else 100.0,
            "elapsed_seconds": time.monotonic() - started_at,
        },
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
