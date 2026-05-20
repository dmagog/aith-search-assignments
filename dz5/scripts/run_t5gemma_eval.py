from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEFAULT_INPUT = ROOT / "artifacts" / "data" / "mirage_eval_sample_1000.jsonl"
DEFAULT_ADAPTER_DIR = ROOT / "artifacts" / "models" / "t5gemma_squad_lora"
PRED_DIR = ROOT / "artifacts" / "predictions"
METRICS_DIR = ROOT / "artifacts" / "metrics"
LOG_DIR = ROOT / "artifacts" / "logs"

EXPERIMENTS = {
    "t5gemma2_270m_squad_lora_closed_book": {"context_mode": "none", "context_source": "none"},
    "t5gemma2_270m_squad_lora_oracle": {"context_mode": "oracle", "context_source": "mirage"},
    "t5gemma2_270m_squad_lora_top1_mixture": {"context_mode": "top1", "context_source": "ha4_mixture"},
    "t5gemma2_270m_squad_lora_top1_dense": {"context_mode": "top1", "context_source": "ha4_dense"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned T5Gemma on MIRAGE sample.")
    parser.add_argument("--base-model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prediction-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-budget-chars", type=int, default=5000)
    parser.add_argument("--min-expected-rows", type=int, default=1000)
    parser.add_argument("--bertscore-model", default="bert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default="cuda")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def concat_passages(passages: list[dict[str, Any]], budget_chars: int) -> tuple[str, list[str]]:
    chunks: list[str] = []
    doc_ids: list[str] = []
    used = 0
    for idx, passage in enumerate(passages, start=1):
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        block = f"[{idx}] {title}\n{text}".strip()
        if not block:
            continue
        next_len = len(block) + (2 if chunks else 0)
        if chunks and used + next_len > budget_chars:
            break
        if not chunks and next_len > budget_chars:
            block = block[:budget_chars]
        chunks.append(block)
        doc_ids.append(str(passage.get("doc_id", "")))
        used += next_len
    return "\n\n".join(chunks), doc_ids


def context_for(row: dict[str, Any], config: dict[str, str], budget_chars: int) -> tuple[str, list[str]]:
    mode = config["context_mode"]
    source = config["context_source"]
    if mode == "none":
        return "", []
    if mode == "oracle":
        return str(row["oracle_context"]), [str(row["oracle_doc_id"])]
    ranker = row.get("ranker_contexts", {}).get(source, {})
    return concat_passages(ranker.get("top1", []), budget_chars)


def build_input(question: str, context: str) -> str:
    if context.strip():
        return (
            "Answer the question with a short factual answer.\n\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            "Answer:"
        )
    return (
        "Answer the question with a short factual answer.\n\n"
        f"Question: {question}\n"
        "Context:\n"
        "Answer:"
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    for prefix in ("Answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for sep in ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]:
        first_line = first_line.split(sep)[0].strip()
    return first_line


def batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def compact_jsonl(path: Path) -> None:
    if not path.exists():
        return
    by_qid: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue
            if qid not in by_qid:
                order.append(qid)
            by_qid[qid] = row
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for qid in order:
            handle.write(json.dumps(by_qid[qid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def existing_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    qids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("query_id", "")).strip()
            if qid and str(row.get("prediction", "")).strip():
                qids.add(qid)
    return qids


def run_metric(pred_path: Path, metric_path: Path, args: argparse.Namespace) -> int:
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "evaluate_predictions.py"),
        "--predictions",
        str(pred_path),
        "--output",
        str(metric_path),
    ]
    if args.skip_bertscore:
        command.append("--skip-bertscore")
    else:
        command.extend(
            [
                "--bertscore-model",
                args.bertscore_model,
                "--bertscore-batch-size",
                str(args.bertscore_batch_size),
                "--bertscore-device",
                args.bertscore_device,
            ]
        )
    process = subprocess.run(command, cwd=ROOT)
    return int(process.returncode)


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    rows = read_jsonl(args.input)
    if len(rows) < args.min_expected_rows:
        raise SystemExit(f"Loaded {len(rows)} rows, expected at least {args.min_expected_rows}")
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    unknown = [experiment for experiment in args.experiments if experiment not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}")

    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "t5gemma_eval_state.json"
    write_json(
        state_path,
        {
            "status": "loading_model",
            "updated_at_utc": utc_now(),
            "base_model": args.base_model,
            "adapter_dir": str(args.adapter_dir),
            "device": device,
            "experiments": args.experiments,
            "rows_total": len(rows),
        },
    )

    tokenizer_source = args.adapter_dir if (args.adapter_dir / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    base_model.config.use_cache = True
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    model.eval()

    for experiment_index, experiment in enumerate(args.experiments, start=1):
        config = EXPERIMENTS[experiment]
        pred_path = args.prediction_dir / f"{experiment}.jsonl"
        metric_path = args.metrics_dir / f"{experiment}_metrics.json"
        progress_path = args.prediction_dir / f"{experiment}.progress.json"
        compact_jsonl(pred_path)
        done = existing_qids(pred_path)
        todo = [row for row in rows if str(row["query_id"]) not in done]
        started_at = time.monotonic()
        write_json(
            state_path,
            {
                "status": "running",
                "updated_at_utc": utc_now(),
                "experiment": experiment,
                "experiment_index": experiment_index,
                "experiment_total": len(args.experiments),
                "rows_total": len(rows),
                "rows_done_before_start": len(done),
                "rows_remaining": len(todo),
                "prediction_file": str(pred_path),
                "metrics_file": str(metric_path),
            },
        )

        processed = 0
        with pred_path.open("a", encoding="utf-8") as handle:
            for batch_index, batch in enumerate(batched(todo, max(1, args.batch_size)), start=1):
                inputs: list[str] = []
                metadata: list[dict[str, Any]] = []
                for row in batch:
                    context, doc_ids = context_for(row, config, args.context_budget_chars)
                    inputs.append(build_input(str(row["question"]), context))
                    metadata.append({"row": row, "doc_ids": doc_ids})

                encoded = tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for raw, meta in zip(decoded, metadata, strict=True):
                    row = meta["row"]
                    record = {
                        "experiment_id": experiment,
                        "query_id": row["query_id"],
                        "source": row["source"],
                        "question": row["question"],
                        "gold_answers": row["answers"],
                        "context_mode": config["context_mode"],
                        "context_source": config["context_source"],
                        "context_doc_ids": meta["doc_ids"],
                        "prediction": clean_answer(raw),
                        "raw_output": raw,
                        "model": f"{args.base_model}+{args.adapter_dir.name}",
                        "generation_config": {
                            "max_new_tokens": args.max_new_tokens,
                            "do_sample": False,
                        },
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                handle.flush()
                os.fsync(handle.fileno())

                completed = len(done) + processed
                elapsed = max(time.monotonic() - started_at, 1e-6)
                rows_per_second = processed / elapsed
                remaining = max(len(rows) - completed, 0)
                percent = 100.0 * completed / len(rows)
                progress = {
                    "status": "running",
                    "updated_at_utc": utc_now(),
                    "experiment": experiment,
                    "rows_total": len(rows),
                    "rows_completed": completed,
                    "rows_remaining": remaining,
                    "percent": percent,
                    "batch_index": batch_index,
                    "batch_size": args.batch_size,
                    "rows_per_second": rows_per_second,
                    "eta_seconds": remaining / rows_per_second if rows_per_second > 0 else None,
                }
                write_json(progress_path, progress)
                write_json(state_path, {**progress, "experiment_index": experiment_index, "experiment_total": len(args.experiments)})
                print(f"{experiment}: completed={completed}/{len(rows)} ({percent:.1f}%)", flush=True)

        compact_jsonl(pred_path)
        exit_code = run_metric(pred_path, metric_path, args)
        if exit_code:
            write_json(
                state_path,
                {
                    "status": "failed_metrics",
                    "updated_at_utc": utc_now(),
                    "experiment": experiment,
                    "exit_code": exit_code,
                    "metrics_file": str(metric_path),
                },
            )
            raise SystemExit(exit_code)
        write_json(
            progress_path,
            {
                "status": "complete",
                "updated_at_utc": utc_now(),
                "experiment": experiment,
                "rows_total": len(rows),
                "rows_completed": len(existing_qids(pred_path)),
                "rows_remaining": max(len(rows) - len(existing_qids(pred_path)), 0),
                "metrics_file": str(metric_path),
            },
        )

    write_json(
        state_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "base_model": args.base_model,
            "adapter_dir": str(args.adapter_dir),
            "experiments": args.experiments,
        },
    )


if __name__ == "__main__":
    main()
