from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


DZ5_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DZ5_ROOT / "artifacts" / "data" / "mirage_eval_sample_1000.jsonl"
PREDICTIONS_DIR = DZ5_ROOT / "artifacts" / "predictions"

EXPERIMENTS = {
    "roberta_oracle": {"context_mode": "oracle", "context_source": "mirage"},
    "roberta_top1_mixture": {"context_mode": "top1", "context_source": "ha4_mixture"},
    "roberta_top1_dense": {"context_mode": "top1", "context_source": "ha4_dense"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractive QA experiments for HA5.")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="deepset/roberta-base-squad2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-expected-rows", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-len", type=int, default=32)
    parser.add_argument("--n-best-size", type=int, default=20)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-safetensors", action="store_true")
    return parser.parse_args()


def detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def compact_jsonl(path: Path) -> None:
    if not path.exists():
        return
    by_qid: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(record.get("query_id", "")).strip()
            if not qid:
                continue
            if qid not in by_qid:
                order.append(qid)
            by_qid[qid] = record
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for qid in order:
            handle.write(json.dumps(by_qid[qid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def existing_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(record.get("query_id", "")).strip()
            if qid:
                done.add(qid)
    return done


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def context_for(row: dict[str, Any], config: dict[str, str]) -> tuple[str, list[str]]:
    mode = config["context_mode"]
    source = config["context_source"]
    if mode == "oracle":
        return str(row["oracle_context"]), [str(row["oracle_doc_id"])]
    ranker = row.get("ranker_contexts", {}).get(source, {})
    docs = ranker.get("top1", [])
    if not docs:
        return "", []
    return str(docs[0].get("text", "")), [str(docs[0].get("doc_id", ""))]


def best_span(
    *,
    question: str,
    context: str,
    tokenizer,
    model,
    device: str,
    max_seq_len: int,
    doc_stride: int,
    max_answer_len: int,
    n_best_size: int,
) -> dict[str, Any]:
    if not context.strip():
        return {"answer": "", "score": float("-inf"), "start": None, "end": None}

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    sequence_ids = [encoded.sequence_ids(i) for i in range(encoded["input_ids"].shape[0])]
    model_inputs = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)
    start_logits = outputs.start_logits.detach().cpu().numpy()
    end_logits = outputs.end_logits.detach().cpu().numpy()
    offsets_np = offsets.numpy()

    best = {"answer": "", "score": float("-inf"), "start": None, "end": None}
    for feature_idx in range(start_logits.shape[0]):
        start_indexes = np.argsort(start_logits[feature_idx])[-n_best_size:][::-1]
        end_indexes = np.argsort(end_logits[feature_idx])[-n_best_size:][::-1]
        context_token = np.array([sid == 1 for sid in sequence_ids[feature_idx]])
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > max_answer_len:
                    continue
                if start_idx >= len(context_token) or end_idx >= len(context_token):
                    continue
                if not context_token[start_idx] or not context_token[end_idx]:
                    continue
                start_char, _ = offsets_np[feature_idx][start_idx]
                _, end_char = offsets_np[feature_idx][end_idx]
                if end_char <= start_char:
                    continue
                score = float(start_logits[feature_idx][start_idx] + end_logits[feature_idx][end_idx])
                if score > best["score"]:
                    best = {
                        "answer": context[int(start_char) : int(end_char)].strip(),
                        "score": score,
                        "start": int(start_char),
                        "end": int(end_char),
                    }
    return best


def main() -> None:
    args = parse_args()
    config = EXPERIMENTS[args.experiment]
    rows = read_jsonl(args.input, limit=args.limit)
    if args.min_expected_rows and len(rows) < args.min_expected_rows:
        raise SystemExit(f"Loaded {len(rows)} rows, expected at least {args.min_expected_rows}")
    device = detect_device(args.device)

    output = args.output or (PREDICTIONS_DIR / f"{args.experiment}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress or output.with_suffix(".progress.json")
    compact_jsonl(output)
    done = existing_qids(output)
    todo = [row for row in rows if str(row["query_id"]) not in done]
    print(f"{args.experiment}: rows={len(rows)} done={len(done)} todo={len(todo)} output={output}", flush=True)
    write_progress(
        progress_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "experiment_id": args.experiment,
            "model": args.model,
            "device": device,
            "output": str(output),
            "rows_total": len(rows),
            "rows_done_before_start": len(done),
            "rows_completed": len(done),
            "rows_remaining": len(todo),
            "max_seq_len": args.max_seq_len,
            "doc_stride": args.doc_stride,
        },
    )
    if not todo:
        write_progress(
            progress_path,
            {
                "status": "complete",
                "updated_at_utc": utc_now(),
                "experiment_id": args.experiment,
                "model": args.model,
                "device": device,
                "output": str(output),
                "rows_total": len(rows),
                "rows_completed": len(done),
                "rows_remaining": 0,
                "percent": 100.0,
                "elapsed_seconds": 0.0,
            },
        )
        return

    print(f"{args.experiment}: loading model {args.model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        use_safetensors=args.use_safetensors,
    ).to(device)
    model.eval()
    print(f"{args.experiment}: model loaded", flush=True)

    processed = 0
    started_at = time.monotonic()
    with output.open("a", encoding="utf-8") as handle:
        for row in todo:
            context, doc_ids = context_for(row, config)
            result = best_span(
                question=str(row["question"]),
                context=context,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_seq_len=args.max_seq_len,
                doc_stride=args.doc_stride,
                max_answer_len=args.max_answer_len,
                n_best_size=args.n_best_size,
            )
            record = {
                "experiment_id": args.experiment,
                "query_id": row["query_id"],
                "source": row["source"],
                "question": row["question"],
                "gold_answers": row["answers"],
                "context_mode": config["context_mode"],
                "context_source": config["context_source"],
                "context_doc_ids": doc_ids,
                "prediction": result["answer"],
                "raw_output": result,
                "model": args.model,
                "inference_config": {
                    "max_seq_len": args.max_seq_len,
                    "doc_stride": args.doc_stride,
                    "max_answer_len": args.max_answer_len,
                    "n_best_size": args.n_best_size,
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
            percent = 100.0 * completed / len(rows) if rows else 100.0
            write_progress(
                progress_path,
                {
                    "status": "running",
                    "updated_at_utc": utc_now(),
                    "experiment_id": args.experiment,
                    "model": args.model,
                    "device": device,
                    "output": str(output),
                    "rows_total": len(rows),
                    "rows_done_before_start": len(done),
                    "rows_processed_this_run": processed,
                    "rows_completed": completed,
                    "rows_remaining": remaining,
                    "percent": percent,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": rows_per_second,
                    "eta_seconds": remaining / rows_per_second if rows_per_second > 0 else None,
                    "max_seq_len": args.max_seq_len,
                    "doc_stride": args.doc_stride,
                },
            )
            if processed == 1 or processed % 25 == 0 or completed == len(rows):
                print(
                    f"{args.experiment}: completed={completed}/{len(rows)} "
                    f"({percent:.1f}%) rate={rows_per_second:.3f} rows/s",
                    flush=True,
                )

    compact_jsonl(output)
    final_done = existing_qids(output)
    elapsed = time.monotonic() - started_at
    write_progress(
        progress_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "experiment_id": args.experiment,
            "model": args.model,
            "device": device,
            "output": str(output),
            "rows_total": len(rows),
            "rows_completed": len(final_done),
            "rows_remaining": max(len(rows) - len(final_done), 0),
            "percent": 100.0 * len(final_done) / len(rows) if rows else 100.0,
            "elapsed_seconds": elapsed,
        },
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
