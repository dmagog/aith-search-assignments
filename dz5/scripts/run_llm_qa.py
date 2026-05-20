from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DZ5_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DZ5_ROOT / "artifacts" / "data" / "mirage_eval_sample_1000.jsonl"
PREDICTIONS_DIR = DZ5_ROOT / "artifacts" / "predictions"

EXPERIMENTS = {
    "gemma_closed_book": {"context_mode": "none", "context_source": "none"},
    "gemma_oracle": {"context_mode": "oracle", "context_source": "mirage"},
    "gemma_top1_mixture": {"context_mode": "top1", "context_source": "ha4_mixture"},
    "gemma_top1_dense": {"context_mode": "top1", "context_source": "ha4_dense"},
    "gemma_top5_mixture": {"context_mode": "top5", "context_source": "ha4_mixture"},
    "gemma_top5_dense": {"context_mode": "top5", "context_source": "ha4_dense"},
    "gemma_mirage_mixed": {"context_mode": "mixed", "context_source": "mirage"},
}

CONTEXT_SUFFIXES = {
    "closed_book": {"context_mode": "none", "context_source": "none"},
    "oracle": {"context_mode": "oracle", "context_source": "mirage"},
    "top1_mixture": {"context_mode": "top1", "context_source": "ha4_mixture"},
    "top1_dense": {"context_mode": "top1", "context_source": "ha4_dense"},
    "top5_mixture": {"context_mode": "top5", "context_source": "ha4_mixture"},
    "top5_dense": {"context_mode": "top5", "context_source": "ha4_dense"},
    "mirage_mixed": {"context_mode": "mixed", "context_source": "mirage"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generative SLM QA experiments for HA5.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-expected-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--context-budget-chars", type=int, default=6000)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def config_for_experiment(experiment_id: str) -> dict[str, str]:
    if experiment_id in EXPERIMENTS:
        return EXPERIMENTS[experiment_id]
    for suffix, config in CONTEXT_SUFFIXES.items():
        if experiment_id.endswith(f"_{suffix}"):
            return config
    options = ", ".join(sorted(CONTEXT_SUFFIXES))
    raise SystemExit(f"Cannot infer context config from experiment '{experiment_id}'. Known suffixes: {options}")


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
    if mode == "mixed":
        return concat_passages(row.get("mixed_contexts", []), budget_chars)
    ranker = row.get("ranker_contexts", {}).get(source, {})
    key = "top1" if mode == "top1" else "top5"
    return concat_passages(ranker.get(key, []), budget_chars)


def build_prompt(question: str, context: str) -> str:
    if not context.strip():
        return f"Answer the question with a short factual answer. Do not explain.\nQuestion: {question}\nAnswer:"
    return (
        "Use the context to answer the question with a short factual answer. "
        "If the context is insufficient, answer with the best short answer only.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
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


def apply_chat_template(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


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
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(record.get("query_id", "")).strip()
            pred = str(record.get("prediction", "")).strip()
            if not qid or not pred:
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
            pred = str(record.get("prediction", "")).strip()
            if qid and pred:
                done.add(qid)
    return done


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    config = config_for_experiment(args.experiment)
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
    started_at = time.monotonic()
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
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "context_budget_chars": args.context_budget_chars,
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
                "output": str(output),
                "rows_total": len(rows),
                "rows_completed": len(done),
                "rows_remaining": 0,
                "elapsed_seconds": 0.0,
            },
        )
        return

    write_progress(
        progress_path,
        {
            "status": "loading_model",
            "updated_at_utc": utc_now(),
            "experiment_id": args.experiment,
            "model": args.model,
            "device": device,
            "output": str(output),
            "rows_total": len(rows),
            "rows_done_before_start": len(done),
            "rows_completed": len(done),
            "rows_remaining": len(todo),
        },
    )
    print(f"{args.experiment}: loading model {args.model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    print(f"{args.experiment}: model loaded", flush=True)

    processed = 0
    with output.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batched(todo, max(1, args.batch_size)), start=1):
            prompts: list[str] = []
            metadata: list[dict[str, Any]] = []
            for row in batch:
                context, doc_ids = context_for(row, config, args.context_budget_chars)
                prompt = apply_chat_template(tokenizer, build_prompt(str(row["question"]), context))
                prompts.append(prompt)
                metadata.append({"row": row, "context": context, "doc_ids": doc_ids})

            encoded = tokenizer(
                prompts,
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
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_length = encoded["input_ids"].shape[1]
            for idx, sequence in enumerate(outputs):
                generated_ids = sequence[input_length:]
                raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
                row = metadata[idx]["row"]
                record = {
                    "experiment_id": args.experiment,
                    "query_id": row["query_id"],
                    "source": row["source"],
                    "question": row["question"],
                    "gold_answers": row["answers"],
                    "context_mode": config["context_mode"],
                    "context_source": config["context_source"],
                    "context_doc_ids": metadata[idx]["doc_ids"],
                    "prediction": clean_answer(raw),
                    "raw_output": raw,
                    "model": args.model,
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
            eta_seconds = remaining / rows_per_second if rows_per_second > 0 else None
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
                    "batch_index": batch_index,
                    "batch_size": args.batch_size,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": rows_per_second,
                    "eta_seconds": eta_seconds,
                },
            )
            print(
                f"{args.experiment}: batch={batch_index} completed={completed}/{len(rows)} "
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
