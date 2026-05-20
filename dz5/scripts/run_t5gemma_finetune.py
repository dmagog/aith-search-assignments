from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "models" / "t5gemma_squad_lora"
DEFAULT_STATE_PATH = ROOT / "artifacts" / "logs" / "t5gemma_finetune_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5Gemma 2 on SQuAD with LoRA.")
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_input(question: str, context: str) -> str:
    return (
        "Answer the question with a short factual answer.\n\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        "Answer:"
    )


def preprocess_dataset(dataset, tokenizer, *, max_source_length: int, max_target_length: int):
    def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = [build_input(question, context) for question, context in zip(batch["question"], batch["context"], strict=True)]
        targets = []
        for answers in batch["answers"]:
            texts = answers.get("text", []) if isinstance(answers, dict) else []
            targets.append(str(texts[0]) if texts else "")

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing SQuAD",
    )


def choose_lora_targets(model: torch.nn.Module) -> list[str]:
    suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in suffixes:
            found.add(leaf)
    if found:
        return sorted(found)

    fallback_suffixes = ["query", "key", "value", "dense"]
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in fallback_suffixes:
            found.add(leaf)
    if not found:
        raise RuntimeError("Could not infer LoRA target modules from model linear layers.")
    return sorted(found)


def latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints)[1])


class StateCallback(TrainerCallback):
    def __init__(self, path: Path, base_payload: dict[str, Any]) -> None:
        self.path = path
        self.base_payload = base_payload

    def on_log(self, args, state, control, logs=None, **kwargs):
        write_state(
            self.path,
            {
                **self.base_payload,
                "status": "training",
                "updated_at_utc": utc_now(),
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": state.epoch,
                "log": logs or {},
            },
        )

    def on_save(self, args, state, control, **kwargs):
        write_state(
            self.path,
            {
                **self.base_payload,
                "status": "checkpoint_saved",
                "updated_at_utc": utc_now(),
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": state.epoch,
                "checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}"),
            },
        )


def training_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "predict_with_generate": False,
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": args.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
    return kwargs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_state(
        args.state_path,
        {
            "status": "starting",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    write_state(
        args.state_path,
        {
            "status": "tokenizer_loaded",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
        },
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    write_state(
        args.state_path,
        {
            "status": "base_model_loaded",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "dtype": str(dtype),
            "cuda": torch.cuda.is_available(),
        },
    )

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = choose_lora_targets(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    write_state(
        args.state_path,
        {
            "status": "lora_attached",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "lora_target_modules": target_modules,
        },
    )

    raw = load_dataset("squad")
    train_dataset = raw["train"]
    eval_dataset = raw["validation"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    train_tokenized = preprocess_dataset(
        train_dataset,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    eval_tokenized = preprocess_dataset(
        eval_dataset,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    write_state(
        args.state_path,
        {
            "status": "datasets_tokenized",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "lora_target_modules": target_modules,
        },
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": Seq2SeqTrainingArguments(**training_args_kwargs(args)),
        "train_dataset": train_tokenized,
        "eval_dataset": eval_tokenized,
        "data_collator": data_collator,
        "callbacks": [
            StateCallback(
                args.state_path,
                {
                    "model": args.model,
                    "output_dir": str(args.output_dir),
                    "train_rows": len(train_dataset),
                    "eval_rows": len(eval_dataset),
                    "lora_target_modules": target_modules,
                },
            )
        ],
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    checkpoint = latest_checkpoint(args.output_dir) if args.resume == "auto" else None
    write_state(
        args.state_path,
        {
            "status": "training",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "resume_from_checkpoint": checkpoint,
            "lora_target_modules": target_modules,
        },
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_state(
        args.state_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "model": args.model,
            "output_dir": str(args.output_dir),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "lora_target_modules": target_modules,
        },
    )


if __name__ == "__main__":
    main()
