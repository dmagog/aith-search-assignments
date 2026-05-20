from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
from pathlib import Path
from typing import Any

import numpy as np


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HA5 QA predictions.")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="bert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default=None)
    parser.add_argument("--bertscore-local-files-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize_squad(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_mirage(text: str) -> str:
    return str(text).lower()


def squad_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_squad(prediction) == normalize_squad(gold))


def squad_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_squad(prediction).split()
    gold_tokens = normalize_squad(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mirage_strict(prediction: str, gold_answers: list[str]) -> float:
    answer_set = {normalize_mirage(answer) for answer in gold_answers}
    model_prediction = normalize_mirage(prediction)
    return float(any(answer == model_prediction for answer in answer_set))


def mirage_loose(prediction: str, gold_answers: list[str]) -> float:
    answer_set = {normalize_mirage(answer) for answer in gold_answers}
    model_prediction = normalize_mirage(prediction)
    return float(any(answer in model_prediction for answer in answer_set))


def mirage_f1(prediction: str, gold_answers: list[str]) -> float:
    predicted_tokens = set(normalize_mirage(prediction).split())
    answer_tokens = set(token for answer in gold_answers for token in normalize_mirage(answer).split())
    true_positives = len(predicted_tokens.intersection(answer_tokens))
    precision = true_positives / len(predicted_tokens) if predicted_tokens else 0.0
    recall = true_positives / len(answer_tokens) if answer_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def best_over_gold(prediction: str, gold_answers: list[str], scorer) -> float:
    if not gold_answers:
        return 0.0
    return max(float(scorer(prediction, gold)) for gold in gold_answers)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_answers(row: dict[str, Any]) -> list[str]:
    values = row.get("gold_answers", row.get("answers", []))
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if str(value).strip()]


def prediction_text(row: dict[str, Any]) -> str:
    return str(row.get("prediction", row.get("answer", ""))).strip()


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        pred = prediction_text(row)
        golds = gold_answers(row)
        scored.append(
            {
                "query_id": str(row.get("query_id", "")),
                "source": str(row.get("source", "unknown")),
                "experiment_id": str(row.get("experiment_id", "unknown")),
                "squad_em": best_over_gold(pred, golds, squad_exact_match),
                "squad_f1": best_over_gold(pred, golds, squad_f1),
                "mirage_f1": mirage_f1(pred, golds),
                "mirage_em_strict": mirage_strict(pred, golds),
                "mirage_em_loose": mirage_loose(pred, golds),
            }
        )
    return scored


def aggregate(scored: list[dict[str, Any]]) -> dict[str, float]:
    metrics = ["squad_em", "squad_f1", "mirage_f1", "mirage_em_strict", "mirage_em_loose"]
    if not scored:
        return {metric: 0.0 for metric in metrics}
    return {metric: float(np.mean([row[metric] for row in scored])) for metric in metrics}


def aggregate_by_source(scored: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        grouped[row["source"]].append(row)
    return {source: {"count": len(rows), **aggregate(rows)} for source, rows in sorted(grouped.items())}


def bootstrap_ci(scored: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, dict[str, float]]:
    if samples <= 0 or not scored:
        return {}
    rng = np.random.default_rng(seed)
    metrics = ["squad_em", "squad_f1", "mirage_f1", "mirage_em_strict", "mirage_em_loose"]
    values = {metric: [] for metric in metrics}
    n = len(scored)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        for metric in metrics:
            values[metric].append(float(np.mean([scored[i][metric] for i in idx])))
    return {
        metric: {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
        }
        for metric, vals in values.items()
    }


def add_bertscore(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    model_type: str,
    batch_size: int,
    device: str | None,
    local_files_only: bool,
) -> None:
    if local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from bert_score import score as bert_score
    except Exception as exc:  # pragma: no cover - depends on optional package
        payload["bertscore_error"] = f"{type(exc).__name__}: {exc}"
        return

    predictions: list[str] = []
    references: list[str] = []
    row_index: list[int] = []
    for idx, row in enumerate(rows):
        golds = gold_answers(row)
        if not golds:
            continue
        pred = prediction_text(row)
        for gold in golds:
            predictions.append(pred)
            references.append(gold)
            row_index.append(idx)
    if not predictions:
        return

    _, _, f1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
        rescale_with_baseline=False,
    )
    best_by_row: dict[int, float] = {}
    for idx, value in zip(row_index, f1.tolist(), strict=True):
        best_by_row[idx] = max(best_by_row.get(idx, float("-inf")), float(value))
    scores = [best_by_row[idx] for idx in sorted(best_by_row)]
    payload["bertscore_f1"] = float(np.mean(scores)) if scores else 0.0
    payload["bertscore_model"] = model_type
    payload["bertscore_batch_size"] = batch_size
    payload["bertscore_local_files_only"] = local_files_only


def run_self_test() -> None:
    rows = [
        {"query_id": "1", "source": "test", "prediction": "The journalist", "gold_answers": ["journalist"]},
        {"query_id": "2", "source": "test", "prediction": "Paris", "gold_answers": ["Paris"]},
        {"query_id": "3", "source": "test", "prediction": "wrong", "gold_answers": ["right"]},
    ]
    scored = score_rows(rows)
    metrics = aggregate(scored)
    assert metrics["squad_em"] == 2 / 3
    assert metrics["mirage_em_loose"] == 2 / 3
    assert mirage_loose("The journalist", ["journalist"]) == 1.0
    assert mirage_loose("journalist", ["The journalist"]) == 0.0
    assert scored[0]["squad_f1"] == 1.0
    print("self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.predictions is None:
        raise SystemExit("--predictions is required unless --self-test is used")

    rows = read_jsonl(args.predictions)
    scored = score_rows(rows)
    payload: dict[str, Any] = {
        "prediction_file": str(args.predictions),
        "count": len(rows),
        "aggregate": aggregate(scored),
        "by_source": aggregate_by_source(scored),
        "bootstrap_ci": bootstrap_ci(scored, samples=args.bootstrap, seed=args.seed),
        "metric_note": (
            "MIRAGE F1/EM metrics reproduce the official nlpai-lab/MIRAGE LLM_Evaluator formulas "
            "from LLM.py: lowercase-only matching, EM_loose = any gold answer substring in prediction, "
            "EM_strict = exact lowercase string match, F1 over token sets."
        ),
    }
    if not args.skip_bertscore:
        add_bertscore(
            rows,
            payload["aggregate"],
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
            device=args.bertscore_device,
            local_files_only=args.bertscore_local_files_only,
        )

    output = args.output
    if output is None:
        output = args.predictions.parent.parent / "metrics" / f"{args.predictions.stem}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
