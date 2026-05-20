from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "artifacts" / "tables"
TIMINGS_DIR = PROJECT_ROOT / "artifacts" / "timings"


def load_aggregate_metrics() -> dict[str, dict[str, float]]:
    by_run: dict[str, dict[str, float]] = {}
    for path in sorted(TABLES_DIR.glob("*_aggregate_metrics.csv")):
        run_id = path.name.removesuffix("_aggregate_metrics.csv")
        metrics: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics[row["measure"]] = float(row["value"])
        by_run[run_id] = metrics
    return by_run


def load_timings() -> dict[str, dict[str, object]]:
    by_run: dict[str, dict[str, object]] = {}
    for path in sorted(TIMINGS_DIR.glob("*_timings.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        by_run[data["run_id"]] = data
    return by_run


def main() -> None:
    metrics_by_run = load_aggregate_metrics()
    timings_by_run = load_timings()

    output_path = TABLES_DIR / "experiment_summary.csv"
    fieldnames = [
        "run_id",
        "method",
        "variant",
        "split",
        "documents",
        "queries",
        "top_k",
        "build_seconds",
        "preprocess_seconds",
        "total_query_seconds",
        "avg_query_seconds",
        "P@1",
        "P@10",
        "P@20",
        "AP",
        "nDCG@20",
    ]

    rows: list[dict[str, object]] = []
    for run_id in sorted(timings_by_run):
        timing = timings_by_run[run_id]
        metrics = metrics_by_run.get(run_id, {})
        rows.append(
            {
                "run_id": run_id,
                "method": timing["method"],
                "variant": timing["variant"],
                "split": timing["split"],
                "documents": timing["documents"],
                "queries": timing["queries"],
                "top_k": timing["top_k"],
                "build_seconds": timing["build_seconds"],
                "preprocess_seconds": timing["preprocess_seconds"],
                "total_query_seconds": timing["total_query_seconds"],
                "avg_query_seconds": timing["avg_query_seconds"],
                "P@1": metrics.get("P@1"),
                "P@10": metrics.get("P@10"),
                "P@20": metrics.get("P@20"),
                "AP": metrics.get("AP"),
                "nDCG@20": metrics.get("nDCG@20"),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)


if __name__ == "__main__":
    main()
