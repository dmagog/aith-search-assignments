from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ir_measures
from ir_measures import AP, P, nDCG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIKIR_ROOT = PROJECT_ROOT.parent / "dz1" / "wikIR1k" / "wikIR1k"
DEFAULT_MEASURES = [P@1, P@10, P@20, AP, nDCG@20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Absolute path to a TREC run file.")
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    return parser.parse_args()


def ensure_dirs() -> None:
    for rel in [
        "artifacts/tables",
        "artifacts/qrels",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    run_path = Path(args.run)
    run_id = run_path.stem
    qrels_path = WIKIR_ROOT / args.split / "qrels"

    qrels = ir_measures.read_trec_qrels(str(qrels_path))
    run = ir_measures.read_trec_run(str(run_path))

    aggregate = ir_measures.calc_aggregate(DEFAULT_MEASURES, qrels, run)

    qrels = ir_measures.read_trec_qrels(str(qrels_path))
    run = ir_measures.read_trec_run(str(run_path))
    per_query = list(ir_measures.iter_calc(DEFAULT_MEASURES, qrels, run))

    aggregate_csv = PROJECT_ROOT / "artifacts" / "tables" / f"{run_id}_aggregate_metrics.csv"
    per_query_csv = PROJECT_ROOT / "artifacts" / "tables" / f"{run_id}_per_query_metrics.csv"
    summary_json = PROJECT_ROOT / "artifacts" / "tables" / f"{run_id}_metrics_summary.json"

    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "measure", "value"])
        writer.writeheader()
        for measure, value in aggregate.items():
            writer.writerow({"run_id": run_id, "measure": str(measure), "value": value})

    with per_query_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "measure", "value"])
        writer.writeheader()
        for metric in per_query:
            writer.writerow(
                {
                    "query_id": metric.query_id,
                    "measure": str(metric.measure),
                    "value": metric.value,
                }
            )

    summary = {
        "run_id": run_id,
        "split": args.split,
        "qrels_path": str(qrels_path),
        "run_path": str(run_path),
        "measures": [str(m) for m in DEFAULT_MEASURES],
        "aggregate": {str(measure): value for measure, value in aggregate.items()},
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
