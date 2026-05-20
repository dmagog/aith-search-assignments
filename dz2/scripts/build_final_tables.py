from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "artifacts" / "tables"
TUNING_DIR = PROJECT_ROOT / "artifacts" / "bm25_tuning"

BASELINE_RUNS = [
    "tfidf_original_test",
    "tfidf_stemmed_test",
    "tfidf_lemmatized_test",
    "bm25_original_test_k1_1.5_b_0.75",
    "bm25_stemmed_test_k1_1.5_b_0.75",
    "bm25_lemmatized_test_k1_1.5_b_0.75",
]

DEFAULT_TUNED_PAIRS = {
    "original": {
        "default": "bm25_original_test_k1_1.5_b_0.75",
        "tuned": "bm25_original_test_k1_1.2_b_0.75",
    },
    "stemmed": {
        "default": "bm25_stemmed_test_k1_1.5_b_0.75",
        "tuned": "bm25_stemmed_test_k1_1.2_b_0.75",
    },
    "lemmatized": {
        "default": "bm25_lemmatized_test_k1_1.5_b_0.75",
        "tuned": "bm25_lemmatized_test_k1_1.2_b_0.75",
    },
}


def main() -> None:
    summary = pd.read_csv(TABLES_DIR / "experiment_summary.csv")

    baseline = summary[summary["run_id"].isin(BASELINE_RUNS)].copy()
    baseline = baseline[
        [
            "run_id",
            "method",
            "variant",
            "P@1",
            "P@10",
            "P@20",
            "AP",
            "nDCG@20",
            "build_seconds",
            "preprocess_seconds",
            "total_query_seconds",
            "avg_query_seconds",
        ]
    ].sort_values(["method", "variant"])
    baseline.to_csv(TABLES_DIR / "baseline_results_summary.csv", index=False)

    tuning_rows = []
    for variant in ["original", "stemmed", "lemmatized"]:
        path = TUNING_DIR / f"bm25_tuning_{variant}_validation_best.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        best = data["best"]
        tuning_rows.append(
            {
                "variant": variant,
                "optimize": data["optimize"],
                "best_k1": best["k1"],
                "best_b": best["b"],
                "validation_P@1": best["P@1"],
                "validation_P@10": best["P@10"],
                "validation_P@20": best["P@20"],
                "validation_AP": best["AP"],
                "validation_nDCG@20": best["nDCG@20"],
            }
        )
    pd.DataFrame(tuning_rows).to_csv(TABLES_DIR / "bm25_tuning_best_params_summary.csv", index=False)

    compare_rows = []
    for variant, runs in DEFAULT_TUNED_PAIRS.items():
        default_row = summary.loc[summary["run_id"] == runs["default"]].iloc[0]
        tuned_row = summary.loc[summary["run_id"] == runs["tuned"]].iloc[0]
        compare_rows.append(
            {
                "variant": variant,
                "default_run_id": runs["default"],
                "tuned_run_id": runs["tuned"],
                "default_P@1": default_row["P@1"],
                "tuned_P@1": tuned_row["P@1"],
                "delta_P@1": tuned_row["P@1"] - default_row["P@1"],
                "default_P@10": default_row["P@10"],
                "tuned_P@10": tuned_row["P@10"],
                "delta_P@10": tuned_row["P@10"] - default_row["P@10"],
                "default_P@20": default_row["P@20"],
                "tuned_P@20": tuned_row["P@20"],
                "delta_P@20": tuned_row["P@20"] - default_row["P@20"],
                "default_AP": default_row["AP"],
                "tuned_AP": tuned_row["AP"],
                "delta_AP": tuned_row["AP"] - default_row["AP"],
                "default_nDCG@20": default_row["nDCG@20"],
                "tuned_nDCG@20": tuned_row["nDCG@20"],
                "delta_nDCG@20": tuned_row["nDCG@20"] - default_row["nDCG@20"],
            }
        )
    pd.DataFrame(compare_rows).to_csv(TABLES_DIR / "bm25_default_vs_tuned_summary.csv", index=False)

    checklist_rows = [
        {
            "requirement": "Basic statistics for test queries",
            "status": "done",
            "evidence": "report section 4 + test_queries_audit.csv + test_qrels_per_query.csv",
        },
        {
            "requirement": "6 baseline runs for tf-idf/BM25 x original/stemmed/lemmatized",
            "status": "done",
            "evidence": "baseline_results_summary.csv + artifacts/runs/*.trec",
        },
        {
            "requirement": "Execution time estimates",
            "status": "done",
            "evidence": "baseline_results_summary.csv + experiment_summary.csv",
        },
        {
            "requirement": "TREC run format outputs",
            "status": "done",
            "evidence": "artifacts/runs/*.trec",
        },
        {
            "requirement": "Evaluation with P@1, P@10, P@20, MAP, nDCG@20",
            "status": "done",
            "evidence": "aggregate/per-query metric CSVs + report sections 5 and 6",
        },
        {
            "requirement": "Analysis of easy/hard queries and query properties",
            "status": "done",
            "evidence": "report section 7 + per-query analysis CSVs",
        },
        {
            "requirement": "BM25 tuning on validation and comparison on test",
            "status": "done",
            "evidence": "bm25_tuning_best_params_summary.csv + bm25_default_vs_tuned_summary.csv",
        },
    ]
    with (TABLES_DIR / "assignment_coverage_checklist.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["requirement", "status", "evidence"])
        writer.writeheader()
        writer.writerows(checklist_rows)

    print(TABLES_DIR)


if __name__ == "__main__":
    main()
