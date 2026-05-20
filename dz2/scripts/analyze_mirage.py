from __future__ import annotations

import json
from pathlib import Path

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import AP, P, nDCG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIRAGE_DIR = PROJECT_ROOT / "artifacts" / "mirage"
TABLES_DIR = MIRAGE_DIR / "tables"
FIGURES_DIR = MIRAGE_DIR / "figures"
MEASURES = [P@1, P@10, P@20, AP, nDCG@20]


def evaluate_run(run_path: Path) -> dict[str, float]:
    qrels = ir_measures.read_trec_qrels(str(MIRAGE_DIR / "qrels.trec"))
    run = ir_measures.read_trec_run(str(run_path))
    return {str(k): v for k, v in ir_measures.calc_aggregate(MEASURES, qrels, run).items()}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for summary_path in sorted((MIRAGE_DIR / "timings").glob("*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = evaluate_run(Path(summary["run_path"]))
        rows.append(
            {
                "run_id": summary["run_id"],
                "method": summary["method"],
                "variant": summary["variant"],
                "k1": summary["k1"],
                "b": summary["b"],
                "preprocess_seconds": summary["preprocess_seconds"],
                "build_seconds": summary["build_seconds"],
                "avg_query_seconds": summary["avg_query_seconds"],
                **metrics,
            }
        )

    df = pd.DataFrame(rows).sort_values(["method", "variant"])
    df.to_csv(TABLES_DIR / "mirage_experiment_summary.csv", index=False)

    plt.figure(figsize=(10, 4))
    metrics = ["P@1", "AP", "nDCG@20"]
    x = range(len(df))
    width = 0.25
    for i, metric in enumerate(metrics):
        plt.bar([xi + (i - 1) * width for xi in x], df[metric], width=width, label=metric)
    plt.xticks(list(x), df["method"].str.upper() + "\n" + df["variant"], rotation=25)
    plt.title("Результаты поиска на MIRAGE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_metric_comparison.png", dpi=160)
    plt.close()

    audit = pd.read_csv(TABLES_DIR / "mirage_query_audit.csv")
    plt.figure(figsize=(7, 4))
    plt.hist(audit["query_length_words"], bins=20, color="#386cb0", edgecolor="white")
    plt.title("Распределение длины запросов в MIRAGE")
    plt.xlabel("Слова")
    plt.ylabel("Число запросов")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_query_length_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(audit["relevant_passages"], bins=20, color="#4daf4a", edgecolor="white")
    plt.title("Число релевантных фрагментов на запрос в MIRAGE")
    plt.xlabel("Релевантные фрагменты")
    plt.ylabel("Число запросов")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_relevant_passages_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(df["method"].str.upper() + "\n" + df["variant"], df["avg_query_seconds"], color="#a6761d")
    plt.title("Среднее время ответа на запрос в MIRAGE")
    plt.ylabel("Секунды")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_avg_query_time.png", dpi=160)
    plt.close()

    print(TABLES_DIR / "mirage_experiment_summary.csv")


if __name__ == "__main__":
    main()
