from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "artifacts" / "tables"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
BASELINE_RUNS = [
    "tfidf_original_test",
    "tfidf_stemmed_test",
    "tfidf_lemmatized_test",
    "bm25_original_test_k1_1.5_b_0.75",
    "bm25_stemmed_test_k1_1.5_b_0.75",
    "bm25_lemmatized_test_k1_1.5_b_0.75",
]


def save_query_length_histogram() -> None:
    df = pd.read_csv(TABLES_DIR / "test_queries_audit.csv")
    plt.figure(figsize=(7, 4))
    bins = range(int(df["length_words"].min()), int(df["length_words"].max()) + 2)
    plt.hist(df["length_words"], bins=bins, color="#386cb0", edgecolor="white", align="left")
    plt.title("Test Query Length Distribution")
    plt.xlabel("Query length (words)")
    plt.ylabel("Count")
    plt.xticks(sorted(df["length_words"].unique()))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_query_length_distribution.png", dpi=160)
    plt.close()


def save_relevant_docs_histogram() -> None:
    df = pd.read_csv(TABLES_DIR / "test_qrels_per_query.csv")
    plt.figure(figsize=(7, 4))
    plt.hist(df["relevant_docs"], bins=20, color="#4daf4a", edgecolor="white")
    plt.title("Relevant Documents per Test Query")
    plt.xlabel("Number of relevant documents")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_relevant_docs_distribution.png", dpi=160)
    plt.close()


def save_metric_comparison() -> None:
    df = pd.read_csv(TABLES_DIR / "experiment_summary.csv")
    df = df[df["run_id"].isin(BASELINE_RUNS)].copy()
    df["label"] = df["method"].str.upper() + "\n" + df["variant"]

    metrics = ["P@1", "AP", "nDCG@20"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a", "#e6ab02"]

    for ax, metric in zip(axes, metrics, strict=True):
        ax.bar(df["label"], df[metric], color=colors[: len(df)])
        ax.set_title(metric)
        ax.set_ylim(0, max(df[metric]) * 1.15)
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Baseline Retrieval Comparison on WikIR test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_metric_comparison.png", dpi=160)
    plt.close(fig)


def save_timing_comparison() -> None:
    df = pd.read_csv(TABLES_DIR / "experiment_summary.csv")
    df = df[df["run_id"].isin(BASELINE_RUNS)].copy()
    df["label"] = df["method"].str.upper() + "\n" + df["variant"]

    plt.figure(figsize=(8, 4))
    plt.bar(df["label"], df["avg_query_seconds"], color="#a6761d")
    plt.title("Average Query Time by Method and Text Variant")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_avg_query_time.png", dpi=160)
    plt.close()


def save_query_length_effects() -> None:
    df = pd.read_csv(TABLES_DIR / "query_length_effects.csv")
    selected = df[df["run_id"].isin(
        [
            "tfidf_original_test",
            "tfidf_stemmed_test",
            "bm25_original_test_k1_1.5_b_0.75",
            "bm25_stemmed_test_k1_1.5_b_0.75",
        ]
    )].copy()
    label_map = {
        "tfidf_original_test": "TF-IDF original",
        "tfidf_stemmed_test": "TF-IDF stemmed",
        "bm25_original_test_k1_1.5_b_0.75": "BM25 original",
        "bm25_stemmed_test_k1_1.5_b_0.75": "BM25 stemmed",
    }
    selected["label"] = selected["run_id"].map(label_map)

    pivot = selected.pivot(index="length_bucket", columns="label", values="AP")
    pivot = pivot.reindex(["1 word", "2 words", "3+ words"])
    ax = pivot.plot(kind="bar", figsize=(9, 4), color=["#1b9e77", "#d95f02", "#7570b3", "#66a61e"])
    ax.set_title("AP by Query Length Bucket")
    ax.set_xlabel("Query length bucket")
    ax.set_ylabel("Average AP")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "query_length_effects_ap.png", dpi=160)
    plt.close()


def save_relevant_docs_effects() -> None:
    df = pd.read_csv(TABLES_DIR / "relevant_docs_effects.csv")
    selected = df[df["run_id"].isin(
        [
            "tfidf_original_test",
            "tfidf_stemmed_test",
            "bm25_original_test_k1_1.5_b_0.75",
            "bm25_stemmed_test_k1_1.5_b_0.75",
        ]
    )].copy()
    label_map = {
        "tfidf_original_test": "TF-IDF original",
        "tfidf_stemmed_test": "TF-IDF stemmed",
        "bm25_original_test_k1_1.5_b_0.75": "BM25 original",
        "bm25_stemmed_test_k1_1.5_b_0.75": "BM25 stemmed",
    }
    selected["label"] = selected["run_id"].map(label_map)

    pivot = selected.pivot(index="relevant_docs_bucket", columns="label", values="AP")
    pivot = pivot.reindex(["<=10", "11-20", "21+"])
    ax = pivot.plot(kind="bar", figsize=(9, 4), color=["#1b9e77", "#d95f02", "#7570b3", "#66a61e"])
    ax.set_title("AP by Number of Relevant Documents")
    ax.set_xlabel("Relevant documents per query")
    ax.set_ylabel("Average AP")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "relevant_docs_effects_ap.png", dpi=160)
    plt.close()


def save_tuning_heatmap(variant: str) -> None:
    path = PROJECT_ROOT / "artifacts" / "bm25_tuning" / f"bm25_tuning_{variant}_validation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pivot = df.pivot(index="b", columns="k1", values="AP").sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])
    ax.set_xlabel("k1")
    ax.set_ylabel("b")
    ax.set_title(f"BM25 tuning on validation ({variant})\nAP")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"bm25_tuning_{variant}_ap_heatmap.png", dpi=160)
    plt.close(fig)


def save_default_vs_tuned_bm25() -> None:
    df = pd.read_csv(TABLES_DIR / "experiment_summary.csv")
    wanted = [
        "bm25_original_test_k1_1.5_b_0.75",
        "bm25_original_test_k1_1.2_b_0.75",
        "bm25_stemmed_test_k1_1.5_b_0.75",
        "bm25_stemmed_test_k1_1.2_b_0.75",
        "bm25_lemmatized_test_k1_1.5_b_0.75",
        "bm25_lemmatized_test_k1_1.2_b_0.75",
    ]
    df = df[df["run_id"].isin(wanted)].copy()
    df = df.dropna(subset=["AP", "nDCG@20"])
    variant_order = ["original", "stemmed", "lemmatized"]
    metrics = ["AP", "nDCG@20"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"default": "#7570b3", "tuned": "#d95f02"}
    width = 0.35

    for ax, metric in zip(axes, metrics, strict=True):
        for offset, tune_label in [(-width / 2, "default"), (width / 2, "tuned")]:
            values = []
            for variant in variant_order:
                run_id = (
                    f"bm25_{variant}_test_k1_1.5_b_0.75"
                    if tune_label == "default"
                    else f"bm25_{variant}_test_k1_1.2_b_0.75"
                )
                value = df.loc[df["run_id"] == run_id, metric].iloc[0]
                values.append(value)
            xs = list(range(len(variant_order)))
            ax.bar([x + offset for x in xs], values, width=width, label=tune_label, color=colors[tune_label])
        ax.set_xticks(range(len(variant_order)))
        ax.set_xticklabels(variant_order)
        ax.set_title(metric)
        ax.set_ylim(0, max(df[metric]) * 1.15)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Default vs tuned BM25 on test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bm25_default_vs_tuned.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_query_length_histogram()
    save_relevant_docs_histogram()
    save_metric_comparison()
    save_timing_comparison()
    save_query_length_effects()
    save_relevant_docs_effects()
    save_tuning_heatmap("original")
    save_tuning_heatmap("stemmed")
    save_tuning_heatmap("lemmatized")
    save_default_vs_tuned_bm25()
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
