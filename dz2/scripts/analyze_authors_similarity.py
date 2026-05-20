from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "authors_search"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


def unique_pair_key(query_id: str, doc_id: str) -> str:
    return "||".join(sorted([query_id, doc_id]))


def save_paragraph_length_plot(paragraphs: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    for author, color in [("tolstoy", "#1b9e77"), ("dostoevsky", "#d95f02")]:
        subset = paragraphs[paragraphs["author"] == author]
        plt.hist(
            subset["token_count"],
            bins=30,
            alpha=0.55,
            label=author,
            color=color,
            edgecolor="white",
        )
    plt.title("Распределение длины абзацев")
    plt.xlabel("Токены")
    plt.ylabel("Число абзацев")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paragraph_length_distribution.png", dpi=160)
    plt.close()


def save_top1_score_plot(summary: pd.DataFrame) -> None:
    labels = summary["setup"] + "\n" + summary["method"] + "\n" + summary["idf_scheme"]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, summary["mean_top1_score"], color="#7570b3")
    plt.title("Средняя оценка лучшего совпадения")
    plt.ylabel("Средняя оценка top-1")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top1_score_comparison.png", dpi=160)
    plt.close()


def save_idf_plot(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="setup_method", columns="idf_scheme", values="mean_top1_score")
    ax = pivot.plot(kind="bar", figsize=(10, 4), color=["#66a61e", "#e7298a"])
    ax.set_title("Сравнение local и global IDF")
    ax.set_xlabel("Постановка / метод")
    ax.set_ylabel("Средняя оценка top-1")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "idf_scheme_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paragraphs = pd.read_csv(ARTIFACTS_DIR / "paragraphs.csv")
    matches = pd.read_csv(ARTIFACTS_DIR / "similarity_matches.csv")

    top1 = matches[matches["rank"] == 1].copy()
    top1["setup_method"] = top1["setup"] + " / " + top1["method"]

    summary = (
        top1.groupby(["setup", "method", "idf_scheme"], as_index=False)
        .agg(
            mean_top1_score=("score", "mean"),
            median_top1_score=("score", "median"),
            p90_top1_score=("score", lambda s: s.quantile(0.9)),
        )
        .sort_values(["setup", "method", "idf_scheme"])
    )
    summary["setup_method"] = summary["setup"] + " / " + summary["method"]
    summary.to_csv(ARTIFACTS_DIR / "similarity_summary.csv", index=False)

    idf_comparison = (
        summary.pivot_table(
            index=["setup", "method"],
            columns="idf_scheme",
            values=["mean_top1_score", "median_top1_score", "p90_top1_score"],
        )
        .reset_index()
    )
    idf_comparison.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_") for col in idf_comparison.columns.to_flat_index()
    ]
    idf_comparison.to_csv(ARTIFACTS_DIR / "idf_scheme_comparison.csv", index=False)

    for setup in sorted(matches["setup"].unique()):
        setup_df = matches[(matches["setup"] == setup) & (matches["rank"] == 1)].copy()
        within_book = setup.startswith("within_")
        if within_book:
            setup_df["pair_key"] = setup_df.apply(lambda row: unique_pair_key(row["query_id"], row["doc_id"]), axis=1)
            setup_df = setup_df.sort_values("score", ascending=False).drop_duplicates(
                subset=["method", "idf_scheme", "pair_key"]
            )
        top_rows = (
            setup_df.sort_values(["method", "idf_scheme", "score"], ascending=[True, True, False])
            .groupby(["method", "idf_scheme"], as_index=False, group_keys=False)
            .head(10)
        )
        top_rows.to_csv(ARTIFACTS_DIR / f"top_pairs_{setup}.csv", index=False)

    save_paragraph_length_plot(paragraphs)
    save_top1_score_plot(summary)
    save_idf_plot(summary)
    print(ARTIFACTS_DIR / "similarity_summary.csv")


if __name__ == "__main__":
    main()
